import torch
import logging
import threading
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, TransformerEncoder
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torch.distributed.nn import RemoteModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        S, N = x.size()
        pos = torch.arange(S,
                           dtype=torch.long,
                           device=x.device).unsqueeze(0).expand((N, S)).t()
        return self.pos_embedding(pos)


class TokenTypeEncoding(nn.Module):
    def __init__(self, type_token_num, d_model):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(type_token_num, d_model)

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size()
        if token_type_input is None:
            token_type_input = torch.zeros((S, N),
                                           dtype=torch.long,
                                           device=seq_input.device)
        return self.token_type_embeddings(token_type_input)


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)  # Two sentence type
        self.norm = LayerNorm(ninp)
        self.dropout = Dropout(dropout)

    def forward(self, seq_inputs):
        src, token_type_input = seq_inputs
        src = self.embed(src) + self.pos_embed(src) \
            + self.tok_type_embed(src, token_type_input)
        return self.dropout(self.norm(src))

process_groups_map = {}

class LogRecord:
    seq = 0

    @classmethod
    def init(cls):
        cls.lock = threading.Lock()

def log_print(seq, s):
    #print(f"{dist.get_rank()} {seq} {s}", flush=True)
    return
    LogRecord.writer.write(f"{dist.get_rank()} {seq} {s}\n")
    LogRecord.writer.flush()

class Print(torch.autograd.Function):
    def forward(ctx, msg, x):
        ctx.msg = msg
        #print(f"({dist.get_rank()}): {ctx.msg} (forward)", flush=True)
        return x

    def backward(ctx, gradient):
        #print(f"({dist.get_rank()}): {ctx.msg} (backward) device={gradient.device}", flush=True)
        return None, gradient



class DistributeFunc(torch.autograd.Function):
    def forward(ctx, seq, pg, x):
        ctx.pg = pg
        ctx.seq= seq
        return x

    def backward(ctx, gradient):
        #return None, None, gradient
        log_print(ctx.seq, "DistributeFunc begin")
        gradient.view(-1)[0].item()
        with LogRecord.lock:
            log_print(ctx.seq, "DistributeFunc with lock")
            dist.all_reduce(gradient, group=ctx.pg)
            log_print(ctx.seq, "DistributeFunc after all_reduce")
            gradient.view(-1)[0].item()
        log_print(ctx.seq, "DistributeFunc end")
        return None, None, gradient

class CollectFunc(torch.autograd.Function):
    def forward(ctx, seq, pg, x):
        x.view(-1)[0].item()
        log_print(seq, "CollectFunc begin")
        with LogRecord.lock:
            log_print(seq, "CollectFunc locked")
            dist.all_reduce(x, group=pg)
            x.view(-1)[0].item()
        log_print(seq, "CollectFunc end")
        return x

    def backward(ctx, gradient):
        return None, None, gradient

class OutputCollector(torch.autograd.Function):
    def forward(ctx, *x):
        ctx.num_input = len(x)
        return x[1]

    def backward(ctx, gradient):
        return tuple(gradient*1.0 for i in range(ctx.num_input))

class DummyOutput(torch.autograd.Function):
    def forward(ctx, x):
        return x#torch.zeros([])

    def backward(ctx, gradient):
        return gradient

class OutputCollectorModule(nn.Module):
    def forward(self, *inputs):
        return Print.apply(f"OutputCollector size={len(inputs)}", OutputCollector.apply(*inputs))

class TransformerEncoderLayerShard(nn.Module):
    def __init__(self, pg_name, need_distirbute_input, need_output, d_model, nhead, dim_feedforward,
                 dropout, activation="gelu"):
        super(TransformerEncoderLayerShard, self).__init__()
        try:
            self.pg = process_groups_map[pg_name]
            self.need_distirbute_input = need_distirbute_input
            self.need_output = need_output
            assert d_model % self.pg.size() == 0
            assert self.pg.rank() >= 0
            sharded_d_model = d_model // self.pg.size()
            in_proj_container = InProjContainer(Linear(d_model, sharded_d_model),
                                                Linear(d_model, sharded_d_model),
                                                Linear(d_model, sharded_d_model))
            self.mha = MultiheadAttentionContainer(nhead // self.pg.size(), in_proj_container,
                                                   ScaledDotProduct(), Linear(sharded_d_model, d_model))

            assert dim_feedforward % self.pg.size() == 0
            dim_feedforward = dim_feedforward // self.pg.size()
            self.linear1 = Linear(d_model, dim_feedforward)
            self.dropout = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model)

            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            self.dropout1 = Dropout(dropout)
            self.dropout2 = Dropout(dropout)

            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
            else:
                raise RuntimeError("only relu/gelu are supported, not {}".format(activation))
        except Exception as e:
            print("ERROR:", e)
            raise

    def __getstate__(self):
          return {}

    def init_weights(self):
        self.mha.in_proj_container.query_proj.init_weights()
        self.mha.in_proj_container.key_proj.init_weights()
        self.mha.in_proj_container.value_proj.init_weights()
        self.mha.out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        seq = LogRecord.seq
        LogRecord.seq += 1
        src = Print.apply("TRF start", src)
        if self.need_distirbute_input:
            src = DistributeFunc.apply(seq, self.pg, src)
        attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        attn_output = CollectFunc.apply(seq, self.pg, attn_output)
        
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src = DistributeFunc.apply(seq, self.pg, src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = CollectFunc.apply(seq, self.pg, src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if not self.need_output:
            src = DummyOutput.apply(src)
        return Print.apply("TRF end", src)


class ShardedTransformerEncoderLayer(nn.Module):
    def __init__(self, devices, pg_name, single_input, single_output, d_model, nhead, dim_feedforward, dropout):
        super(ShardedTransformerEncoderLayer, self).__init__()
        self.shards = nn.ModuleList([RemoteModule(device, TransformerEncoderLayerShard, (
            pg_name, not single_input, not single_output or i==0,
            d_model, nhead, dim_feedforward, dropout)) for i, device in enumerate(devices)])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs] * len(self.shards)
        assert len(inputs) == len(self.shards)
        return [shard(input) for shard, input in zip(self.shards, inputs)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        in_proj_container = InProjContainer(Linear(d_model, d_model),
                                            Linear(d_model, d_model),
                                            Linear(d_model, d_model))
        self.mha = MultiheadAttentionContainer(nhead, in_proj_container,
                                               ScaledDotProduct(), Linear(d_model, d_model))
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def init_weights(self):
        self.mha.in_proj_container.query_proj.init_weights()
        self.mha.in_proj_container.key_proj.init_weights()
        self.mha.in_proj_container.value_proj.init_weights()
        self.mha.out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BertModel(nn.Module):
    """Contain a transformer encoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, embed_layer, dropout=0.5):
        super(BertModel, self).__init__()
        self.model_type = 'Transformer'
        self.bert_embed = embed_layer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, seq_inputs):
        src = self.bert_embed(seq_inputs)
        output = self.transformer_encoder(src)
        return output


class MLMTask(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MLMTask, self).__init__()
        embed_layer = BertEmbedding(ntoken, ninp)
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, embed_layer, dropout=dropout)
        self.mlm_span = Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12)
        self.mlm_head = Linear(ninp, ntoken)

    def forward(self, src, token_type_input=None):
        src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model((src, token_type_input))
        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class MLMTask3(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.bert_embed = BertEmbedding(ntoken, ninp)

        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(ninp, nhead, nhid, dropout), nlayers)

        self.mlm_span = Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12)
        self.mlm_head = Linear(ninp, ntoken)

    def forward(self, src, token_type_input=None):
        src = self.bert_embed((src.transpose(0, 1), token_type_input))

        output = self.transformer_encoder(src)

        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class MLMTaskEmbedding(nn.Module):
    def __init__(self, ntoken, ninp):
        super().__init__()
        self.bert_embed = BertEmbedding(ntoken, ninp)

    def forward(self, src, token_type_input=None):
        result = self.bert_embed((src.transpose(0, 1), token_type_input))
        return result


class MLMTaskEncoder(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(ninp, nhead, nhid, dropout), nlayers)

    def forward(self, src):
        return self.transformer_encoder(src)


class MLMTaskHead(nn.Module):
    def __init__(self, ntoken, ninp):
        super().__init__()
        self.mlm_span = Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12)
        self.mlm_head = Linear(ninp, ntoken)

    def forward(self, src):
        output = self.mlm_span(src)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class MLMTask2(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.embed = MLMTaskEmbedding(ntoken, ninp)
        self.encoder = MLMTaskEncoder(ninp, nhead, nhid, nlayers, dropout)
        self.head = MLMTaskHead(ntoken, ninp)

    def forward(self, src, token_type_input=None):
        output = self.embed(src, token_type_input)
        output = self.encoder(output)
        output = self.head(output)
        return output


class NextSentenceTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, bert_model):
        super(NextSentenceTask, self).__init__()
        self.bert_model = bert_model
        self.linear_layer = Linear(bert_model.ninp,
                                   bert_model.ninp)
        self.ns_span = Linear(bert_model.ninp, 2)
        self.activation = nn.Tanh()

    def forward(self, src, token_type_input):
        src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model((src, token_type_input))
        # Send the first <'cls'> seq to a classifier
        output = self.activation(self.linear_layer(output[0]))
        output = self.ns_span(output)
        return output


class QuestionAnswerTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, bert_model):
        super(QuestionAnswerTask, self).__init__()
        self.bert_model = bert_model
        self.activation = F.gelu
        self.qa_span = Linear(bert_model.ninp, 2)

    def forward(self, src, token_type_input):
        output = self.bert_model((src, token_type_input))
        # transpose output (S, N, E) to (N, S, E)
        output = output.transpose(0, 1)
        output = self.activation(output)
        pos_output = self.qa_span(output)
        start_pos, end_pos = pos_output.split(1, dim=-1)
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)
        return start_pos, end_pos
