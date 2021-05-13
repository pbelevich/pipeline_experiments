import argparse
import torch.multiprocessing as mp
import math
import sys
import time
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.nn import RemoteModule
from torch.utils.data import DataLoader
import torch.distributed.rpc as rpc
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd

from .model import MLMTask, MLMTask2, MLMTaskEmbedding, MLMTaskEncoder, MLMTaskHead

from fairscale.experimental.nn.distributed_pipeline import DistributedLoss, DistributedPipeline, PipelineModulesGraph
from fairscale.experimental.nn.distributed_pipeline.trace import make_graph


def collate_batch(batch_data, args, mask_id, cls_id):
    batch_data = torch.tensor(batch_data).long().view(args.batch_size, -1).t().contiguous()
    # Generate masks with args.mask_frac
    data_len = batch_data.size(0)
    ones_num = int(data_len * args.mask_frac)
    zeros_num = data_len - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(data_len)]
    batch_data = torch.cat((torch.tensor([[cls_id] * batch_data.size(1)]).long(), batch_data))
    lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

    #targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0))]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, lm_mask, targets


def process_raw_data(raw_data, args):
    _num = raw_data.size(0) // (args.batch_size * args.bptt)
    raw_data = raw_data[:(_num * args.batch_size * args.bptt)]
    return raw_data


class Loss(nn.Module):
    def __init__(self, criterion, ntokens):
        super().__init__()
        self.ntokens = ntokens
        self.criterion = criterion
        #self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        #print("INPUT:", input.sum().item())
        return self.criterion(input.view(-1, self.ntokens), target.to(input.device))


def run_batch(optimizer, model, loss_module, data, lm_mask, targets):
    with dist_autograd.context() as context_id:
        #data = data.to(0)
        data = data.transpose(0, 1)
        output = model(data)
        #print("OUTPUT:", output.sum().item())
        #output = rpc.RRef(output)
        loss = loss_module(output, rpc.RRef(targets)).to_here()
        #return loss.item()
        dist_autograd.backward(context_id, [loss])
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step(context_id)

    return loss.item()

def train(model, vocab, train_loss_log, train_data,
          optimizer, criterion, ntokens, epoch, args):
    #model.train()
    total_loss = 0.
    start_time = time.time()
    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    train_loss_log.append(0.0)
    dataloader = DataLoader(train_data, batch_size=args.batch_size * args.bptt,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id))

    loss_module = DistributedLoss(Loss, criterion, ntokens)

    for batch, (data, lm_mask, targets) in enumerate(dataloader):
        try:
            loss = run_batch(optimizer, model, loss_module, data, lm_mask, targets)
        except:
            #print(rpc.rpc_sync("w3", torch.cuda.memory_stats, args=(3,)))
            #time.sleep(60)
            raise
        #rpc.rpc_sync(f"w3", torch.cuda.empty_cache)
        print("LOSS:", "%0.3f" % (loss,))
        total_loss += loss

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            train_loss_log[-1] = cur_loss
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(epoch, batch,
                                                        len(train_data) // (args.bptt * args.batch_size),
                                                        args.lr,
                                                        elapsed * 1000 / args.log_interval,
                                                        cur_loss))
            if args.num_batch > 0 and batch >= args.num_batch:
                break
            total_loss = 0
            start_time = time.time()


class NoOp(nn.Module):
    def forward(self, input):
        #import math; print(input.shape,"=",math.prod(input.shape))
        return input

import threading, sys, traceback, signal
def dumpstacks(signal, frame):
    print("\n============================= dumpstacks", os.getpid(), "=============================")
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


def run_main(args):

    import torchtext
    if args.dataset == 'WikiText103':
        from torchtext.experimental.datasets import WikiText103 as WLMDataset
    elif args.dataset == 'WikiText2':
        from torchtext.experimental.datasets import WikiText2 as WLMDataset
    elif args.dataset == 'WMTNewsCrawl':
        from torchtext.experimental.datasets import WMTNewsCrawl as WLMDataset
    elif args.dataset == 'EnWik9':
        from torchtext.datasets import EnWik9
    elif args.dataset == 'BookCorpus':
        from data import BookCorpus
    else:
        print("dataset for MLM task is not supported")

    try:
        vocab = torch.load(args.save_vocab)
    except:
        print(f"WLMDataset = {WLMDataset}")
        train_dataset, valid_dataset, test_dataset = WLMDataset()
        old_vocab = train_dataset.vocab
        print(f"len(old_vocab) = {len(old_vocab)}")
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)

    if args.dataset == 'WikiText103' or args.dataset == 'WikiText2':
        train_dataset, valid_dataset, test_dataset = WLMDataset(vocab=vocab)
        train_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, train_dataset)))
        valid_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, valid_dataset)))
        test_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, test_dataset)))
    elif args.dataset == 'WMTNewsCrawl':
        from torchtext.experimental.datasets import WikiText2
        test_dataset, valid_dataset = WikiText2(vocab=vocab, split=('test', 'valid'))
        valid_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, valid_dataset)))
        test_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, test_dataset)))
        train_dataset = WLMDataset(vocab=vocab, split='train')
        train_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, train_dataset)))
    elif args.dataset == 'EnWik9':
        enwik9 = EnWik9()
        idx1, idx2 = int(len(enwik9) * 0.8), int(len(enwik9) * 0.9)
        train_data = torch.tensor([vocab.stoi[_id]
                                  for _id in enwik9[0:idx1]]).long()
        val_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx1:idx2]]).long()
        test_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx2:]]).long()
        from torchtext.experimental.datasets import LanguageModelingDataset
        train_dataset = LanguageModelingDataset(train_data, vocab, lambda x: x)
        valid_dataset = LanguageModelingDataset(val_data, vocab, lambda x: x)
        test_dataset = LanguageModelingDataset(test_data, vocab, lambda x: x)
    elif args.dataset == 'BookCorpus':
        train_dataset, valid_dataset, test_dataset = BookCorpus(vocab)

    train_data = process_raw_data(train_dataset.data, args)
    val_data = process_raw_data(valid_dataset.data, args)
    test_data = process_raw_data(test_dataset.data, args)

    ntokens = len(train_dataset.get_vocab())
    print(f"Vocabulary size = {ntokens}")

    ngpus = args.world_size // args.num_workers

    layers = [RemoteModule("w0/cuda:0", MLMTaskEmbedding, (ntokens, args.emsize))]
    n_encoders = args.nlayers
    num_parts = min(n_encoders, args.world_size-3)
    for di, device in enumerate([f"w{i}/cuda:{i % ngpus}" for i in range(1, num_parts+1)]):
        this_encoders = n_encoders // (num_parts-di)
        layers.append(RemoteModule(device, MLMTaskEncoder, (args.emsize,  args.nhead, args.nhid, this_encoders, args.dropout)))
        n_encoders -= this_encoders
    layers.append(RemoteModule(f"w{num_parts+1}/cuda:{(num_parts+1) % ngpus}", MLMTaskHead, (ntokens, args.emsize)))
    layers.append(RemoteModule(f"w{num_parts+2}/cuda:{(num_parts+2) % ngpus}", NoOp, ()))
    org_model = nn.Sequential(*layers)

    graph = make_graph(org_model)
    #for node in graph.nodes: print(node.module.on, node.get_name())
    model = DistributedPipeline(graph, chunks=min(args.world_size, args.batch_size))

    params = sum([torch.prod(torch.tensor(p.rpc_sync().size())) for p in model.parameter_rrefs()])
    print(f'Total parameters = {int(params.item() // 1e6)}M')

    criterion = nn.CrossEntropyLoss()
    optimizer = DistributedOptimizer(
            torch.optim.SGD,
            model.parameter_rrefs(),
            lr=args.lr,
        )
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataset.vocab, train_loss_log, train_data,
              optimizer, criterion, ntokens, epoch, args)


def run_worker(rank, args):


    signal.signal(signal.SIGUSR1, dumpstacks) 

    first_rank = (args.world_size // args.num_workers) * int(os.environ.get('SLURM_PROCID', '0'))
    rank += first_rank

    if rank==first_rank:
        print("rank:", rank, "pid", os.getpid())
    torch.cuda.set_per_process_memory_fraction(0.9, rank - first_rank)
    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", '29500')
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=3600)
    for i in range(args.world_size):
        options.set_device_map(f"w{i}", {rank - first_rank: i % (args.world_size // args.num_workers)})
    rpc.init_rpc(
        f"w{rank}",
        rank=rank,
        world_size=args.world_size,
        rpc_backend_options=options
    )

    if rank == 0:
        run_main(args)

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline experiments')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=5431916812,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    # parser.add_argument('--save', type=str, default='mlm_bert.pt',
    #                     help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    parser.add_argument('--dataset', type=str, default='WikiText2',
                        help='dataset used for MLM task')
    # parser.add_argument('--parallel', type=str, default='None',
    #                     help='Use DataParallel to train model')
    parser.add_argument('--world_size', type=int, default=8,
                        help='the world size to initiate DPP')
    parser.add_argument('--rank', type=int, default=None,
                        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of GPUs per worker node to use')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of GPUs per worker node to use')
    parser.add_argument('--rpc', type=str, default='cpu',
                        help='pipeline mode, `cpu` for CPU RPC, `cuda` for CUDA RPC')
    parser.add_argument('--num_batch', type=int, default=0)
    args = parser.parse_args()

    assert args.world_size % args.num_workers == 0

    #run_worker(args.rank, args.world_size, args)
    mp.spawn(run_worker, args=(args,), nprocs=args.world_size // args.num_workers)
