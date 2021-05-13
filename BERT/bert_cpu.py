import argparse
import math
import sys
import time
import os
import socket
import statistics

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd

from model import MLMTask, MLMTask2, MLMTaskEmbedding, MLMTaskEncoder, MLMTaskHead
from utils import run_demo, run_ddp, wrap_up
from sharder import MLMTaskSharder
from cpu_rpc import DistributedCPURPCSequential, WorkerModule, layer_on_device


IS_SLURM = os.getenv('SLURM_LOCALID')
USE_TQDM = os.getenv('USE_TQDM', True if not IS_SLURM else False)


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

    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, lm_mask, targets


def process_raw_data(raw_data, args):
    _num = raw_data.size(0) // (args.batch_size * args.bptt)
    raw_data = raw_data[:(_num * args.batch_size * args.bptt)]
    return raw_data


def train(model, vocab, train_loss_log, train_data,
          optimizer, criterion, ntokens, epoch, args):
    model.train()
    total_loss = 0.
    start_time = time.time()
    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    train_loss_log.append(0.0)
    dataloader = DataLoader(train_data, batch_size=args.batch_size * args.bptt,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id))

    forward_elapsed = []
    backward_elapsed = []
    for batch, (data, lm_mask, targets) in enumerate(dataloader):
        with dist_autograd.context() as context_id:
            data = data.transpose(0, 1)
            forward_start_time = time.time()
            output = model(data)
            output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            loss = criterion(output.view(-1, ntokens), targets)
            forward_elapsed.append((time.time() - forward_start_time) * 1000)
            backward_start_time = time.time()
            dist_autograd.backward(context_id, [loss])
            backward_elapsed.append((time.time() - backward_start_time) * 1000)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step(context_id)
            total_loss += loss.item()

        if batch % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            train_loss_log[-1] = cur_loss
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | forward {:5.2f}±{:5.2f}({:5.2f},min={:5.2f},max={:5.2f}) | backward {:5.2f}±{:5.2f}({:5.2f},min={:5.2f},max={:5.2f}) | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch,
                                                        len(train_data) // (args.bptt * args.batch_size),
                                                        args.lr,
                                                        elapsed * 1000 / args.log_interval,
                                                        statistics.mean(forward_elapsed),
                                                        statistics.stdev(forward_elapsed) if len(forward_elapsed) > 1 else 0.0,
                                                        forward_elapsed[-1],
                                                        min(forward_elapsed),
                                                        max(forward_elapsed),
                                                        statistics.mean(backward_elapsed),
                                                        statistics.stdev(backward_elapsed) if len(backward_elapsed) > 1 else 0.0,
                                                        backward_elapsed[-1],
                                                        min(backward_elapsed),
                                                        max(backward_elapsed),
                                                        cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def half1(ntokens, args):
    return nn.Sequential(
        MLMTaskEmbedding(ntokens, args.emsize),
        MLMTaskEncoder(args.emsize, args.nhead, args.nhid, args.nlayers // 2, args.dropout),
    )
            


def half2(ntokens, args):
    return nn.Sequential(
        MLMTaskEncoder(args.emsize, args.nhead, args.nhid, args.nlayers // 2, args.dropout),
        MLMTaskHead(ntokens, args.emsize),
    )


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

    nworkers = args.world_size - 1
    
    if nworkers == 1:
        model = DistributedCPURPCSequential(
            WorkerModule("worker1", layer_on_device("cuda"), MLMTask, ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout),
        )
    elif nworkers == 2:
        assert(args.nlayers % 2 == 0)
        model = DistributedCPURPCSequential(
            WorkerModule("worker1", layer_on_device("cuda"), half1, ntokens, args),
            WorkerModule("worker2", layer_on_device("cuda"), half2, ntokens, args),
        )
    else:
        assert(args.nlayers % (nworkers - 2) == 0)
        model = DistributedCPURPCSequential(
            WorkerModule("worker1", layer_on_device("cuda"), MLMTaskEmbedding, ntokens, args.emsize),
            *(WorkerModule(f"worker{i}", layer_on_device("cuda"), MLMTaskEncoder, args.emsize, args.nhead, args.nhid, args.nlayers // (nworkers - 2), args.dropout) for i in range(2, args.world_size - 1)),
            WorkerModule(f"worker{args.world_size-1}", layer_on_device("cuda"), MLMTaskHead, ntokens, args.emsize),
        )

    params = sum([torch.prod(torch.tensor(p.rpc_sync().size())) for p in model.parameter_rrefs()])
    print(f'Total parameters = {params.item() // 10**6}M')

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


def run_worker(rank, world_size, args):
    print(f"rank = {rank} host/pid = {socket.gethostname()}/{os.getpid()}")
    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=10800)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_main(args)
    else:
        if not IS_SLURM:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank - 1)
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
    
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
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    parser.add_argument('--dataset', type=str, default='WikiText2',
                        help='dataset used for MLM task')
    parser.add_argument('--world_size', type=int, default=7,
                        help='the world size to initiate DPP')
    parser.add_argument('--rank', type=int, default=None,
                        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help="""Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port.""")
    parser.add_argument('--master_port', type=str, default='29500',
                        help="""Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    if args.rank is None:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
