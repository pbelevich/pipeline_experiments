import argparse
import torch.multiprocessing as mp
import math
import sys
import time
import os
import psutil

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

class Timer(object):
    ALL = []
    def __init__(self, name):
            self._name = name
            self._children = {}
            self._start_time = None
            self.reset()
            Timer.ALL.append(self)

    def reset(self):
        self._elapsed = 0.0
        self._elapsed_sqr = 0.0
        self._count = 0

    def __enter__(self):
            self._start_time = time.time()
            self._count += 1
            return self

    def __exit__(self, *_):
            delta = time.time() - self._start_time
            self._elapsed += delta
            self._elapsed_sqr += delta * delta
            self._start_time = None

    @property
    def name(self):
        return self._name

    def avg(self):
        if self._count == 0: return 0
        return self._elapsed / self._count

    def std_dev(self):
        if self._count == 0: return 
        avg = self._elapsed / self._count
        return math.sqrt(self._elapsed_sqr / self._count - avg * avg)


    @classmethod
    def report(cls):
        r = {}
        s = {}
        for t in cls.ALL:
            r[t.name] = t.avg()
            s[t.name] = t.std_dev()
        print({"avg": r, "var": s})

    @classmethod
    def reset_all(cls):
        for t in cls.ALL:
            t.reset()

timer_fwd = Timer('forward')    
timer_loss = Timer('loss')    
timer_bwd = Timer('backward')    
timer_sync = Timer('sync')    
timer_step = Timer('step')    
timer_sync2 = Timer('sync2')    


def get_item(rref, is_remote):
    if is_remote:
        return rpc.rpc_sync(rref.owner(), get_item, (rref, False))
    return rref.local_value().view(-1)[0].item()

def run_batch(args, optimizer, model, loss_module, data, lm_mask, targets):
    with dist_autograd.context() as context_id:
        data = data.transpose(0, 1)
        with timer_fwd:
            output = model(data)
            output_item = get_item(output, True)
        with timer_loss:
            loss = loss_module(output, rpc.RRef(targets)).to_here()
            loss_item = loss.item()
        with timer_bwd:
            dist_autograd.backward(context_id, [loss])
            with timer_sync:
                sync_devices(args)
        with timer_step:
            optimizer.step(context_id)
            with timer_sync2:
                sync_devices(args)

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

    num_words = 0
    for batch, (data, lm_mask, targets) in enumerate(dataloader):
        num_words += targets.numel()
        try:
            loss = run_batch(args, optimizer, model, loss_module, data, lm_mask, targets)
        except:
            #print(rpc.rpc_sync("w3", torch.cuda.memory_stats, args=(3,)))
            #time.sleep(60)
            raise
        #rpc.rpc_sync(f"w3", torch.cuda.empty_cache)
        print("LOSS:", "%0.3f" % (loss,))
        total_loss += loss

        if batch % args.log_interval == (args.log_interval - 1) and batch > 0:
            Timer.report()
            Timer.reset_all()
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            train_loss_log[-1] = cur_loss
            print(' wps: {:0.2f} | {:5d}/{:5d} batches | s/batch {:5.2f} | loss {:5.2f}'
                    .format(num_words / elapsed,  batch,
                        len(train_data) // (args.bptt * args.batch_size),
                        elapsed / args.log_interval,
                        cur_loss))
            if args.num_batch > 0 and batch >= args.num_batch:
                break
            total_loss = 0
            num_words = 0
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

def sync_devices(args):
    ngpus = args.world_size // args.num_workers
    futs = []
    for i in range(args.world_size):
        futs.append(rpc.rpc_async(f"w{i}", torch.cuda.synchronize, (i%ngpus,)))
    torch.futures.wait_all(futs)

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
    def get_remote_device(i):
        return f"w{i}/cuda:{i % ngpus}"

    layers = [RemoteModule(get_remote_device(0), MLMTaskEmbedding, (ntokens, args.emsize))]
    n_encoders = args.nlayers
    skip_start_layers = int(args.ep_embedding)
    skip_end_layers = int(args.ep_head) + int(args.ep_noop)
    num_parts = min(n_encoders, args.world_size-skip_start_layers-skip_end_layers)
    for di, device in enumerate([get_remote_device(i) for i in range(skip_start_layers, num_parts+skip_start_layers)]):
        this_encoders = n_encoders // (num_parts-di)
        layers.append(RemoteModule(device, MLMTaskEncoder, (args.emsize,  args.nhead, args.nhid, this_encoders, args.dropout)))
        n_encoders -= this_encoders
    next_layer = num_parts + skip_start_layers - 1
    if args.ep_head:
        next_layer += 1
    layers.append(RemoteModule(get_remote_device(next_layer), MLMTaskHead, (ntokens, args.emsize)))
    if args.ep_noop:
        next_layer += 1
        layers.append(RemoteModule(get_remote_device(next_layer), NoOp, ()))

    org_model = nn.Sequential(*layers)

    graph = make_graph(org_model)
    #for node in graph.nodes: print(node.module.on, node.get_name())
    model = DistributedPipeline(graph, chunks=args.num_chunks if args.num_chunks else min(args.world_size, args.batch_size))

    params = sum([torch.prod(torch.tensor(p.rpc_sync().size())).item() for p in model.parameter_rrefs()])
    print(f'Total parameters = {int(params // 1e6)}M')

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
    n_gpu  = args.world_size // args.num_workers

    if rank < 1:
        psutil.Process().cpu_affinity([0,1])

    n_cpu = psutil.cpu_count()
    aff = [(rank + 1 + i) % n_cpu for i in range(n_cpu // 3)]
    if rank < 0:
        rank = 0
        is_master = True
    else:
        is_master = False

    signal.signal(signal.SIGUSR1, dumpstacks) 


    first_rank = n_gpu * int(os.environ.get('SLURM_PROCID', '0'))
    rank += first_rank


    if True:# rank==first_rank or is_master:
        print("rank:", -1 if is_master else rank, "pid", os.getpid())
    torch.cuda.set_per_process_memory_fraction(0.9, rank - first_rank)
    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", '29500')
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=3600)
    for i in range(args.world_size):
        options.set_device_map(f"w{i}", {rank - first_rank: i % (args.world_size // args.num_workers)})
    options.set_device_map("master", {rank - first_rank: 0})
    rpc.init_rpc(
        "master" if is_master else  f"w{rank}",
        rank=args.world_size if is_master else rank,
        world_size=args.world_size + 1,
        rpc_backend_options=options
    )

    if is_master:
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
    parser.add_argument('--num_chunks', type=int, default=0,
                        help='number of GPUs per worker node to use')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of GPUs per worker node to use')
    parser.add_argument('--rpc', type=str, default='cpu',
                        help='pipeline mode, `cpu` for CPU RPC, `cuda` for CUDA RPC')
    parser.add_argument('--num_batch', type=int, default=0)
    parser.add_argument('--ep_embedding', dest='ep_embedding', action='store_true')
    parser.add_argument('--ep_head', dest='ep_head', action='store_true')
    parser.add_argument('--ep_noop', dest='ep_noop', action='store_true')
    parser.set_defaults(ep_embedding=False, ep_head=False, ep_noop=False)
    args = parser.parse_args()

    assert args.world_size % args.num_workers == 0

    c=mp.spawn(run_worker, args=(args,), nprocs=args.world_size // args.num_workers, join=False)
    if int(os.environ.get('SLURM_PROCID', '0')) == 0:
        run_worker(-1, args)
    c.join()
