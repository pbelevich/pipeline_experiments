import argparse
import os

import torch
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms
from tqdm import tqdm

from cpu_rpc import DistributedCPURPCSequential, WorkerModule, layer_on_device


IS_SLURM = os.getenv('SLURM_LOCALID')
USE_TQDM = os.getenv('USE_TQDM', True if not IS_SLURM else False)


def run_main():
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    valid_data = datasets.MNIST('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    model = DistributedCPURPCSequential(
        WorkerModule("worker1", layer_on_device("cuda"), nn.Flatten),
        WorkerModule("worker2", layer_on_device("cuda"), nn.Linear, 28 * 28, 128),
        WorkerModule("worker3", layer_on_device("cuda"), nn.ReLU),
        WorkerModule("worker4", layer_on_device("cuda"), nn.Linear, 128, 64),
        WorkerModule("worker5", layer_on_device("cuda"), nn.ReLU),
        WorkerModule("worker6", layer_on_device("cuda"), nn.Linear, 64, 10),
        microbatch_size=batch_size // 4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = DistributedOptimizer(
        optim.Adam,
        model.parameter_rrefs(),
    )

    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    max_epochs = 10
    for epoch in range(max_epochs):
        print(f"Epoch: {epoch + 1}")
        epoch_correct = 0
        epoch_all = 0
        for k, dataloader in loaders.items():
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader) if USE_TQDM else dataloader):
                if k == "train":
                    model.train()
                    with dist_autograd.context() as context_id:
                        outp = model(x_batch)
                        preds = outp.argmax(-1)
                        correct = (preds == y_batch).sum()
                        all = len(y_batch)
                        epoch_correct += correct.item()
                        epoch_all += all
                        loss = criterion(outp, y_batch)
                        dist_autograd.backward(context_id, [loss])
                        optimizer.step(context_id)
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                        preds = outp.argmax(-1)
                        correct = (preds == y_batch).sum()
                        all = len(y_batch)
                        epoch_correct += correct.item()
                        epoch_all += all

            acc = epoch_correct / epoch_all if epoch_all != 0 else -1
            print(f"Loader: {k}. Accuracy: {acc}")


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_main()
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

    parser.add_argument('--world_size', type=int, default=7,
                        help='the world size')
    parser.add_argument('--rank', type=int, default=None,
                        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help="""Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port.""")
    parser.add_argument('--master_port', type=str, default='29500',
                        help="""Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    if args.rank is None:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    else:
        run_worker(args.rank, args.world_size, args)
