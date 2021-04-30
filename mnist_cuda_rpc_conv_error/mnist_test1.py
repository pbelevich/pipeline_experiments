import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import os
import concurrent.futures
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from tqdm import tqdm
from rpc_framework import MyRPCPipeline, MyRPCPipelineWrapper


def LayerOnDevice(device):
    return nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)


def run_main():
    rref = rpc.remote("worker1", LayerOnDevice, args=("cuda:0",))
    for _ in range(100):
        x = torch.randn(100, 1, 28, 28).to("cuda:0")
        actual = rref.remote().forward(x).to_here()
        expected = rref.rpc_sync().forward(x)
        assert((expected == actual).all())


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        options.set_device_map("worker1", {0:0})
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_main()
    else:
        if rank == 1:
            options.set_device_map("master", {0:0})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

    rpc.shutdown()


if __name__=="__main__":
    gpus = 1
    world_size = gpus + 1
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
