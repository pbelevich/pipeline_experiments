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
from resnet import ResNet50
from tqdm import tqdm

from cuda_rpc_forward_rref import DistributedCUDARPCSequential, WorkerModule, layer_on_device


IS_SLURM = os.getenv('SLURM_LOCALID')
USE_TQDM = os.getenv('USE_TQDM', True if not IS_SLURM else False)


def layer0():
    resnet50 = ResNet50()
    return nn.Sequential(
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
    )

def layer1():
    resnet50 = ResNet50()
    return resnet50.layer1

def layer2():
    resnet50 = ResNet50()
    return resnet50.layer2

def layer3():
    resnet50 = ResNet50()
    return resnet50.layer3

def layer4():
    resnet50 = ResNet50()
    return resnet50.layer4

def layer5():
    resnet50 = ResNet50()
    return nn.Sequential(
        resnet50.avgpool,
        resnet50.flatten,
        resnet50.linear
    )


def run_main():
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    model = DistributedCUDARPCSequential(
        WorkerModule("worker1", layer_on_device("cuda"), layer0),
        WorkerModule("worker2", layer_on_device("cuda"), layer1),
        WorkerModule("worker3", layer_on_device("cuda"), layer2),
        WorkerModule("worker4", layer_on_device("cuda"), layer3),
        WorkerModule("worker5", layer_on_device("cuda"), layer4),
        WorkerModule("worker6", layer_on_device("cuda"), layer5),
        # microbatch_size=batch_size // 4
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
                x_batch = x_batch.to(0)
                y_batch = y_batch.to(0)
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

            print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        options.set_device_map("worker1", {0: 0})
        options.set_device_map("worker2", {0: 0})
        options.set_device_map("worker3", {0: 0})
        options.set_device_map("worker4", {0: 0})
        options.set_device_map("worker5", {0: 0})
        options.set_device_map("worker6", {0: 0})

        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_main()
    else:
        if args.rank is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank - 1)
        if rank == 1:
            options.set_device_map("master", {0:0})
        elif rank == 2:
            options.set_device_map("worker1", {0:0})
        elif rank == 3:
            options.set_device_map("worker2", {0:0})
        elif rank == 4:
            options.set_device_map("worker3", {0:0})
        elif rank == 5:
            options.set_device_map("worker4", {0:0})
        elif rank == 6:
            options.set_device_map("worker5", {0:0})

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
