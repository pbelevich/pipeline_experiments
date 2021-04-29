import argparse
import os

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms
from resnet import ResNet50
from tqdm import tqdm

from cpu_rpc import DistributedCPURPCSequential, WorkerModule, layer_on_device


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

    model = DistributedCPURPCSequential(
        WorkerModule("worker1", layer_on_device("cuda"), layer0),
        WorkerModule("worker2", layer_on_device("cuda"), layer1),
        WorkerModule("worker3", layer_on_device("cuda"), layer2),
        WorkerModule("worker4", layer_on_device("cuda"), layer3),
        WorkerModule("worker5", layer_on_device("cuda"), layer4),
        WorkerModule("worker6", layer_on_device("cuda"), layer5),
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
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader)):
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
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_main()
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline experiments')

    parser.add_argument('--world_size', type=int, default=None,
                        help='the world size')
    parser.add_argument('--rank', type=int, default=None,
                        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help="""Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port.""")
    parser.add_argument('--master_port', type=str, default='29500',
                        help="""Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    run_worker(args.rank, args.world_size, args)

