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

    model = MyRPCPipeline(
        MyRPCPipelineWrapper(nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ), "worker1:0", rpc_mode='cpu'),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = DistributedOptimizer(
        optim.Adam,
        model.parameter_rrefs(),
    )

    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    max_epochs = 10
    accuracy = {"train": [], "valid": []}
    for epoch in range(max_epochs):
        print(f"Epoch: {epoch+1}")
        epoch_correct = 0
        epoch_all = 0
        for k, dataloader in loaders.items():
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader)):
                x_batch = x_batch.to(0)
                y_batch = y_batch
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
                    # with torch.no_grad():
                    outp = model(x_batch)
                    preds = outp.argmax(-1)
                    correct = (preds == y_batch).sum()
                    all = len(y_batch)
                    epoch_correct += correct.item()
                    epoch_all += all
                                  
            print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")
            accuracy[k].append(epoch_correct/epoch_all)


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
