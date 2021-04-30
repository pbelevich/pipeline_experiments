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


class RemoteBaseCUDARPC(nn.Module):
    def __init__(self, underlying, device):
        super().__init__()
        self.underlying = underlying.to(device)
        self.device = device

    def forward(self, x_rref):
        return self.underlying(x_rref.to_here())

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class MyRPCPipelineWrapper(nn.Module):
    def __init__(self, underlying, remote_device):
        super().__init__()
        self.underlying = underlying
        self.worker, self.device = remote_device.split(":")
        self.device = int(self.device)
        self.shard = None

    def move_underlying_to_device(self):
        self.shard = rpc.remote(self.worker, RemoteBaseCUDARPC, args=(self.underlying, self.device))

    def train(self, mode=True):
        self.shard.rpc_sync().train(mode)

    def eval(self):
        self.shard.rpc_sync().eval()

    def forward(self, *args, **kwargs):
        return self.shard.remote().forward(*args, **kwargs)

    def parameter_rrefs(self):
        return self.shard.remote().parameter_rrefs()


class MyRPCPipeline(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            concurrent.futures.wait([executor.submit(lambda l: l.move_underlying_to_device(), layer) for layer in self])

    def forward(self, x):
        return super().forward(RRef(x)).to_here()

    def parameter_rrefs(self):
        remote_params = []
        for layer in self:
            remote_params.extend(layer.parameter_rrefs().to_here())
        return remote_params


def run_main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    valid_data = datasets.MNIST('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=100)

    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = MyRPCPipeline(
        MyRPCPipelineWrapper(nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
        ), "worker1:0"),
        MyRPCPipelineWrapper(nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        ), "worker2:1"),
        MyRPCPipelineWrapper(nn.Sequential(
            nn.Linear(hidden_sizes[1], output_size),
        ), "worker3:2"),
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
            for i, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(0)
                y_batch = y_batch.to(2)
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
        options.set_device_map("worker2", {1:1})
        options.set_device_map("worker3", {2:2})

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
        elif rank == 2:
            options.set_device_map("worker1", {1:0})
        elif rank == 3:
            options.set_device_map("worker2", {2:1})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

    rpc.shutdown()


if __name__=="__main__":
    gpus = 3
    world_size = gpus + 1
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
