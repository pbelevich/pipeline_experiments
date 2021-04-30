import argparse
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
from cuda_rpc import DistributedCUDARPCSequential, WorkerModule, layer_on_device, pipeline_on_devices


def creator(a, b, c):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(a, b),
        nn.ReLU(),
        nn.Linear(b, c),
        nn.ReLU(),
        nn.Linear(c, 10),
    )


class LocalPipelineImpl(nn.Module):
    def __init__(self, seq, devices=None):
        super().__init__()
        self.seq = seq
        self.devices = devices

    def forward(self, x):
        for gpu, module in enumerate(self.seq):
            device = self.devices[gpu] if self.devices is not None else next(module.parameters()).device
            x = x.to(device)
            x = module(x)
        return x

    def parameter_rrefs(self):
        return [RRef(p) for p in self.seq.parameters()]


def Sharder(devices, config, a, b, c):
    gpus = len(devices)
    layers = []
    if gpus == 6:
        layers.append(nn.Flatten().to(devices[0]))
        layers.append(nn.Linear(a, b).to(devices[1]))
        layers.append(nn.ReLU().to(devices[2]))
        layers.append(nn.Linear(b, c).to(devices[3]))
        layers.append(nn.ReLU().to(devices[4]))
        layers.append(nn.Linear(c, 10).to(devices[5]))
    return LocalPipelineImpl(nn.Sequential(*layers), devices=devices)

def run_main(world_size):
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    valid_data = datasets.MNIST('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    # model = DistributedCUDARPCSequential(
    #     WorkerModule("worker1", layer_on_device("cuda:0"), creator, 28 * 28, 128, 64)
    # )

    # model = DistributedCUDARPCSequential(
    #     WorkerModule("worker1", layer_on_device("cuda:0"), nn.Flatten),
    #     WorkerModule("worker2", layer_on_device("cuda:1"), nn.Linear, 28 * 28, 128),
    #     WorkerModule("worker3", layer_on_device("cuda:2"), nn.ReLU),
    #     WorkerModule("worker4", layer_on_device("cuda:3"), nn.Linear, 128, 64),
    #     WorkerModule("worker5", layer_on_device("cuda:4"), nn.ReLU),
    #     WorkerModule("worker6", layer_on_device("cuda:5"), nn.Linear, 64, 10),
    # )

    model = DistributedCUDARPCSequential(
        WorkerModule("worker1", pipeline_on_devices(0, 1, 2, 3, 4, 5), Sharder, 28 * 28, 128, 64),
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
                y_batch = y_batch.to(5)
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
                                  
            print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")
            accuracy[k].append(epoch_correct/epoch_all)


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        # options.set_device_map("worker1", {0:0})
        # options.set_device_map("worker2", {1:1})
        # options.set_device_map("worker3", {2:2})
        # options.set_device_map("worker4", {3:3})
        # options.set_device_map("worker5", {4:4})
        # options.set_device_map("worker6", {5:5})

        # for i in range(1, world_size):
        #     gpu = i - 1
        #     options.set_device_map(f"worker{i}", {gpu:gpu})
        #     # print(f'options.set_device_map("worker{i}", {gpu}:{gpu})')

        options.set_device_map("worker1", {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_main(world_size)
    else:
        # if rank == 1:
        #     options.set_device_map("worker2", {0:1})
        # elif rank == 2:
        #     options.set_device_map("worker3", {1:2})
        # elif rank == 3:
        #     options.set_device_map("worker4", {2:3})
        # elif rank == 4:
        #     options.set_device_map("worker5", {3:4})
        # elif rank == 5:
        #     options.set_device_map("worker6", {4:5})
        
        # if rank != world_size - 1:
        #     src_gpu = rank - 1
        #     dest_worker = f"worker{rank+1}"
        #     dest_gpu = rank
        #     options.set_device_map(dest_worker, {src_gpu:dest_gpu})
        #     # print(f'options.set_device_map("{dest_worker}", {src_gpu}:{dest_gpu})')

        options.set_device_map("master", {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

    rpc.shutdown()


if __name__=="__main__":
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
