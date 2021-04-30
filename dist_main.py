import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer

import torchvision
import torchvision.transforms as transforms

from resnet import ResNet50Base, ResNet50OneGPU, ResNet50TwoGPUs, ResNet50SixGPUs

# from utils import progress_bar
import copy

from dist import DistModule

def run_master(split_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    loss_fn = nn.CrossEntropyLoss()

    model = ResNet50Base()
    parts = [
        model._part1(),
        model._part2(),
        model._part3(),
        model._part4(),
        model._part5(),
        model._part6(),
    ]

    local1_devices = [f"cuda:{i}" for i in range(6)]
    local1_model = ResNet50SixGPUs(copy.deepcopy(parts), local1_devices)
    local1_opt = optim.SGD(local1_model.parameters(), lr=0.05)

    dist1_devices = [f"cuda:{i}" for i in range(6)]
    dist1_workers = [f"worker{i+1}" for i in range(6)]
    dist1_model = DistModule(copy.deepcopy(parts), dist1_devices, dist1_workers, split_size=100)
    dist1_opt = DistributedOptimizer(
        optim.SGD,
        dist1_model.parameter_rrefs(),
        lr=0.05,
    )

    def train(epoch):
        total = 0

        local1_model.train()
        local1_train_loss = 0
        local1_correct = 0
        local1_start = 0
        local1_finish = 0

        dist1_model.train()
        dist1_train_loss = 0
        dist1_correct = 0
        dist1_start = 0
        dist1_finish = 0

        pbar = tqdm(trainloader)

        for batch_idx, (inputs, labels) in enumerate(pbar):
            total += labels.size(0)

            local1_start = time.time()
            local1_labels = labels.to(local1_devices[-1])
            local1_opt.zero_grad()
            local1_outputs = local1_model(inputs.to(local1_devices[0]))
            local1_loss = loss_fn(local1_outputs, local1_labels)
            local1_loss.backward()
            local1_opt.step()
            local1_train_loss += local1_loss.item()
            _, local1_predicted = local1_outputs.max(1)
            local1_correct += local1_predicted.eq(local1_labels).sum().item()
            local1_finish = time.time()

            dist1_start = time.time()
            # The distributed autograd context is the dedicated scope for the
            # distributed backward pass to store gradients, which can later be
            # retrieved using the context_id by the distributed optimizer.
            with dist_autograd.context() as context_id:
                dist1_outputs = dist1_model(inputs)
                dist1_loss = loss_fn(dist1_outputs, labels)
                dist_autograd.backward(context_id, [dist1_loss])
                dist1_opt.step(context_id)

            dist1_train_loss += dist1_loss.item()
            _, dist1_predicted = dist1_outputs.max(1)
            dist1_correct += dist1_predicted.eq(labels).sum().item()
            dist1_finish = time.time()

            pbar.set_postfix({'local': (local1_finish - local1_start), 'dist': (dist1_finish - dist1_start)})

            # progress_bar(batch_idx, len(trainloader), '| %.3f | %.3f%% (%d/%d) | %.3f | %.3f%% (%d/%d)'
            #             % (local1_train_loss/(batch_idx+1), 100.*local1_correct/total, local1_correct, total,
            #                dist1_train_loss/(batch_idx+1), 100.*dist1_correct/total, dist1_correct, total))

            # assert local1_train_loss == dist1_train_loss
            # assert local1_correct == dist1_correct

    def test(epoch):
        total = 0

        local1_model.eval()
        local1_test_loss = 0
        local1_correct = 0
        local1_start = 0
        local1_finish = 0

        dist1_model.eval()
        dist1_test_loss = 0
        dist1_correct = 0
        dist1_start = 0
        dist1_finish = 0

        pbar = tqdm(testloader)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                total += labels.size(0)
                
                local1_start = time.time()
                local1_labels = labels.to(local1_devices[-1])
                local1_outputs = local1_model(inputs.to(local1_devices[0]))
                local1_loss = loss_fn(local1_outputs, local1_labels)
                local1_test_loss += local1_loss.item()
                _, local1_predicted = local1_outputs.max(1)
                local1_correct += local1_predicted.eq(local1_labels).sum().item()
                local1_finish = time.time()

                dist1_start = time.time()
                dist1_outputs = dist1_model(inputs)
                dist1_loss = loss_fn(dist1_outputs, labels)
                dist1_test_loss += dist1_loss.item()
                _, dist1_predicted = dist1_outputs.max(1)
                dist1_correct += dist1_predicted.eq(labels).sum().item()
                dist1_finish = time.time()

                pbar.set_postfix({'local': (local1_finish - local1_start), 'dist': (dist1_finish - dist1_start)})

                # progress_bar(batch_idx, len(testloader), '| %.3f | %.3f%% (%d/%d) | %.3f | %.3f%% (%d/%d)'
                #         % (local1_test_loss/(batch_idx+1), 100.*local1_correct/total, local1_correct, total,
                #            dist1_test_loss/(batch_idx+1), 100.*dist1_correct/total, dist1_correct, total))

                # assert local1_test_loss == dist1_test_loss
                # assert local1_correct == dist1_correct

    for epoch in range(10):
        print('\nEpoch: %d' % epoch)
        train(epoch)
        test(epoch)

def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 7
    num_split = 1
    mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
