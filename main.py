import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from resnet import ResNet50Base, ResNet50OneGPU, ResNet50TwoGPUs, ResNet50SixGPUs

# from utils import progress_bar
import copy

if __name__=="__main__":
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

    local1_devices = ["cuda:0"]
    local1_model = ResNet50OneGPU(copy.deepcopy(parts), local1_devices)
    local1_opt = optim.SGD(local1_model.parameters(), lr=0.05)

    local2_devices = ["cuda:1", "cuda:2"]
    local2_model = ResNet50TwoGPUs(copy.deepcopy(parts), local2_devices)
    local2_opt = optim.SGD(local2_model.parameters(), lr=0.05)

    local3_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]
    local3_model = ResNet50SixGPUs(copy.deepcopy(parts), local3_devices)
    local3_opt = optim.SGD(local3_model.parameters(), lr=0.05)

    def train(epoch):
        total = 0

        local1_model.train()
        local1_train_loss = 0
        local1_correct = 0
        local1_start = 0
        local1_finish = 0

        local2_model.train()
        local2_train_loss = 0
        local2_correct = 0
        local2_start = 0
        local2_finish = 0

        local3_model.train()
        local3_train_loss = 0
        local3_correct = 0
        local3_start = 0
        local3_finish = 0

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

            local2_start = time.time()
            local2_labels = labels.to(local2_devices[-1])
            local2_opt.zero_grad()
            local2_outputs = local2_model(inputs.to(local2_devices[0]))
            local2_loss = loss_fn(local2_outputs, local2_labels)
            local2_loss.backward()
            local2_opt.step()
            local2_train_loss += local2_loss.item()
            _, local2_predicted = local2_outputs.max(1)
            local2_correct += local2_predicted.eq(local2_labels).sum().item()
            local2_finish = time.time()

            local3_start = time.time()
            local3_labels = labels.to(local3_devices[-1])
            local3_opt.zero_grad()
            local3_outputs = local3_model(inputs.to(local3_devices[0]))
            local3_loss = loss_fn(local3_outputs, local3_labels)
            local3_loss.backward()
            local3_opt.step()
            local3_train_loss += local3_loss.item()
            _, local3_predicted = local3_outputs.max(1)
            local3_correct += local3_predicted.eq(local3_labels).sum().item()
            local3_finish = time.time()

            pbar.set_postfix({'l1': (local1_finish - local1_start), 'l2': (local2_finish - local2_start), 'l3': (local3_finish - local3_start)})

            # progress_bar(batch_idx, len(trainloader), '| %.3f | %.3f%% (%d/%d) | %.3f | %.3f%% (%d/%d) | %.3f | %.3f%% (%d/%d)'
            #             % (local1_train_loss/(batch_idx+1), 100.*local1_correct/total, local1_correct, total,
            #                local2_train_loss/(batch_idx+1), 100.*local2_correct/total, local2_correct, total,
            #                local3_train_loss/(batch_idx+1), 100.*local3_correct/total, local3_correct, total))

            assert local1_train_loss == local2_train_loss == local3_train_loss
            assert local1_correct == local2_correct == local3_correct

    def test(epoch):
        total = 0

        local1_model.eval()
        local1_test_loss = 0
        local1_correct = 0
        local1_start = 0
        local1_finish = 0

        local2_model.eval()
        local2_test_loss = 0
        local2_correct = 0
        local2_start = 0
        local2_finish = 0

        local3_model.eval()
        local3_test_loss = 0
        local3_correct = 0
        local3_start = 0
        local3_finish = 0

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

                local2_start = time.time()
                local2_labels = labels.to(local2_devices[-1])
                local2_outputs = local2_model(inputs.to(local2_devices[0]))
                local2_loss = loss_fn(local2_outputs, local2_labels)
                local2_test_loss += local2_loss.item()
                _, local2_predicted = local2_outputs.max(1)
                local2_correct += local2_predicted.eq(local2_labels).sum().item()
                local2_finish = time.time()

                local3_start = time.time()
                local3_labels = labels.to(local3_devices[-1])
                local3_outputs = local3_model(inputs.to(local3_devices[0]))
                local3_loss = loss_fn(local3_outputs, local3_labels)
                local3_test_loss += local3_loss.item()
                _, local3_predicted = local3_outputs.max(1)
                local3_correct += local3_predicted.eq(local3_labels).sum().item()
                local3_finish = time.time()

                pbar.set_postfix({'l1': (local1_finish - local1_start), 'l2': (local2_finish - local2_start), 'l3': (local3_finish - local3_start)})

                # progress_bar(batch_idx, len(testloader), '| %.3f | %.3f%% (%d/%d) | %.3f | %.3f%% (%d/%d) | %.3f | %.3f%% (%d/%d)'
                #         % (local1_test_loss/(batch_idx+1), 100.*local1_correct/total, local1_correct, total,
                #            local2_test_loss/(batch_idx+1), 100.*local2_correct/total, local2_correct, total,
                #            local3_test_loss/(batch_idx+1), 100.*local3_correct/total, local3_correct, total))

                assert local1_test_loss == local2_test_loss == local3_test_loss
                assert local1_correct == local2_correct == local3_correct

    for epoch in range(10):
        print('\nEpoch: %d' % epoch)
        train(epoch)
        test(epoch)
