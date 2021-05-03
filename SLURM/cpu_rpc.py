import os
import socket
import subprocess

import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.distributed.rpc import RRef


class DistributedCPURPCSequential(nn.Module):
    def __init__(self, *worker_layers):
        super().__init__()
        self.worker_layers = worker_layers

    def forward(self, x):  # x is cpu tensor on master
        x_rref = RRef(x)  # x_rref is initially on master cpu
        for worker_layer in self.worker_layers:
            x_rref = worker_layer(x_rref)  # pass to worker layer
        return x_rref.to_here()  # get x to master cpu

    def train(self, mode=True):
        for worker_layer in self.worker_layers:
            worker_layer.train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def parameter_rrefs(self):
        remote_params = []
        for worker_layer in self.worker_layers:
            remote_params.extend(worker_layer.parameter_rrefs().to_here())
        return remote_params


class WorkerModule(nn.Module):
    def __init__(self, worker, remote_class_creator, *args, **kwargs):
        super().__init__()
        self.remote_module = rpc.remote(worker, LocalWrapper, (remote_class_creator, *args), kwargs)

    def train(self, mode=True):
        self.remote_module.rpc_sync().train(mode=mode)

    def forward(self, x_rref):
        return self.remote_module.remote().forward(x_rref)

    def parameter_rrefs(self):
        return self.remote_module.remote().parameter_rrefs()


class LocalWrapper(nn.Module):
    def __init__(self, local_net_creator, *args, **kwargs):
        super().__init__()
        self.local_net = local_net_creator(*args, **kwargs)
        first_parameter = next(self.local_net.parameters(), None)
        self.first_device = local_net_creator.device

    def train(self, mode=True):
        self.local_net.train(mode=mode)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.first_device)
        return self.local_net(x).cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.local_net.parameters()]


def get_my_gpu_index():
    try:
        return int(os.getenv("CUDA_VISIBLE_DEVICES"))
    except:
        None


class _layer_on_device_helper():
    def __init__(self, device):
        self.device = device

    def __call__(self, layer_class, *args, **kwargs):
        res = layer_class(*args, **kwargs).to(self.device)
        print(f"Materializing {layer_class} on {socket.gethostname()}:{get_my_gpu_index()}")
        return res


def layer_on_device(device):
    return _layer_on_device_helper(device)
