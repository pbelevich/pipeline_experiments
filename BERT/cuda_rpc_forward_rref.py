import os
import socket
import subprocess

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.distributed.rpc import RRef


class DistributedCUDARPCSequential(nn.Module):
    def __init__(self, *worker_layers):
        super().__init__()
        self.worker_layers = worker_layers

    def forward(self, x):  # x is cpu tensor on master
        x_rref = RRef(x, devices=[0])  # x_rref is initially on master cpu
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
        x = x_rref.to_here()
        return self.local_net(x)

    def parameter_rrefs(self):
        return [RRef(p, devices=[0]) for p in self.local_net.parameters()]


def get_my_gpu_index():
    try:
        return int(os.getenv("CUDA_VISIBLE_DEVICES"))
    except:
        try:
            p1 = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["grep", str(os.getpid())], stdin=p1.stdout, stdout=subprocess.PIPE)
            p3 = subprocess.Popen(["awk", "{print $2}"], stdin=p2.stdout, stdout=subprocess.PIPE)
            return int(p3.communicate()[0])
        except:
            return None


def count_model_param(nn_model):
    model_parameters = filter(lambda p: p.requires_grad, nn_model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params.item()


class _layer_on_device_helper():
    def __init__(self, device):
        self.device = device

    def __call__(self, layer_class, *args, **kwargs):
        res = layer_class(*args, **kwargs).to(self.device)
        print(f"Materializing {layer_class} with {count_model_param(res) // 1e6}M params on {socket.gethostname()}:{get_my_gpu_index()}")
        return res


def layer_on_device(device):
    return _layer_on_device_helper(device)


class _pipeline_on_devices_helper():
    def __init__(self, *devices, **config):
        self.devices = devices
        self.device = devices[0]
        self.config = config

    def __call__(self, pipeline_class, *args, **kwargs):
        return pipeline_class(self.devices, self.config, *args, **kwargs)


def pipeline_on_devices(*devices, **kwargs):
    return _pipeline_on_devices_helper(*devices, **kwargs)
