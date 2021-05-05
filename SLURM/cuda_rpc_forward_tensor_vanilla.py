import os
import socket
import subprocess
import concurrent.futures
import threading

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.distributed.rpc import RRef


class DistributedCUDARPCSequential(nn.Module):
    def __init__(self, *worker_layers):
        super().__init__()
        self.worker_layers = worker_layers

        with concurrent.futures.ThreadPoolExecutor() as executor:
            concurrent.futures.wait([executor.submit(lambda wl: wl.materialize(), wl) for wl in self.worker_layers])

    def forward(self, x):
        for layer in self.worker_layers:
            x = layer.forward(x)
        return x

    def train(self, mode=True):
        for worker_layer in self.worker_layers:
            worker_layer.train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def parameter_rrefs(self):
        remote_params = []
        for worker_layer in self.worker_layers:
            remote_params.extend(worker_layer.parameter_rrefs())
        return remote_params


class WorkerModule():
    def __init__(self, worker, remote_class_creator, *args, **kwargs):
        self.worker = worker
        self.remote_class_creator = remote_class_creator
        self.args = args
        self.kwargs = kwargs
        self.remote_module = None

    def materialize(self):
        self.remote_module = rpc.remote(self.worker, LocalWrapper, args=(self.remote_class_creator, *(self.args)), kwargs=self.kwargs)

    def train(self, mode=True):
        self.remote_module.rpc_sync().train(mode=mode)

    def parameter_rrefs(self):
        return self.remote_module.rpc_sync().parameter_rrefs()

    def forward(self, x):
        return self.remote_module.rpc_sync().forward(x)


class LocalWrapper(nn.Module):
    def __init__(self, local_net_creator, *args, **kwargs):
        super().__init__()
        self.local_net = local_net_creator(*args, **kwargs)
        self.next_shard = None
        self._lock = threading.Lock()


    def train(self, mode=True):
        self.local_net.train(mode=mode)
        if self.next_shard is not None:
            self.next_shard.rpc_sync().train(mode=mode)

    def forward(self, x):
        with self._lock:
            return self.local_net(x)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.local_net.parameters()]


def get_my_gpu_index():
    try:
        return int(os.getenv("CUDA_VISIBLE_DEVICES"))
    except:
        return None


class _layer_on_device_helper():
    def __init__(self, device):
        self.device = device

    def __call__(self, layer_class, *args, **kwargs):
        res = layer_class(*args, **kwargs).to(self.device)
        print(f"Materializing {layer_class} on {socket.gethostname()}:{get_my_gpu_index()}")
        return res


def layer_on_device(device):
    return _layer_on_device_helper(device)
