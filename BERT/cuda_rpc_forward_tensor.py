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

        for i in range(len(self.worker_layers) - 1):
            self.worker_layers[i].remote_module.rpc_sync().set_next_shard(self.worker_layers[i + 1].remote_module)

        self.first_shard = self.worker_layers[0].remote_module

    def forward(self, x):
        x = self.first_shard.remote().forward(x)
        for _ in self.worker_layers:
            x = x.to_here()
        return x

    def train(self, mode=True):
        self.first_shard.rpc_sync().train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def parameter_rrefs(self):
        return self.first_shard.remote().parameter_rrefs().to_here()


class WorkerModule():
    def __init__(self, worker, remote_class_creator, *args, **kwargs):
        self.worker = worker
        self.remote_class_creator = remote_class_creator
        self.args = args
        self.kwargs = kwargs
        self.remote_module = None

    def materialize(self):
        self.remote_module = rpc.remote(self.worker, LocalWrapper, args=(self.remote_class_creator, *(self.args)), kwargs=self.kwargs)


class LocalWrapper(nn.Module):
    def __init__(self, local_net_creator, *args, **kwargs):
        super().__init__()
        self.local_net = local_net_creator(*args, **kwargs)
        self.next_shard = None
        self._lock = threading.Lock()

    def set_next_shard(self, next_shard):
        self.next_shard = next_shard

    def train(self, mode=True):
        self.local_net.train(mode=mode)
        if self.next_shard is not None:
            self.next_shard.rpc_sync().train(mode=mode)

    def forward(self, x):
        with self._lock:
            x = self.local_net(x)
        if self.next_shard is not None:
            return self.next_shard.remote().forward(x)
        return x

    def parameter_rrefs(self):
        param_rrefs = [RRef(p) for p in self.local_net.parameters()]
        if self.next_shard is not None:
            param_rrefs.extend(self.next_shard.remote().parameter_rrefs().to_here())
        return param_rrefs


def get_my_gpu_index():
    try:
        return int(os.getenv("CUDA_VISIBLE_DEVICES"))
    except:
        return None


def count_model_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _layer_on_device_helper():
    def __init__(self, device):
        self.device = device

    def __call__(self, layer_class, *args, **kwargs):
        res = layer_class(*args, **kwargs).to(self.device)
        print(f"Materializing {layer_class} with {count_model_param(res) // 10**6}M params on {socket.gethostname()}:{get_my_gpu_index()}")
        return res


def layer_on_device(device):
    return _layer_on_device_helper(device)


# assuming CUDA_VISIBLE_DEVICES are configured in a way that each process only sees
# an exclusive set of device
def sync_all_device():
  for d in range(torch.cuda.device_count()):
    torch.cuda.synchronize(d)


def global_sync(world_size):
    futs = []
    for i in range(world_size):
        futs.append(rpc.rpc_async(i, sync_all_device))
    torch.futures.wait_all(futs)
