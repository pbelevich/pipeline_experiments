import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

class DistributedCUDARPCSequential(nn.Module):
    def __init__(self, *worker_layers):
        super().__init__()
        self.worker_layers = worker_layers
        first = worker_layers[0]
        self.first_shard = rpc.remote(first.worker, LocalWrapper, (worker_layers[1:], first.remote_class_creator, *(first.args)), **(first.kwargs))

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


class LocalWrapper(nn.Module):
    def __init__(self, worker_layers, local_net_creator, *args, **kwargs):
        super().__init__()
        self.local_net = local_net_creator(*args, **kwargs)
        if worker_layers is None or len(worker_layers) == 0:
            self.next_shard = None
        else:
            first = worker_layers[0]
            self.next_shard = rpc.remote(first.worker, LocalWrapper, (worker_layers[1:], first.remote_class_creator, *(first.args)), **(first.kwargs))

    def train(self, mode=True):
        self.local_net.train(mode=mode)
        if self.next_shard is not None:
            self.next_shard.rpc_sync().train(mode=mode)

    def forward(self, x):
        x = self.local_net(x)
        if self.next_shard is not None:
            return self.next_shard.remote().forward(x)
        return x

    def parameter_rrefs(self):
        param_rrefs = [RRef(p) for p in self.local_net.parameters()]
        if self.next_shard is not None:
            param_rrefs.extend(self.next_shard.remote().parameter_rrefs().to_here())
        return param_rrefs


class _layer_on_device_helper():
    def __init__(self, device):
        self.device = device

    def __call__(self, layer_class, *args, **kwargs):
        return layer_class(*args, **kwargs).to(self.device)

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
