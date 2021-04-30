import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import concurrent.futures
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef


class RemoteBaseCPURPC(nn.Module):
    def __init__(self, underlying, device):
        super().__init__()
        self.underlying = underlying.to(device)
        self.device = device

    def forward(self, x_rref):
        return self.underlying(x_rref.to_here().to(self.device)).cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


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
    def __init__(self, underlying, remote_device, rpc_mode):
        super().__init__()
        self.underlying = underlying
        self.worker, self.device = remote_device.split(":")
        self.device = int(self.device)
        self.rpc_mode = rpc_mode
        self.shard = None

    def move_underlying_to_device(self):
        if self.rpc_mode == 'cpu':
            remote_base_class = RemoteBaseCPURPC
        elif self.rpc_mode == 'cuda':
            remote_base_class = RemoteBaseCUDARPC
        else:
            raise 
        self.shard = rpc.remote(self.worker, remote_base_class, args=(self.underlying, self.device))

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