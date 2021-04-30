import threading

import torch
import torch.nn as nn

import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

class RemoteBase(nn.Module):
    def __init__(self, underlying, device):
        super(RemoteBase, self).__init__()
        self.underlying = underlying.to(device)
        self.device = device
        self._lock = threading.Lock()

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.underlying(x)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

class DistModule(nn.Module):
    def __init__(self, shards, devices, workers, split_size=1, *args, **kwargs):
        super(DistModule, self).__init__()

        self.split_size = split_size

        self.shards = [rpc.remote(worker, RemoteBase, args=(shard, device)) for worker, shard, device in zip(workers, shards, devices)]

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            for shard in self.shards[:-1]:
                x_rref = shard.remote().forward(x_rref)
            z_fut = self.shards[-1].rpc_async().forward(x_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        for shard in self.shards:
            remote_params.extend(shard.remote().parameter_rrefs().to_here())
        return remote_params
