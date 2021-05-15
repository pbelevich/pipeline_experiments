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
    def __init__(self, *worker_layers, microbatch_size=None):
        super().__init__()
        self.worker_layers = worker_layers
        with concurrent.futures.ThreadPoolExecutor() as executor:
            concurrent.futures.wait([executor.submit(lambda wl: wl.materialize(), wl) for wl in self.worker_layers])
        self.microbatch_size = microbatch_size

    def forward(self, xs):
        if self.microbatch_size == None:
            x_rref = RRef(xs)  # x_rref is initially on master cpu
            for worker_layer in self.worker_layers:
                x_rref = worker_layer(x_rref)  # pass to worker layer
            return x_rref.to_here()  # get x to master cpu
        else:
            out_futures = []
            for x in iter(xs.split(self.microbatch_size, dim=0)):
                x_rref = RRef(x)
                for worker_layer in self.worker_layers[:-1]:
                    x_rref = worker_layer(x_rref)
                z_fut = self.worker_layers[-1].remote_module.rpc_async().forward(x_rref)
                out_futures.append(z_fut)
            return torch.cat(torch.futures.wait_all(out_futures))

    def get_fwd_compute_delay(self):
        return sum([worker_layer.get_fwd_compute_delay() for worker_layer in self.worker_layers])

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
        self.worker = worker
        self.remote_class_creator = remote_class_creator
        self.args = args
        self.kwargs = kwargs
        self.remote_module = None

    def materialize(self):
        self.remote_module = rpc.remote(self.worker, LocalWrapper, args=(self.remote_class_creator, *(self.args)), kwargs=self.kwargs)

    def train(self, mode=True):
        self.remote_module.rpc_sync().train(mode=mode)

    def forward(self, x_rref):
        return self.remote_module.remote().forward(x_rref)

    def get_fwd_compute_delay(self):
        return self.remote_module.rpc_sync().get_fwd_compute_delay()

    def parameter_rrefs(self):
        return self.remote_module.remote().parameter_rrefs()


class LocalWrapper(nn.Module):
    def __init__(self, local_net_creator, *args, **kwargs):
        super().__init__()
        self.local_net = local_net_creator(*args, **kwargs)
        first_parameter = next(self.local_net.parameters(), None)
        self.first_device = local_net_creator.device
        self._lock = threading.Lock()
        self.fwd_tik = torch.cuda.Event(enable_timing=True)
        self.fwd_tok = torch.cuda.Event(enable_timing=True)

    def train(self, mode=True):
        self.local_net.train(mode=mode)

    def forward(self, x_rref):
        x = x_rref.to_here()
        with self._lock:
            # NB: the two cuda events below records the comp
            # delay on this shard
            self.fwd_tik.record()
            res = self.local_net(x)
            self.fwd_tok.record()
        return res

    def get_fwd_compute_delay(self):
        self.fwd_tok.synchronize()
        return self.fwd_tik.elapsed_time(self.fwd_tok)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.local_net.parameters()]


def get_my_gpu_index():
    try:
        return int(os.getenv("CUDA_VISIBLE_DEVICES"))
    except:
        None


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
