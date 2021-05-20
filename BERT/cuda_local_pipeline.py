import torch
import torch.nn as nn

class LocalSequential(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
        self.before_comm = torch.cuda.Event(enable_timing=True)
        self.after_comm = torch.cuda.Event(enable_timing=True)
        self.before_comp = torch.cuda.Event(enable_timing=True)
        self.after_comp = torch.cuda.Event(enable_timing=True)
        self.fwd_compute_delay = 0
        self.fwd_communication_delay = 0

    def forward(self, x):
        self.fwd_compute_delay = 0
        self.fwd_communication_delay = 0
        for layer in self:
            device = next(layer.parameters()).device
            sync_all_device(len(self))
            self.before_comm.record()
            x = x.to(device)
            sync_all_device(len(self))
            self.after_comm.record()
            self.before_comp.record()
            x = layer(x)
            sync_all_device(len(self))
            self.after_comp.record()

            self.after_comm.synchronize()
            self.fwd_communication_delay += self.before_comm.elapsed_time(self.after_comm)

            self.after_comp.synchronize()
            self.fwd_compute_delay += self.before_comp.elapsed_time(self.after_comp)
        return x

    def get_fwd_communication_delay(self):
        return self.fwd_communication_delay # * 1000

    def get_fwd_compute_delay(self):
        return self.fwd_compute_delay # * 1000


# assuming CUDA_VISIBLE_DEVICES are configured in a way that each process only sees
# an exclusive set of device
def sync_all_device(gpus):
  for d in range(gpus):
    torch.cuda.synchronize(d)