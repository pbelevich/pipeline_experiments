import torch
import torch.nn as nn

class LocalSequential(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)

    def forward(self, x):
        for layer in self:
            device = next(layer.parameters()).device
            x = x.to(device)
            x = layer(x)
        return x


# assuming CUDA_VISIBLE_DEVICES are configured in a way that each process only sees
# an exclusive set of device
def sync_all_device():
  for d in range(torch.cuda.device_count()):
    torch.cuda.synchronize(d)