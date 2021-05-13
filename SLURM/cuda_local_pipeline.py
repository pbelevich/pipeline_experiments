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
