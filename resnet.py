import torch.nn as nn
from torchvision.models.resnet import Bottleneck

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet50Base(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Base, self).__init__()

        self._block = Bottleneck
        self._norm_layer = nn.BatchNorm2d
        # self.inplanes = inplanes
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.num_classes = num_classes

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))
        inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _part1(self):
        return self._init_weights(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            self._norm_layer(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ))

    def _part2(self):
        return self._init_weights(self._make_layer(64, 64, 3))

    def _part3(self):
        return self._init_weights(self._make_layer(256, 128, 4, stride=2))

    def _part4(self):
        return self._init_weights(self._make_layer(512, 256, 6, stride=2))

    def _part5(self):
        return self._init_weights(self._make_layer(1024, 512, 3, stride=2))

    def _part6(self):
        return self._init_weights(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512 * self._block.expansion, self.num_classes)
        ))

    def _init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        return modules

class ResNet50OneGPU(ResNet50Base):
    def __init__(self, parts, devices, num_classes=10):
        super(ResNet50OneGPU, self).__init__(num_classes=num_classes)
        self.devices = devices
        self.seq = nn.Sequential(*parts).to(devices[0])
    
    def forward(self, x):
        return self.seq(x)

class ResNet50TwoGPUs(ResNet50Base):
    def __init__(self, parts, devices, num_classes=10):
        super(ResNet50TwoGPUs, self).__init__(num_classes=num_classes)
        self.devices = devices
        self.seq1 = nn.Sequential(*parts[:3]).to(devices[0])
        self.seq2 = nn.Sequential(*parts[3:]).to(devices[1])
    
    def forward(self, x):
        return self.seq2(self.seq1(x).to(self.devices[1]))

class ResNet50SixGPUs(ResNet50Base):
    def __init__(self, parts, devices, num_classes=10):
        super(ResNet50SixGPUs, self).__init__(num_classes=num_classes)
        self.devices = devices
        self.seq = []
        for i in range(6):
            self.seq.append(parts[i].to(devices[i]))
        self.seq = nn.Sequential(*self.seq)
    
    def forward(self, x):
        for i in range(6):
            x = self.seq[i](x)
            if i != 5:
                x = x.to(self.devices[i+1])
        return x
