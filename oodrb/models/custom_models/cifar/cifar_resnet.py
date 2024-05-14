"""ResNet in PyTorch.

Code from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        beta_value: float = 0.0,
        decay_value: float = 1.0,
    ) -> None:
        super().__init__()

        # IAA
        if beta_value > 0:
            self.nonlin = nn.Softplus(beta=beta_value)
        else:
            self.nonlin = nn.ReLU(inplace=True)
        self.decay = decay_value

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.nonlin(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out.mul_(self.decay).add_(self.shortcut(x))
        out = self.nonlin(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        beta_value: float = 0.0,
        decay_value: float = 1.0,
    ) -> None:
        super().__init__()

        # IAA
        if beta_value > 0:
            self.nonlin = nn.Softplus(beta=beta_value)
        else:
            self.nonlin = nn.ReLU(inplace=True)
        self.decay = decay_value

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.nonlin(self.bn1(self.conv1(x)))
        out = self.nonlin(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out.mul_(self.decay).add_(self.shortcut(x))
        out = self.nonlin(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        num_blocks: int,
        num_classes: int = 10,
        decays: list[int] | None = None,
        beta_value: float = 0.0,
        **kwargs,
    ) -> None:
        _ = kwargs  # Unused
        super().__init__()
        self.in_planes = 64
        self.beta_value = beta_value

        # IAA
        if decays is None:
            decays = [1.0] * 4

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], strd=1, decay=decays[0]
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], strd=2, decay=decays[1]
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], strd=2, decay=decays[2]
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], strd=2, decay=decays[3]
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # We place this layer declaration at the end because named_modules()
        # returns the modules in the order they are declared, and NA attack
        # needs to know the order of the layers.
        if beta_value > 0:
            self.nonlin = nn.Softplus(beta=beta_value)
        else:
            self.nonlin = nn.ReLU(inplace=True)

    def _make_layer(
        self, block, planes, num_blocks, strd, decay: float | None = None
    ):
        strides = [strd] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    strd,
                    beta_value=self.beta_value,
                    decay_value=decay,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlin(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
