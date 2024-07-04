import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, activation, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.activation(self.bn1(x))
        else:
            out = self.activation(self.bn1(x))
        out = self.activation(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, activation, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            activation, block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, activation, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block( activation, 
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, activation, depth=34, num_classes=10, widen_factor=10, dropRate=0.0,  **kwargs):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(activation, n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(activation, n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(activation, n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.activation = activation
        self.classifier = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, feature=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if feature:
            return out
        return self.classifier(out)

def wrn_28_10(activation, **kwargs):
    return WideResNet(activation, depth=28, widen_factor=10, **kwargs)


def wrn_28_4(activation, **kwargs):
    return WideResNet(activation, depth=28, widen_factor=4, **kwargs)


def wrn_28_5(activation, **kwargs):
    return WideResNet(activation, depth=28, widen_factor=5, **kwargs)


def wrn_28_1(activation, **kwargs):
    return WideResNet(activation, depth=28, widen_factor=1, **kwargs)


def wrn_34_10(activation, **kwargs):
    return WideResNet(activation, depth=34, widen_factor=10, **kwargs)


def wrn_40_2(activation, **kwargs):
    return WideResNet(activation, depth=40, widen_factor=2, **kwargs)


def wrn_34_10(activation, **kwargs):
    return WideResNet(activation, depth=34, widen_factor=10, **kwargs)


# ~4x slower than wrn-28-10
def wrn_34_20(activation, **kwargs):
    return WideResNet(activation, depth=34, widen_factor=20, **kwargs)


# ~6x slower than wrn-28-10
def wrn_70_16(activation, **kwargs):
    return WideResNet(activation, depth=70, widen_factor=16, **kwargs)
