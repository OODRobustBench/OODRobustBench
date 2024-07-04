import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x, activate):
        out = activate(self.bn1(x))
        out = self.conv1(out)
        out = activate(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class wide_layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_blocks, stride, activate):
        super(wide_layer, self).__init__()
        strides = [stride] + [1]*(int(num_blocks)-1)
        self.blocks = []
        self.activate = activate
        
        for i, stride in enumerate(strides):
            self.blocks.append(block(in_planes, planes, stride))
            setattr(self, 'block'+str(i), self.blocks[-1])
            in_planes = planes
    
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x, self.activate)
        return x
    
class Wide_ResNet(nn.Module):    
    def __init__(self, width, depth, out_dim, activation='relu', **kwargs):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        self.width = width
        self.depth = depth
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = width
        
        nStages = [16, 16*k, 32*k, 64*k]

        if activation == 'softplus':
            self.activate = F.softplus
        elif activation == 'silu':
            self.activate = F.silu
        else:
            self.activate = F.relu
        self.activation = activation
        
        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = wide_layer(wide_basic, nStages[0], nStages[1], n, 1, self.activate)
        self.layer2 = wide_layer(wide_basic, nStages[1], nStages[2], n, 2, self.activate)
        self.layer3 = wide_layer(wide_basic, nStages[2], nStages[3], n, 2, self.activate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(nStages[3], out_dim)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.activate(self.bn1(out))
        out = self.aap(out)
        out = out.view(out.size(0), -1)
        out = self.fc0(out)
        
        return out

    def hyperparams_log(self):
        return {'width' : self.width,
                'depth' : self.depth,
                'activation': self.activation}
