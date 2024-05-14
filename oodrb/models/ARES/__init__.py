import os
import argparse
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import transforms
import gdown

# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy

from .utils import NormalizeByChannelMeanStd, distributed_init
from .resnet import resnet50, wide_resnet50_2
from .resnet_denoise import resnet152_fd
from .model_zoo import model_zoo

def gelu(inplace=None):
    return nn.GELU()

def get_model(model_name):
    backbone=model_zoo[model_name]['model']
    url = model_zoo[model_name]['url']

    src_path='./models/imagenet/Linf'
    ckpt_name=f'{model_name}_checkpoint.pth'
    ckpt_dir=os.path.join(src_path, ckpt_name)
    ckpt_list=os.listdir(src_path)
    if ckpt_name not in ckpt_list:
        gdown.download(url, ckpt_dir, quiet=False)
    
    mean=model_zoo[model_name]['mean']
    std=model_zoo[model_name]['std']
    pretrained=model_zoo[model_name]['pretrained']
    act_gelu=model_zoo[model_name]['act_gelu']
    
    if backbone=='resnet50_rl':
        model=resnet50()
    elif backbone=='wide_resnet50_2_rl':
        model=wide_resnet50_2()
    elif backbone=='resnet152_fd':
        model = resnet152_fd()
    elif backbone=='vit_base_patch16' or backbone=='vit_large_patch16':
        model=vit_mae.__dict__[backbone](num_classes=1000, global_pool='')
    else:
        model_kwargs=dict({'num_classes': 1000})
        if act_gelu:
            model_kwargs['act_layer']= gelu
        model = create_model(backbone, pretrained=pretrained, **model_kwargs)
    
    if not pretrained:
        ckpt=torch.load(ckpt_dir, map_location='cpu')
        model.load_state_dict(ckpt)

    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = torch.nn.Sequential(normalize, model)
    return model
