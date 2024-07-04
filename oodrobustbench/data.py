import os
from addict import Dict

from torch.utils.data import DataLoader

from .datasets import cifar, imagenet

NATURAL_SHIFTS = Dict()
NATURAL_SHIFTS.cifar10 = list(cifar.DATASETS.keys())
NATURAL_SHIFTS.imagenet = imagenet.ImageNet.VARIANTS

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def load_natural_shift_data(root, dataset, shift, n_examples, transform):
    if dataset == 'cifar10':
        dataset = cifar.DATASETS[shift](root, transform=transform)
    elif dataset == 'imagenet':
        dataset = imagenet.ImageNet(root, shift, transform=transform)

    loader = DataLoader(dataset, batch_size=n_examples, shuffle=False, num_workers=4)
    
    data = next(iter(loader))
    return data[:2]
        

    
    
    
