"""ConvMixer for CIFAR-10/100 datasets.

Code adapted from https://github.com/locuslab/convmixer-cifar10/blob/main/train.py.
"""

from __future__ import annotations

from torch import nn

# Original train transform
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
#     transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
#     transforms.ToTensor(),
#     transforms.Normalize(cifar10_mean, cifar10_std),
#     transforms.RandomErasing(p=args.reprob)
# ])

# Get ~92.5%
# python train.py --lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.005 --scale=1.0 --jitter=0 --reprob=0
# Get 94%
# python train.py --lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.005 --scale=1.0 --jitter=0.2 --reprob=0.2 --epochs=100

# parser.add_argument('--scale', default=0.75, type=float)
# parser.add_argument('--reprob', default=0.25, type=float)
# parser.add_argument('--ra-m', default=8, type=int)
# parser.add_argument('--ra-n', default=1, type=int)
# parser.add_argument('--jitter', default=0.1, type=float)

# parser.add_argument('--hdim', default=256, type=int)
# parser.add_argument('--depth', default=8, type=int)
# parser.add_argument('--psize', default=2, type=int)
# parser.add_argument('--conv-ks', default=5, type=int)


class Residual(nn.Module):
    """ConvMixer Residual."""

    def __init__(self, fn):
        """Initialize Residual."""
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """Residual forward pass."""
        return self.fn(x) + x


def build_conv_mixer(
    dim=256, depth=8, kernel_size=5, patch_size=2, num_classes=10, **kwargs
) -> nn.Module:
    """Create ConvMixer model."""
    _ = kwargs  # Unused
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            dim, dim, kernel_size, groups=dim, padding="same"
                        ),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                ),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes),
    )
