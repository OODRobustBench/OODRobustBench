"""Input normalization layer."""

import torch
from torch import nn


class Normalize(nn.Module):
    """Normalize images by mean and std."""

    def __init__(self, mean, std, *args, **kwargs) -> None:
        """Initialize Normalize.

        Args:
            mean: Mean of images per-chaneel.
            std: Std of images per-channel.
        """
        _ = args, kwargs  # Unused
        super().__init__()
        if mean is None or std is None:
            self.mean, self.std = None, None
        else:
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, images):
        """Normalize images."""
        if self.mean is None:
            return images
        return (images - self.mean) / self.std
