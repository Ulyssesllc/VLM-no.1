"""Simplified InfoGAN generator module.

Provides a lightweight noise->image generator for integration into the
classification training pipeline (synthetic augmentation). All training
loops and discriminator code removed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _norm(x: torch.Tensor) -> torch.Tensor:
    if x.device != IMAGENET_MEAN.device:
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    return (x - mean) / std


class InfoGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_size: int,
        out_channels: int = 3,
        code_dim: int = 2,
        n_classes: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.n_classes = n_classes
        self.init_size = img_size // 4
        input_dim = latent_dim + code_dim + n_classes
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, 128 * self.init_size * self.init_size)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        with torch.no_grad():
            labels = F.one_hot(
                torch.randint(0, self.n_classes, (b,), device=z.device), self.n_classes
            ).float()
            code = torch.empty(b, self.code_dim, device=z.device).uniform_(-1, 1)
        gen_input = torch.cat((z, labels, code), dim=1)
        out = self.l1(gen_input)
        out = out.view(b, 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = (img + 1) * 0.5  # [0,1]
        return _norm(img)


__all__ = ["InfoGANGenerator"]
