"""Simplified BicycleGAN generator (encoder & discriminators removed).

Generates images from latent noise only (condition path removed) for use as
auxiliary synthetic data in training pipeline.
"""

from __future__ import annotations
import torch
import torch.nn as nn

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


class BicycleGANGenerator(nn.Module):
    def __init__(
        self, latent_dim: int, img_size: int, out_channels: int = 3, base: int = 64
    ):
        super().__init__()
        # Use a small MLP + reshape + UNet-like up path
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base * 8 * 4 * 4),
            nn.BatchNorm1d(base * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self.resize = nn.Upsample(
            size=(img_size, img_size), mode="bilinear", align_corners=False
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), -1, 4, 4)
        x = self.up(x)
        x = self.final(x)
        x = (x + 1) * 0.5
        x = self.resize(x)
        return _norm(x)


__all__ = ["BicycleGANGenerator"]
