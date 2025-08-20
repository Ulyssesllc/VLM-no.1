"""Simplified ClusterGAN generator (encoder/discriminator removed)."""

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


class ClusterGANGenerator(nn.Module):
    def __init__(
        self, latent_dim: int, img_size: int, out_channels: int = 3, n_classes: int = 10
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.base = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2, True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.resize = nn.Upsample(
            size=(img_size, img_size), mode="bilinear", align_corners=False
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        with torch.no_grad():
            idx = torch.randint(0, self.n_classes, (b,), device=z.device)
            one_hot = F.one_hot(idx, self.n_classes).float()
        h = self.base(torch.cat([z, one_hot], 1))
        h = h.view(b, 128, 7, 7)
        img = self.deconv(h)
        img = self.resize(img)
        return _norm(img)


__all__ = ["ClusterGANGenerator"]
