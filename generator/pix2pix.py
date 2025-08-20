"""Minimal Pix2Pix-style UNet generator (no discriminator/training loop)."""

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


class _Block(nn.Module):
    def __init__(self, in_c, out_c, down=True, act=True):
        super().__init__()
        if down:
            self.op = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, True),
            )
        else:
            self.op = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            )
        if not act:
            # remove activation
            self.op = self.op[:-1]

    def forward(self, x):
        return self.op(x)


class Pix2PixGenerator(nn.Module):
    def __init__(
        self, latent_dim: int, img_size: int, out_channels: int = 3, base: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feat = nn.Sequential(
            nn.Linear(latent_dim, base * 8 * 4 * 4),
            nn.BatchNorm1d(base * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.up = nn.Sequential(
            _Block(base * 8, base * 4, down=False),
            _Block(base * 4, base * 2, down=False),
            _Block(base * 2, base, down=False),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self.resize = nn.Upsample(
            size=(img_size, img_size), mode="bilinear", align_corners=False
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.feat(z)
        x = x.view(z.size(0), -1, 4, 4)
        x = self.up(x)
        x = self.final(x)
        x = (x + 1) * 0.5
        x = self.resize(x)
        return _norm(x)


__all__ = ["Pix2PixGenerator"]
