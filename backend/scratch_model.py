"""Compact first-party video model for laptop-scale CPU experiments.

This model is randomly initialised. It does not download pretrained weights or
call any hosted AI service. Expect very limited results until substantial,
well-aligned replacement data has been collected and training is evaluated.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.5


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.5


class ReferenceEncoder(nn.Module):
    """Learn appearance features from the supplied reference photo."""
    def __init__(self, base: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base, 3, padding=1), ResBlock2D(base))
        self.level1 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), ResBlock2D(base * 2))
        self.level2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), ResBlock2D(base * 4))
        self.level3 = nn.Sequential(nn.Conv2d(base * 4, base * 6, 3, stride=2, padding=1), ResBlock2D(base * 6))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        r0 = self.stem(image)
        r1 = self.level1(r0)
        r2 = self.level2(r1)
        r3 = self.level3(r2)
        return r0, r1, r2, r3


class ReferenceFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )

    def forward(self, video: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        reference = F.interpolate(reference, size=video.shape[-2:], mode="bilinear", align_corners=False)
        reference = reference.unsqueeze(2).expand(-1, -1, video.shape[2], -1, -1)
        return video + self.mix(torch.cat((video, reference), dim=1)) * 0.5


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, 3, padding=1),
            ResBlock3D(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(x.shape[2], skip.shape[3], skip.shape[4]),
            mode="trilinear", align_corners=False,
        )
        return self.mix(torch.cat((x, skip), dim=1))


class EffectLabScratchVideoAI(nn.Module):
    """Small reference-conditioned spatiotemporal network trained from zero.

    The defaults intentionally reduce memory use for CPU experiments. It is
    a research prototype, not a pretrained or production-quality swap model.
    """
    def __init__(self, base_channels: int = 8):
        super().__init__()
        if base_channels < 4 or base_channels > 32:
            raise ValueError("base_channels must be between 4 and 32.")
        c = base_channels
        self.base_channels = c
        self.reference_encoder = ReferenceEncoder(c)
        self.video_stem = nn.Conv3d(3, c, 3, padding=1)
        self.fuse0 = ReferenceFusion(c)
        self.enc0 = ResBlock3D(c)
        self.down1 = nn.Conv3d(c, c * 2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fuse1 = ReferenceFusion(c * 2)
        self.enc1 = ResBlock3D(c * 2)
        self.down2 = nn.Conv3d(c * 2, c * 4, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fuse2 = ReferenceFusion(c * 4)
        self.enc2 = ResBlock3D(c * 4)
        self.down3 = nn.Conv3d(c * 4, c * 6, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fuse3 = ReferenceFusion(c * 6)
        self.mid = ResBlock3D(c * 6)
        self.up2 = UpBlock3D(c * 6, c * 4, c * 4)
        self.up1 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up0 = UpBlock3D(c * 2, c, c)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(c), c), nn.SiLU(),
            nn.Conv3d(c, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, reference: torch.Tensor, driving: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 4 or driving.ndim != 5:
            raise ValueError("Expected reference [B,3,H,W] and driving [B,3,T,H,W].")
        if reference.shape[0] != driving.shape[0] or reference.shape[1] != 3 or driving.shape[1] != 3:
            raise ValueError("Reference and driving must share a batch size and use RGB channels.")
        if reference.shape[-2:] != driving.shape[-2:]:
            raise ValueError("Reference and driving frames must have the same dimensions.")
        if reference.shape[-2] % 8 or reference.shape[-1] % 8:
            raise ValueError("Image height and width must be divisible by 8.")
        r0, r1, r2, r3 = self.reference_encoder(reference)
        x0 = self.enc0(self.fuse0(self.video_stem(driving), r0))
        x1 = self.enc1(self.fuse1(self.down1(x0), r1))
        x2 = self.enc2(self.fuse2(self.down2(x1), r2))
        x3 = self.mid(self.fuse3(self.down3(x2), r3))
        y = self.up2(x3, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        return self.output(y).clamp(0, 1)
