"""EffectLab Scratch: an original, uninitialised reference-conditioned video network.

No pretrained checkpoints, model hubs, remote APIs, or imported AI model definitions.
The network has to be trained on aligned person-replacement examples before inference.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.5


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.5


class ReferenceEncoder(nn.Module):
    """Learn appearance features directly from the supplied full-body photo."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), ResBlock2D(32))
        self.level1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), ResBlock2D(64)
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), ResBlock2D(128)
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=2, padding=1), ResBlock2D(192)
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f0 = self.stem(image)
        f1 = self.level1(f0)
        f2 = self.level2(f1)
        f3 = self.level3(f2)
        return f0, f1, f2, f3


class ReferenceFusion(nn.Module):
    """Inject learned identity/wardrobe features into every driving-video frame."""
    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            ResBlock3D(channels),
        )
        self.affine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels * 2, 1),
        )

    def forward(self, video: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _, _ = video.shape
        # Reference appearance map is broadcast in time, not pasted over the video.
        appearance = reference.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        scale, shift = self.affine(reference).chunk(2, dim=1)
        modulated = video * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
        return self.mix(torch.cat((modulated, appearance), dim=1))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, 3, padding=1),
            ResBlock3D(out_channels),
            ResBlock3D(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(skip.shape[2], skip.shape[3], skip.shape[4]),
            mode="trilinear", align_corners=False,
        )
        return self.mix(torch.cat((x, skip), dim=1))


class EffectLabScratchVideoAI(nn.Module):
    """Train-from-zero full-frame person replacement network.

    Inputs:
      reference: (B,3,H,W), full-body photo of the identity to render
      driving:   (B,3,T,H,W), source video whose movement/camera to follow
    Output:
      generated: (B,3,T,H,W), RGB video in [0,1]

    This architecture starts from random weights. It is not useful for real
    replacement until trained on aligned reference/driving/target-video triples.
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()
        if base_channels != 32:
            raise ValueError("This checkpoint format currently expects base_channels=32.")
        self.reference_encoder = ReferenceEncoder()
        self.video_stem = nn.Conv3d(3, 32, 3, padding=1)
        self.fuse0 = ReferenceFusion(32)
        self.enc0 = ResBlock3D(32)
        self.down1 = nn.Conv3d(32, 64, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fuse1 = ReferenceFusion(64)
        self.enc1 = ResBlock3D(64)
        self.down2 = nn.Conv3d(64, 128, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fuse2 = ReferenceFusion(128)
        self.enc2 = ResBlock3D(128)
        self.down3 = nn.Conv3d(128, 192, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fuse3 = ReferenceFusion(192)
        self.mid = nn.Sequential(ResBlock3D(192), ResBlock3D(192))
        self.up2 = UpBlock3D(192, 128, 128)
        self.up1 = UpBlock3D(128, 64, 64)
        self.up0 = UpBlock3D(64, 32, 32)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv3d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, reference: torch.Tensor, driving: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 4 or driving.ndim != 5:
            raise ValueError("Expected reference [B,3,H,W] and driving [B,3,T,H,W].")
        if reference.shape[0] != driving.shape[0]:
            raise ValueError("Reference and driving batch sizes do not match.")
        if reference.shape[1] != 3 or driving.shape[1] != 3:
            raise ValueError("Inputs must be RGB tensors.")
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
