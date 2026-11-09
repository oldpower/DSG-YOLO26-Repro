"""Custom DPSE and SGCF layers for DSG-YOLO26.

This is a clean-room, architecture-level reconstruction from the public paper
README. The authors have not released layer-by-layer source or formulas, so this
file is not represented as their official implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import Conv

__all__ = ("DPSE", "SGCF", "DetailPreservingShallowEnhancement", "ScaleGatedCrossLevelFusion")


class _ResidualDepthwiseBlock(nn.Module):
    """A lightweight residual depthwise/pointwise convolution block."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = Conv(channels, channels, k=3, s=1, g=channels)
        self.pointwise = Conv(channels, channels, k=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.depthwise(x))


class DetailPreservingShallowEnhancement(nn.Module):
    """Enhance shallow P2 details with residual depthwise convolutions.

    Spatial resolution is never reduced inside DPSE. Each residual block uses a
    depthwise 3x3 convolution for local spatial detail and a 1x1 projection for
    channel mixing; the skip path retains the original shallow response.
    """

    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        if c1 <= 0 or c2 <= 0 or n <= 0:
            raise ValueError(f"DPSE expects positive c1/c2/n, got {(c1, c2, n)}")
        self.input_projection = nn.Identity() if c1 == c2 else Conv(c1, c2, k=1)
        self.blocks = nn.ModuleList(_ResidualDepthwiseBlock(c2) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return x


class ScaleGatedCrossLevelFusion(nn.Module):
    """Filter upsampled P3 semantics with channel-wise cross-level scale gates.

    Inputs are ordered as ``[upsampled_P3_semantics, shallow_P2_details]``. Each
    input is projected to ``c2`` channels. Global descriptors produce S x C logits,
    and softmax normalizes weights across S scales independently for every channel.
    The filtered semantic/detail sum is refined at the original P2 resolution.
    """

    def __init__(self, channels: Sequence[int], c2: int, reduction: int = 4):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if len(self.channels) != 2:
            raise ValueError(f"SGCF expects P3 and P2 inputs, got {len(self.channels)} inputs")
        if any(c <= 0 for c in self.channels) or c2 <= 0:
            raise ValueError(f"Channels must be positive, got inputs={self.channels}, c2={c2}")
        if reduction <= 0:
            raise ValueError(f"Reduction must be positive, got {reduction}")

        self.c2 = int(c2)
        self.num_scales = len(self.channels)
        self.projections = nn.ModuleList(
            nn.Identity() if c == c2 else Conv(c, c2, k=1) for c in self.channels
        )
        descriptor_channels = self.num_scales * self.c2
        hidden_channels = max(self.c2 // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(descriptor_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, descriptor_channels, kernel_size=1),
        )
        # Equal initial scale weights provide a stable, neutral fusion starting point.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.refine = Conv(self.c2, self.c2, k=3)

    def _validate_inputs(self, inputs: Sequence[torch.Tensor]) -> None:
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(f"SGCF expects a list/tuple of tensors, got {type(inputs).__name__}")
        if len(inputs) != self.num_scales:
            raise ValueError(f"SGCF expects {self.num_scales} inputs, got {len(inputs)}")
        reference_shape = inputs[0].shape
        if len(reference_shape) != 4:
            raise ValueError(f"SGCF inputs must be BCHW tensors, got {tuple(reference_shape)}")
        for index, (x, expected_channels) in enumerate(zip(inputs, self.channels)):
            if x.ndim != 4:
                raise ValueError(f"SGCF input {index} must be BCHW, got {tuple(x.shape)}")
            if x.shape[1] != expected_channels:
                raise ValueError(
                    f"SGCF input {index} has {x.shape[1]} channels; expected {expected_channels}"
                )
            if x.shape[0] != reference_shape[0] or x.shape[-2:] != reference_shape[-2:]:
                raise ValueError(
                    "SGCF inputs must have identical batch and spatial dimensions; "
                    f"input 0 is {tuple(reference_shape)}, input {index} is {tuple(x.shape)}"
                )

    def project(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Validate and project P3/P2 features to the common output channel count."""
        self._validate_inputs(inputs)
        return [projection(x) for projection, x in zip(self.projections, inputs)]

    def scale_weights(self, projected: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute gates with shape ``(batch, scales, channels)`` for projected inputs."""
        if len(projected) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} projected inputs, got {len(projected)}")
        descriptors = torch.cat([F.adaptive_avg_pool2d(x, output_size=1) for x in projected], dim=1)
        batch = projected[0].shape[0]
        logits = self.gate(descriptors).view(batch, self.num_scales, self.c2)
        return logits.softmax(dim=1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Project, gate, sum and refine the P3/P2 cross-level features."""
        projected = self.project(inputs)
        weights = self.scale_weights(projected).unsqueeze(-1).unsqueeze(-1)
        stacked = torch.stack(projected, dim=1)
        fused = (stacked * weights).sum(dim=1)
        return self.refine(fused)


DPSE = DetailPreservingShallowEnhancement
SGCF = ScaleGatedCrossLevelFusion
