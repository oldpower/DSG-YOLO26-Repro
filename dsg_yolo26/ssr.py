"""Optional training-only Scale-Sensitive Regression (SSR) weights.

The public DSG-YOLO26 abstract specifies that SSR reweights localization
supervision during training and adds no inference cost. It does not disclose a
weighting formula or integration with a particular IoU loss. This configurable
utility therefore represents an explicit engineering choice, not official code.
It is deliberately absent from the detection graph and all inference paths.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ("ScaleSensitiveRegression",)


class ScaleSensitiveRegression(nn.Module):
    """Compute normalized box-area weights for a localization loss.

    Args:
        power (float): Exponent applied to box area. Negative values emphasize
            smaller boxes, positive values emphasize larger boxes, and zero gives
            uniform weights. The default -0.5 is a reproducible design choice.
        eps (float): Numerical stabilizer for zero/small areas.
        box_format (str): Input box format, either ``xyxy`` or ``xywh``.

    Input shape is ``(..., 4)`` and output shape is ``(...,)``. Invalid boxes with
    zero area receive zero weight. Valid weights are normalized to mean one.
    """

    def __init__(self, power: float = -0.5, eps: float = 1e-6, box_format: str = "xyxy"):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if box_format not in {"xyxy", "xywh"}:
            raise ValueError(f"box_format must be 'xyxy' or 'xywh', got '{box_format}'")
        self.power = float(power)
        self.eps = float(eps)
        self.box_format = box_format

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """Return one scale-sensitive localization weight per target box."""
        if boxes.ndim < 1 or boxes.shape[-1] != 4:
            raise ValueError(f"Expected boxes with shape (..., 4), got {tuple(boxes.shape)}")
        if self.box_format == "xyxy":
            widths = boxes[..., 2] - boxes[..., 0]
            heights = boxes[..., 3] - boxes[..., 1]
        else:
            widths, heights = boxes[..., 2], boxes[..., 3]
        widths, heights = widths.clamp(min=0), heights.clamp(min=0)
        area = widths * heights
        valid = area > 0
        raw_weights = (area + self.eps).pow(self.power)
        count = valid.sum().clamp(min=1)
        mean_weight = (raw_weights * valid).sum() / count
        weights = raw_weights / mean_weight.clamp(min=self.eps)
        return torch.where(valid, weights, torch.zeros_like(weights))

    def extra_repr(self) -> str:
        return f"power={self.power}, eps={self.eps}, box_format='{self.box_format}'"
