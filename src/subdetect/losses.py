"""Custom losses for TerraTorch tasks not covered by built-in strings.

Ported from `earthpv.losses`, which needed exactly one of these for the PV
coverage-fraction regression head. The substation fraction target has the same
zero-inflation problem (a chip is overwhelmingly background), so the same fix
applies -- plus an optional recall-first asymmetry that the PV task did not need.
"""

from __future__ import annotations

import torch
from torch import nn


class TargetWeightedMSE(nn.Module):
    """MSE weighted by (1 + k*target), to counter zero-inflation in fraction regression.

    Most pixels have zero substation coverage; without weighting the loss is dominated by
    getting zeros exactly right and the model can collapse to predict-zero. Weighting by
    the target itself (not a fixed class balance) keeps the gradient scaled to how much
    signal a pixel actually carries, without needing a separate positive/negative split.

    `under_weight > 1` additionally scales the error on *under*-predicted pixels
    (output < target), which is this project's recall-first stance carried over from the
    Tversky beta>alpha asymmetry of the segmentation head: a missed substation costs more
    than a false positive a validator can reject in seconds. At the default 1.0 the loss
    is symmetric and byte-identical to earthpv's, so the first fraction run stays a
    faithful port with only the target representation changed.

    PixelwiseRegressionTask uses an nn.Module loss unwrapped (no ignore_index wrapper is
    applied, unlike its built-in string losses), so this module must mask ignore_index
    itself.
    """

    def __init__(self, k: float = 10.0, ignore_index: float = -1.0,
                 under_weight: float = 1.0):
        super().__init__()
        self.k = k
        self.ignore_index = ignore_index
        self.under_weight = under_weight

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if output.ndim == target.ndim + 1:
            output = output.squeeze(1)
        valid = (target != self.ignore_index).float()
        weight = (1.0 + self.k * target.clamp(min=0)) * valid
        if self.under_weight != 1.0:
            weight = weight * torch.where(
                output < target,
                torch.as_tensor(self.under_weight, dtype=weight.dtype, device=weight.device),
                torch.ones((), dtype=weight.dtype, device=weight.device),
            )
        se = (output - target) ** 2 * weight
        return se.sum() / weight.sum().clamp(min=1.0)
