"""Load Lindsay-Lab/substation-seg's pretrained multi-temporal SWIN checkpoint
(neurograce/SubstationDataset on HuggingFace) as a standalone inference module.

CAVEAT (documented, not silently assumed): their dataloader.py contains two mutually
inconsistent hard-coded 9-band index lists for "swin only takes 9 channels" and no
published exact training invocation for the released checkpoint, so the precise
band selection/order/normalization used to produce this specific .pth is not
fully recoverable from their repo. This module uses Satlas's own officially
documented band convention (github.com/allenai/satlas/blob/main/Normalization.md:
TCI R,G,B /255 + B05,B06,B07,B08,B11,B12 /8160 clipped to [0,1]) since the backbone
was pretrained by Satlas on exactly that convention and Lindsay-Lab fine-tuned from
it -- the best-justified choice given genuine ambiguity, NOT a verified match to
their exact preprocessing. We also don't have an actual Sentinel-2 TCI (rendered,
gamma-stretched 8-bit preview) asset, only raw reflectance DN, so the TCI channels
are approximated via a simple linear stretch. Treat any results from this module as
a directional/exploratory signal, not a clean apples-to-apples ablation.

Checkpoint state-dict keys confirmed by direct inspection (torch.load):
  fpn_model.backbone.backbone.backbone.features.0.0.weight -> shape (128, 9, 4, 4)
    (9 input channels, torchvision swin_v2_b, confirms Satlas Sentinel2_SwinB_MI_MS)
  fpn_model.backbone.backbone.* -> AggregationBackbone wrapping SwinBackbone
    (satlaspretrain_models.models.backbones.AggregationBackbone: runs the shared
    single-image backbone independently per revisit, then torch.amax over time
    per FPN depth level -- this IS the "latent-space max-fusion" mechanism)
  fpn_model.fpn.* -> satlaspretrain_models FPN
  upsampler.{0,2,3,5,6,8}.{weight,bias} -> custom 4-block ConvTranspose upsampler,
    final conv out_channels=2 (softmax 2-class, index 1 = substation prob)
  no "head.*" keys -> fpn=True, head=None when constructing (Lindsay's SwinWithUpSample
    supplies its own upsampler instead of satlaspretrain's SimpleHead)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from satlaspretrain_models.model import Model as SatlasModel
from satlaspretrain_models.utils import Backbone

MID_CHANNELS = 128


class SwinWithUpSample(nn.Module):
    """Reimplementation of Lindsay-Lab's models.py class (not imported -- their repo
    has extra deps we don't need since we're not training). num_outputs=2 confirmed
    by the checkpoint's upsampler.8 weight shape (2, 32, 3, 3)."""

    def __init__(self, fpn_model: nn.Module, num_outputs: int = 2):
        super().__init__()
        self.fpn_model = fpn_model
        self.upsampler = nn.Sequential(
            nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(MID_CHANNELS, MID_CHANNELS // 2, kernel_size=2, stride=2),
            nn.Conv2d(MID_CHANNELS // 2, MID_CHANNELS // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(MID_CHANNELS // 2, MID_CHANNELS // 4, kernel_size=2, stride=2),
            nn.Conv2d(MID_CHANNELS // 4, MID_CHANNELS // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(MID_CHANNELS // 4, num_outputs, 3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # NOT self.fpn_model(image): current satlaspretrain_models unconditionally
        # applies its own Upsample module after the FPN when fpn=True, but the
        # checkpoint has no fpn_model.upsample.* weights -- Lindsay-Lab's version
        # predates that addition, so feature[0] below is the raw FPN's highest-res
        # (1/4 input resolution, 128ch) output; Lindsay's own upsampler (2x
        # ConvTranspose stride=2 stages) takes it the rest of the way to full res.
        x = self.fpn_model.backbone(image)
        feature = self.fpn_model.fpn(x)
        raw = self.upsampler(feature[0])
        return torch.softmax(raw, dim=1)


def load_swin_lindsay(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    fpn_model = SatlasModel(num_channels=9, multi_image=True, backbone=Backbone.SWINB,
                            fpn=True, head=None, weights=None)
    del fpn_model.upsample  # see forward()'s comment
    model = SwinWithUpSample(fpn_model, num_outputs=2)
    sd = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)


# Our 10-band composite order (config.LOCAL_BANDS): B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
#                                                     0   1   2   3   4   5   6   7   8   9
# Satlas 9-band MS order: TCI-R(B04) TCI-G(B03) TCI-B(B02) B05 B06 B07 B08 B11 B12
_SATLAS_BAND_IDX = [2, 1, 0, 3, 4, 5, 6, 8, 9]  # indices into our 10-band array


def to_satlas_input(s2_dn: np.ndarray) -> np.ndarray:
    """(10, H, W) raw uint16-ish reflectance DN -> (9, H, W) float32 Satlas-convention input.

    TCI channels approximated from raw reflectance (we have no rendered TCI asset):
    a simple linear stretch (DN/10000 reflectance * 3.5, clipped to [0,1], * 255) that
    puts typical bright land around the upper-middle of the 0-255 TCI range, then divided
    by 255 per Satlas's own doc -- equivalent to reflectance * 3.5 clipped to [0,1] directly.
    Non-TCI bands: raw DN / 8160 clipped to [0,1] per Satlas's documented convention.
    """
    sel = s2_dn[_SATLAS_BAND_IDX].astype("float32")
    out = np.empty_like(sel)
    out[:3] = np.clip(sel[:3] / 10000.0 * 3.5, 0.0, 1.0)  # approximate TCI
    out[3:] = np.clip(sel[3:] / 8160.0, 0.0, 1.0)
    return out
