"""Lightning datamodule for substation chips (S2 10-band, optionally + S1 VV/VH).

Single modality (`modalities=[S2L2A]`) returns a plain image tensor — byte-identical to
earthpv's proven path. Dual (`[S2L2A, S1RTC]`) returns a modality dict
`{"image": {"S2L2A": t, "S1RTC": t}, "mask": m}`, which TerraTorch's segmentation task,
`pixel_wise_model` and `TerraMindViT.forward` accept directly (verified against the
installed terratorch: one pretrained patch-embed per modality, tokens merged by mean).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from subdetect.config import S1_MEAN, S1_OFFSET_DB, S1_SCALE, S1_STD

# S2 chips store reflectance DN (x10000); the backbone is fine-tuned on reflectance in
# [0, ~1.5] (no internal standardization). S1 is standardized to the pretrain dB stats.
DN_SCALE = 10000.0
_S1_MEAN = np.array(S1_MEAN, dtype="float32")[:, None, None]
_S1_STD = np.array(S1_STD, dtype="float32")[:, None, None]


def _load_s1(path: str) -> np.ndarray:
    """Read the S1 uint16 DN chip -> standardized dB, nodata (DN==0) -> mean (x=0)."""
    with rasterio.open(path) as src:
        dn = src.read().astype("float32")  # (2, H, W)
    nodata = dn <= 0
    db = dn / S1_SCALE - S1_OFFSET_DB
    x = (db - _S1_MEAN) / _S1_STD
    x[nodata] = 0.0
    return x


class SubChipDataset(Dataset):
    def __init__(self, index: pd.DataFrame, modalities: list[str], augment: bool = False,
                 upsample: int = 1):
        self.index = index.reset_index(drop=True)
        self.modalities = modalities
        self.dual = "S1RTC" in modalities
        self.augment = augment
        # upsample > 1: bicubic-upsample inputs (nearest for the mask) so a small
        # substation spans more ViT tokens. Adds no information -- it attacks the
        # token-granularity limit (one 16 px patch = 160 m at native 10 m, bigger
        # than most small substations). See the v12 config header.
        self.upsample = upsample

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        row = self.index.iloc[i]
        with rasterio.open(row["image"]) as src:
            s2 = src.read().astype("float32") / DN_SCALE  # (10, H, W)
        with rasterio.open(row["mask"]) as src:
            mask = src.read(1).astype("int64")
        s1 = _load_s1(row["s1"]) if self.dual else None

        if self.augment:
            k = np.random.randint(4)
            flip = np.random.rand() < 0.5
            s2 = np.rot90(s2, k, (1, 2)).copy()
            mask = np.rot90(mask, k, (0, 1)).copy()
            if s1 is not None:
                s1 = np.rot90(s1, k, (1, 2)).copy()
            if flip:
                s2 = s2[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
                if s1 is not None:
                    s1 = s1[:, :, ::-1].copy()

        s2_t, mask_t = torch.from_numpy(s2), torch.from_numpy(mask)
        s1_t = torch.from_numpy(s1) if s1 is not None else None
        if self.upsample > 1:
            f = float(self.upsample)
            s2_t = torch.nn.functional.interpolate(
                s2_t[None], scale_factor=f, mode="bicubic", align_corners=False)[0]
            if s1_t is not None:
                s1_t = torch.nn.functional.interpolate(
                    s1_t[None], scale_factor=f, mode="bilinear", align_corners=False)[0]
            mask_t = torch.nn.functional.interpolate(
                mask_t[None, None].float(), scale_factor=f, mode="nearest")[0, 0].long()
        if self.dual:
            img = {"S1RTC": s1_t}
            if "S2L2A" in self.modalities:
                img["S2L2A"] = s2_t
            return {"image": img, "mask": mask_t}
        return {"image": s2_t, "mask": mask_t}


class SubDataModule(LightningDataModule):
    def __init__(
        self,
        index_path: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        modalities: list[str] | None = None,
        min_val_chips: int = 8,
        upsample: int = 1,
    ):
        super().__init__()
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities or ["S2L2A"]
        self.min_val_chips = min_val_chips
        self.upsample = upsample

    def setup(self, stage: str | None = None) -> None:
        index = pd.read_parquet(self.index_path)
        train = index[index.split == "train"]
        val = index[index.split == "val"]
        if len(val) < self.min_val_chips:
            val = train.sample(frac=0.2, random_state=42)
            train = train.drop(val.index)
        self.train_ds = SubChipDataset(train, self.modalities, augment=True,
                                       upsample=self.upsample)
        self.val_ds = SubChipDataset(val, self.modalities, upsample=self.upsample)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
