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
from subdetect.tower_prior import DN_SCALE as _TOWER_DN_SCALE

# S2 chips store reflectance DN (x10000); the backbone is fine-tuned on reflectance in
# [0, ~1.5] (no internal standardization). S1 is standardized to the pretrain dB stats.
DN_SCALE = 10000.0
_S1_MEAN = np.array(S1_MEAN, dtype="float32")[:, None, None]
_S1_STD = np.array(S1_STD, dtype="float32")[:, None, None]


def _drop_from_page_cache(path: str) -> None:
    """Tell the kernel this chip's pages need not stay cached (earthpv's guard). A corpus larger
    than a unit's MemoryMax otherwise fills the page cache charged to the unit, and the
    in-cgroup reclaim that follows is the pressure systemd-oomd kills on (subdetect-v17-s2b,
    2026-11-03: 18.6 GB, mostly chip page cache). Reads are unchanged."""
    import os

    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def _load_s1(path: str) -> np.ndarray:
    """Read the S1 uint16 DN chip -> standardized dB, nodata (DN==0) -> mean (x=0)."""
    with rasterio.open(path) as src:
        dn = src.read().astype("float32")  # (2, H, W)
    nodata = dn <= 0
    db = dn / S1_SCALE - S1_OFFSET_DB
    x = (db - _S1_MEAN) / _S1_STD
    x[nodata] = 0.0
    return x


def _load_tower(path: str) -> np.ndarray:
    """Read the tower-prior uint16 DN chip -> [0, 1] float32, nodata (DN==0) -> 0.0.

    Already bounded in [0, 1] (see tower_prior.tower_prior_dn) -- no standardization,
    same convention as reflectance-scale S2 fed to a backbone with no internal norm."""
    with rasterio.open(path) as src:
        dn = src.read().astype("float32")  # (1, H, W)
    nodata = dn <= 0
    x = np.clip((dn - 1.0) / _TOWER_DN_SCALE, 0.0, 1.0)
    x[nodata] = 0.0
    return x


class SubChipDataset(Dataset):
    def __init__(self, index: pd.DataFrame, modalities: list[str], augment: bool = False,
                 upsample: int = 1, n_revisits: int = 1, target: str = "class",
                 jitter: dict | None = None, drop_page_cache: bool = False):
        self.index = index.reset_index(drop=True)
        # Radiometric jitter (training only, earthpv's v22 recipe): with probability `p`, each
        # S2 band x a gain from 1 +/- `gain`, plus one offset from +/- `offset` (reflectance),
        # the same draw for every revisit of a multi-date stack. S1 is left alone.
        self.jitter = jitter
        self.drop_page_cache = drop_page_cache
        self.modalities = modalities
        # target="fraction": masks are float32 sub-pixel coverage in [0, 1] (-1 = ignore)
        # for PixelwiseRegressionTask, instead of int64 class ids for the segmentation
        # task. See chips.py::_burn_fraction and configs/terramind_sub_v15_fraction.yaml.
        self.target = target
        self.has_s1 = "S1RTC" in modalities
        self.has_tower = "power_tower_prior" in modalities
        self.dual = self.has_s1 or self.has_tower
        self.augment = augment
        # upsample > 1: bicubic-upsample inputs (nearest for the mask) so a small
        # substation spans more ViT tokens. Adds no information -- it attacks the
        # token-granularity limit (one 16 px patch = 160 m at native 10 m, bigger
        # than most small substations). See the v12 config header.
        self.upsample = upsample
        # n_revisits > 1: "image" points at an (n_revisits*10, H, W) stack of
        # individual per-date S2 composites (band-order [date0's 10 bands, date1's
        # 10 bands, ...], see scripts/build_multidate_chips.py) instead of one
        # median-collapsed image -- fed to a TemporalWrapper-wrapped backbone
        # (see train.py) as a (10, T, H, W) tensor per sample, batched to the
        # (B, C, T, H, W) shape terratorch's TemporalWrapper expects. S1/tower are
        # NOT multi-date (kept as single median composites) -- this is an S2-only
        # ablation, see configs/terramind_sub_v14_multitemporal*.yaml.
        self.n_revisits = n_revisits

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        row = self.index.iloc[i]
        with rasterio.open(row["image"]) as src:
            s2 = src.read().astype("float32") / DN_SCALE  # (n_revisits*10, H, W) or (10, H, W)
        with rasterio.open(row["mask"]) as src:
            mask = src.read(1).astype("float32" if self.target == "fraction" else "int64")
        s1 = _load_s1(row["s1"]) if self.has_s1 else None
        tower = _load_tower(row["tower"]) if self.has_tower else None
        if self.drop_page_cache:
            for col in ("image", "mask", *(["s1"] if self.has_s1 else [])):
                _drop_from_page_cache(row[col])

        if self.augment:
            k = np.random.randint(4)
            flip = np.random.rand() < 0.5
            s2 = np.rot90(s2, k, (1, 2)).copy()
            mask = np.rot90(mask, k, (0, 1)).copy()
            if s1 is not None:
                s1 = np.rot90(s1, k, (1, 2)).copy()
            if tower is not None:
                tower = np.rot90(tower, k, (1, 2)).copy()
            if flip:
                s2 = s2[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
                if s1 is not None:
                    s1 = s1[:, :, ::-1].copy()
                if tower is not None:
                    tower = tower[:, :, ::-1].copy()
            j = self.jitter
            if j and np.random.rand() < j.get("p", 0.5):
                g, o = float(j.get("gain", 0.0)), float(j.get("offset", 0.0))
                gain = np.random.uniform(1 - g, 1 + g, (10, 1, 1)).astype("float32")
                gain = np.tile(gain, (s2.shape[0] // 10, 1, 1))  # same draw for each revisit
                s2 = (s2 * gain + np.float32(np.random.uniform(-o, o))).astype("float32")

        s2_t, mask_t = torch.from_numpy(s2), torch.from_numpy(mask)
        s1_t = torch.from_numpy(s1) if s1 is not None else None
        tower_t = torch.from_numpy(tower) if tower is not None else None
        if self.n_revisits > 1:
            # rot90/flip above ran on the full (n_revisits*10, H, W) channel axis --
            # spatially identical across revisits, so a single augmentation draw
            # applies consistently to all of them for free. Reshape only now.
            t, h, w = self.n_revisits, s2_t.shape[-2], s2_t.shape[-1]
            s2_t = s2_t.view(t, 10, h, w).permute(1, 0, 2, 3).contiguous()  # (10, T, H, W)
        if self.upsample > 1 and self.n_revisits > 1:
            raise NotImplementedError("upsample + n_revisits not combined in this ablation")
        if self.upsample > 1:
            f = float(self.upsample)
            s2_t = torch.nn.functional.interpolate(
                s2_t[None], scale_factor=f, mode="bicubic", align_corners=False)[0]
            if s1_t is not None:
                s1_t = torch.nn.functional.interpolate(
                    s1_t[None], scale_factor=f, mode="bilinear", align_corners=False)[0]
            if tower_t is not None:
                tower_t = torch.nn.functional.interpolate(
                    tower_t[None], scale_factor=f, mode="bilinear", align_corners=False)[0]
            # Nearest for both targets: it preserves the -1 ignore sentinel exactly (any
            # smoothing would smear it into valid pixels), and coverage fraction is an
            # intensive quantity, so replicating a 10 m pixel's fraction across the four
            # 5 m pixels it becomes is the correct nearest-neighbour answer.
            mask_t = torch.nn.functional.interpolate(
                mask_t[None, None].float(), scale_factor=f, mode="nearest")[0, 0]
            if self.target != "fraction":
                mask_t = mask_t.long()
        if self.dual:
            img = {}
            if "S2L2A" in self.modalities:
                img["S2L2A"] = s2_t
            if self.has_s1:
                img["S1RTC"] = s1_t
            if self.has_tower:
                img["power_tower_prior"] = tower_t
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
        n_revisits: int = 1,
        target: str = "class",
        jitter: dict | None = None,
        drop_page_cache: bool = False,
    ):
        super().__init__()
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities or ["S2L2A"]
        self.min_val_chips = min_val_chips
        self.upsample = upsample
        self.n_revisits = n_revisits
        self.target = target
        self.jitter = jitter
        self.drop_page_cache = drop_page_cache

    def setup(self, stage: str | None = None) -> None:
        index = pd.read_parquet(self.index_path)
        train = index[index.split == "train"]
        val = index[index.split == "val"]
        if len(val) < self.min_val_chips:
            val = train.sample(frac=0.2, random_state=42)
            train = train.drop(val.index)
        self.train_ds = SubChipDataset(train, self.modalities, augment=True,
                                       upsample=self.upsample, n_revisits=self.n_revisits,
                                       target=self.target, jitter=self.jitter,
                                       drop_page_cache=self.drop_page_cache)
        self.val_ds = SubChipDataset(val, self.modalities, upsample=self.upsample,
                                     n_revisits=self.n_revisits, target=self.target,
                                     drop_page_cache=self.drop_page_cache)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
