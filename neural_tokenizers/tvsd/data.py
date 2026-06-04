"""TVSD data loader.

Loads THINGS_normMUA.mat files from TVSDAssembly and returns (B, C, T) tensors
aligned with ALLMAT image indices for label resolution.

Data shapes after loading:
  train_MUA  : (22248, 1024) float64 per timebin per monkey
  signal out : (22248, C, T) float32  C = region channels, T = timebins

Region channel slices (monkeyF / monkeyN):
  V1   0:512   (512 ch both monkeys)
  IT   512:832 / 768:1024  (320 / 256 ch)
  V4   832:1024 / 512:768  (192 / 256 ch)
  Full 0:1024
"""

from __future__ import annotations

import os
from typing import Literal

import h5py
import numpy as np
import torch


ASSEMBLY_ROOT = "/scratch/users/liubr/cs375/TVSDAssembly"

_REGION_SLICES: dict[str, dict[str, tuple[int, int]]] = {
    "monkeyF": {"V1": (0, 512), "IT": (512, 832), "V4": (832, 1024), "Full": (0, 1024)},
    "monkeyN": {"V1": (0, 512), "V4": (512, 768), "IT": (768, 1024), "Full": (0, 1024)},
}

# 10ms timebins available: 25 ms to 195 ms post-stimulus onset (step 10 ms)
_TIMEBINS_10MS = list(range(25, 205, 10))  # [25, 35, ..., 195] = 18 bins


def _mat_path(root: str, timebin: int | str, monkey: str) -> str:
    if timebin == "default":
        return os.path.join(root, "defaultBins", monkey, "THINGS_normMUA.mat")
    return os.path.join(root, "10msBins", str(timebin), monkey, "THINGS_normMUA.mat")


def _load_allmat_image_ids(root: str, monkey: str) -> np.ndarray:
    """Return (22248,) int64 array of 1-indexed image IDs for training trials.

    Loaded from ALLMAT row 1 in THINGS_normMUA_all.mat. Values are:
      0       → catch trial (no THINGS image shown)
      1–22248 → 1-indexed into the stimulus set train_imgs ordering
    """
    all_mat_path = os.path.join(root, "10msBins", str(_TIMEBINS_10MS[0]), monkey,
                                "THINGS_normMUA_all.mat")
    with h5py.File(all_mat_path, "r") as f:
        # ALLMAT shape in h5py: (6, 25248) — rows=metadata fields, cols=trials
        # First 22248 columns are training trials
        image_ids = f["ALLMAT"][1, :22248]
    return image_ids.astype(np.int64)


def load_tvsd(
    root: str = ASSEMBLY_ROOT,
    monkey: str = "monkeyF",
    region: str = "IT",
    timebin_ms: int | Literal["default"] = 10,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load TVSD training MUA as a (B, C, T) tensor.

    Args:
        root:        TVSDAssembly root directory.
        monkey:      'monkeyF' or 'monkeyN'.
        region:      'IT', 'V1', 'V4', or 'Full'.
        timebin_ms:  10 → stack 18 bins (25–195 ms) → T=18;
                     'default' → single default-bin epoch → T=1.

    Returns:
        signal:      (B, C, T) float32 tensor, B=22248 training trials.
        image_ids:   (B,) int64 numpy array; 1-indexed THINGS image index,
                     0 = catch trial (no valid image → label will be -1).
    """
    if monkey not in _REGION_SLICES:
        raise ValueError(f"monkey must be 'monkeyF' or 'monkeyN', got {monkey!r}")
    if region not in _REGION_SLICES[monkey]:
        raise ValueError(f"region must be one of {list(_REGION_SLICES[monkey])}, got {region!r}")

    ch_start, ch_end = _REGION_SLICES[monkey][region]

    if timebin_ms == "default":
        timebins = ["default"]
    elif timebin_ms == 10:
        timebins = _TIMEBINS_10MS
    else:
        raise ValueError(f"timebin_ms must be 10 or 'default', got {timebin_ms!r}")

    slices: list[np.ndarray] = []
    for tb in timebins:
        path = _mat_path(root, tb, monkey)
        with h5py.File(path, "r") as f:
            # h5py loads MATLAB arrays transposed: (cols, rows) → (22248, 1024)
            data = f["train_MUA"][:, ch_start:ch_end]  # (22248, C)
        slices.append(data[:, :, np.newaxis])  # (22248, C, 1)

    signal_np = np.concatenate(slices, axis=2).astype(np.float32)  # (22248, C, T)
    signal = torch.from_numpy(signal_np)

    image_ids = _load_allmat_image_ids(root, monkey)
    return signal, image_ids
