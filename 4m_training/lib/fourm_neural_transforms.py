"""4M-compatible AbstractTransform adapters for neural modalities."""

from __future__ import annotations

import numpy as np
import torch
from fourm.data.modality_transforms import AbstractTransform

from neural_constants import EEG_CODE_MAX, MEG_CODE_MAX
from neural_trial_transform import (
    EegTrialSampleTransform,
    MegTrialSampleTransform,
)


class MegTokTransform(AbstractTransform):
    """Pick one MEG trial per call and return a (MEG_TOKENS_PER_TRIAL,) int64 tensor."""

    def __init__(self, training: bool = True, seed: int | None = None):
        self.sampler = MegTrialSampleTransform(training=training, seed=seed)

    def load(self, path: str) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    def preprocess(self, sample: np.ndarray) -> torch.Tensor:
        tokens, _valid = self.sampler(sample)
        tokens = np.clip(tokens, 0, MEG_CODE_MAX)
        return torch.from_numpy(tokens).long()

    def image_augment(
        self,
        v,
        crop_coords,
        flip,
        orig_size,
        target_size,
        rand_aug_idx,
        resample_mode=None,
    ):
        return v

    def postprocess(self, sample):
        return sample


class EegTokTransform(AbstractTransform):
    """Pick one EEG trial per call and return a (EEG_TOKENS_PER_TRIAL,) int64 tensor."""

    def __init__(self, training: bool = True, seed: int | None = None):
        self.sampler = EegTrialSampleTransform(training=training, seed=seed)

    def load(self, path: str) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    def preprocess(self, sample: np.ndarray) -> torch.Tensor:
        tokens, _valid = self.sampler(sample)
        tokens = np.clip(tokens, 0, EEG_CODE_MAX)
        return torch.from_numpy(tokens).long()

    def image_augment(self, v, *args, **kwargs):
        return v

    def postprocess(self, sample):
        return sample


class MaskFlagTransform(AbstractTransform):
    """Pass-through for uint8 presence flags (``meg_mask`` / ``eeg_mask``)."""

    def load(self, path: str) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    def preprocess(self, sample: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(sample.astype(np.int64))

    def image_augment(self, v, *args, **kwargs):
        return v

    def postprocess(self, sample):
        return sample
