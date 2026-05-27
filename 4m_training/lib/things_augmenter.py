"""Pretokenized THINGS vision adapters.

THINGS RGB/depth shards lack the two things stock 4M's pretokenized path
assumes: a ``crop_settings/`` tar and an on-disk *augmentation axis*. Both
shims live here so the "THINGS has no baked-in augmentations" assumption is
expressed in one place:

  * ``ThingsImageAugmenter`` — supplies a fixed center crop when ``crop_settings``
    is absent.
  * ``ThingsTokTransform`` — keeps the full ``(196,)`` token grid instead of
    indexing an augmentation axis that THINGS shards do not have.
"""

from __future__ import annotations

import numpy as np
import torch
from fourm.data.image_augmenter import PreTokenizedImageAugmenter
from fourm.data.modality_transforms import TokTransform

from neural_constants import THINGS_CENTER_CROP, THINGS_IMAGE_SIZE

DEFAULT_CROP_SETTINGS = np.array([list(THINGS_CENTER_CROP)], dtype=np.float32)


class ThingsTokTransform(TokTransform):
    """``TokTransform`` for shards stored without an augmentation axis.

    Stock ``image_augment`` does ``v[rand_aug_idx]``, which assumes the on-disk
    array is ``(n_augmentations, n_tokens)`` and selects one crop. THINGS RGB /
    depth are flat ``(n_tokens,)`` (one crop, no augmentations — verified on the
    Modal ``project`` volume), so the stock index would silently collapse the
    grid to a single scalar token.

    A 1-D array is returned whole; a 2-D ``(n_augs, n_tokens)`` array falls back
    to stock augmentation selection, so CC12M-style shards keep working.
    """

    def image_augment(
        self, v, crop_coords, flip, orig_size, target_size, rand_aug_idx, resample_mode=None
    ):
        if np.ndim(v) == 1:
            return torch.as_tensor(np.asarray(v))
        return super().image_augment(
            v, crop_coords, flip, orig_size, target_size, rand_aug_idx, resample_mode
        )


class ThingsImageAugmenter(PreTokenizedImageAugmenter):
    """``PreTokenizedImageAugmenter`` that falls back to center crop when needed.

    Stock 4M always indexes ``crop_settings[0]`` even with ``no_aug=True``.
    THINGS pretokenized shards may omit ``crop_settings/``; this subclass
    supplies a fixed THINGS_IMAGE_SIZE center crop in that case.
    """

    def __call__(self, mod_dict, crop_settings):
        if crop_settings is None:
            crop_settings = DEFAULT_CROP_SETTINGS

        if self.main_domain in mod_dict and "tok" not in self.main_domain:
            image = (
                mod_dict[self.main_domain]
                if self.main_domain is not None
                else mod_dict[list(mod_dict.keys())[0]]
            )
            orig_width, orig_height = image.size
            orig_size = (orig_height, orig_width)
        else:
            orig_size = None

        rand_aug_idx = 0 if self.no_aug else np.random.randint(len(crop_settings))
        top, left, h, w, flip = crop_settings[rand_aug_idx]
        crop_coords = (top, left, h, w)
        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx
