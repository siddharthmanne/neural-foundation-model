"""Reusable train/(val)/test split policy for THINGS-MEG.

Why this module exists separately from data.py:
  Splits are a *policy* (which trials go where), data.py is a *loader* (how
  to read .fif files). The same policy applies whether you're feeding a
  μ-transform calibration step or a Cho2026 finetune loop.

Why per-image (not per-trial):
  THINGS-MEG repeats each of the ~22,449 unique images across multiple
  trials per subject. A random per-trial split puts the same image in
  train and test → the linear probe (§5.3) gets a free boost from
  memorizing image-specific noise. See meg/CLAUDE.md §9.

The critical invariant maintained here:
  **The TEST set is determined by (seed, test_frac) alone.**
  Whether you ask for val_frac=0 (Phase 1) or val_frac=0.1 (Phase 2+),
  the test images are the same. This makes the §5 leaderboard meaningful
  across phases — every tokenizer is graded on the same held-out trials.

When the team-wide things_split.json lands later, the *only* change is the
implementation of `split_by_image`. Every downstream caller still gets a
`Splits(train, val, test)` object back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meg_config import SplitDefaults


@dataclass(frozen=True)
class Splits:
    """Trial-index arrays for one subject, partitioned by image identity.

    Each array holds *trial* indices into a subject's epoch list, not image
    IDs. `val` may be a length-0 array if val_frac=0 was requested — callers
    should treat it as "no val set" rather than special-casing the field.

    Attributes:
        train, val, test: int64 arrays of trial indices (sorted ascending).
        image_ids_test:   the image IDs that ended up in test, kept around
                          for debugging / cross-phase consistency checks.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    image_ids_test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def split_by_image(
    image_ids_per_trial: np.ndarray,
    defaults: SplitDefaults,
) -> Splits:
    """Partition trial indices into train/val/test grouped by image identity.

    Args:
        image_ids_per_trial: (n_trials,) int array of THINGS image codes,
            one entry per trial in the subject's epoch list. Equivalent to
            `epochs.events[:, 2]` from MNE.
        defaults: SplitDefaults with (train_frac, val_frac, test_frac, seed).

    Returns:
        Splits with .train, .val, .test arrays of trial indices. .val is a
        length-0 array if defaults.val_frac == 0.

    Invariant:
        Calling this twice with the same `image_ids_per_trial` and the same
        (test_frac, seed) — but DIFFERENT val_frac values — yields the same
        .test array. This is what guarantees Phase 1 and Phase 2 evaluate on
        the same held-out trials.
    """
    if image_ids_per_trial.ndim != 1:
        raise ValueError(
            f"image_ids_per_trial must be 1-D, got shape {image_ids_per_trial.shape}"
        )

    unique_images = np.unique(image_ids_per_trial)
    n_images = len(unique_images)

    # Deterministic shuffle of image IDs by seed only — independent of
    # val_frac. This is what enforces the cross-phase test-set invariant.
    rng = np.random.default_rng(defaults.seed)
    shuffled = unique_images.copy()
    rng.shuffle(shuffled)

    n_test = _round_count(n_images, defaults.test_frac)
    n_val = _round_count(n_images, defaults.val_frac)
    n_train = n_images - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"After taking test ({n_test}) and val ({n_val}) out of {n_images} "
            f"unique images, no train images left."
        )

    # Order: [train | val | test]. Test is the LAST `n_test` images so its
    # identity depends only on (seed, test_frac), not on val_frac.
    train_images = shuffled[:n_train]
    val_images = shuffled[n_train : n_train + n_val]
    test_images = shuffled[n_train + n_val :]

    train_idx = _trials_for_images(image_ids_per_trial, train_images)
    val_idx = _trials_for_images(image_ids_per_trial, val_images)
    test_idx = _trials_for_images(image_ids_per_trial, test_images)
    return Splits(
        train=train_idx,
        val=val_idx,
        test=test_idx,
        image_ids_test=np.sort(test_images),
    )


def _round_count(n_total: int, frac: float) -> int:
    """Round to nearest int, clamped to [0, n_total]."""
    return int(np.clip(round(frac * n_total), 0, n_total))


def _trials_for_images(
    image_ids_per_trial: np.ndarray,
    selected_images: np.ndarray,
) -> np.ndarray:
    """Return sorted trial indices whose image_id is in `selected_images`."""
    if len(selected_images) == 0:
        return np.empty(0, dtype=np.int64)
    mask = np.isin(image_ids_per_trial, selected_images)
    return np.nonzero(mask)[0].astype(np.int64)
