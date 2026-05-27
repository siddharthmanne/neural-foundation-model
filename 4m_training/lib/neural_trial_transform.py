"""Generalized trial-sampling for stacked neural token arrays (MEG / EEG).

Each ``tok_meg/<image_id>.npy`` or ``tok_eeg/<image_id>.npy`` stores::

    arr.shape == (n_trials, *trial_shape)   # int16

For images lacking neural data, the packer writes a sentinel::

    np.full((1, *trial_shape), NEURAL_SENTINEL_VALUE, dtype=np.int16)

Pure-Python; only depends on numpy.
"""

from __future__ import annotations

import numpy as np

from neural_constants import (
    EEG_OUT_MODALITY,
    EEG_TRIAL_SHAPE,
    EEG_TOKENS_PER_TRIAL,
    MEG_GRID_SHAPE,
    MEG_RVQ_OUT_MODALITIES,
    MEG_TRIAL_SHAPE,
    MEG_TOKENS_PER_TRIAL,
    NEURAL_PLACEHOLDER_CODE,
    NEURAL_SENTINEL_VALUE,
)

# Backward-compatible aliases (prefer neural_constants in new code).
SENTINEL_VALUE: int = NEURAL_SENTINEL_VALUE
PLACEHOLDER_FILL_VALUE: int = NEURAL_PLACEHOLDER_CODE


def is_placeholder(arr: np.ndarray, trial_shape: tuple[int, ...]) -> bool:
    """Detect sentinel pattern: shape ``(1, *trial_shape)`` filled with -1."""
    expected = (1, *trial_shape)
    return bool(
        arr.ndim == len(expected)
        and tuple(arr.shape) == expected
        and (arr == NEURAL_SENTINEL_VALUE).all()
    )


class NeuralTrialSampleTransform:
    """Pick one trial per call and reshape it to a 4M-ready token array.

    The picked trial (``trial_shape``) is reshaped to ``out_shape`` — same element
    count, but laid out the way the encoder embedding expects. MEG keeps its RVQ
    axis (``(128, 4)``) so the embedding can sum residual codebooks per cell; EEG
    stays a flat ``(17,)`` sequence. See ``notes/4m_neural_modality_design.md``.

    Args:
        trial_shape: On-disk inner shape per trial (e.g. ``MEG_TRIAL_SHAPE``).
        out_shape: Reshaped per-trial output. Defaults to flat ``(prod(trial_shape),)``.
        training: If True, sample a uniform random trial; if False, always trial 0.
        seed: Optional RNG seed for reproducibility within one transform instance.
    """

    def __init__(
        self,
        trial_shape: tuple[int, ...],
        out_shape: tuple[int, ...] | None = None,
        training: bool = True,
        seed: int | None = None,
    ):
        self.trial_shape = trial_shape
        self.tokens_per_trial = int(np.prod(trial_shape))
        self.out_shape = out_shape or (self.tokens_per_trial,)
        if int(np.prod(self.out_shape)) != self.tokens_per_trial:
            raise ValueError(
                f"out_shape {self.out_shape} has a different element count than "
                f"trial_shape {trial_shape}"
            )
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __call__(self, arr: np.ndarray) -> tuple[np.ndarray, bool]:
        ndim = 1 + len(self.trial_shape)
        if arr.ndim != ndim or tuple(arr.shape[1:]) != self.trial_shape:
            raise ValueError(
                f"Trial array must have shape (n_trials, {self.trial_shape}); "
                f"got {arr.shape}"
            )

        if is_placeholder(arr, self.trial_shape):
            tokens = np.full(self.out_shape, NEURAL_PLACEHOLDER_CODE, dtype=np.int32)
            return tokens, False

        n_trials = arr.shape[0]
        idx = int(self.rng.integers(0, n_trials)) if self.training else 0
        tokens = arr[idx].astype(np.int32).reshape(self.out_shape)
        np.clip(tokens, 0, None, out=tokens)
        return tokens, True


class MegTrialSampleTransform(NeuralTrialSampleTransform):
    """MEG trial sampler: ``(n_trials, 16, 8, 4)`` -> ``(128, 4)`` (cells x RVQ layers)."""

    def __init__(self, training: bool = True, seed: int | None = None):
        super().__init__(
            trial_shape=MEG_TRIAL_SHAPE,
            out_shape=MEG_GRID_SHAPE,
            training=training,
            seed=seed,
        )


class EegTrialSampleTransform(NeuralTrialSampleTransform):
    """EEG trial sampler: ``(n_trials, 17)`` -> ``(17,)`` (flat 1D sequence)."""

    def __init__(self, training: bool = True, seed: int | None = None):
        super().__init__(trial_shape=EEG_TRIAL_SHAPE, training=training, seed=seed)


class NeuralTargetSplitter:
    """Materialize neural OUTPUT heads from one shared trial, in-place on a sample dict.

    The dataloader's ``rename_modalities`` step fans the on-disk ``tok_meg`` / ``tok_eeg``
    folder out to every output modality that reads it, so all four ``tok_meg_rvq*`` keys
    initially **alias the same** ``(n_trials, 16, 8, 4)`` array. This splitter picks **one**
    trial for that array and slices it into the four per-RVQ-layer ``(128,)`` targets — so
    the four MEG heads always describe the *same* MEG token. EEG (single codebook) is a
    single ``(17,)`` head. Absent (sentinel) data becomes the placeholder code; presence
    masking then zeroes its loss. See ``notes/4m_neural_modality_design.md`` §6.

    One pick per source per call ⇒ coherent across heads; the picked trial is exactly what
    the loss scores, because that single array is both embedded and used as the target ids.
    """

    def __init__(self, training: bool = True, seed: int | None = None):
        self._meg = MegTrialSampleTransform(training=training, seed=seed)
        self._eeg = EegTrialSampleTransform(training=training, seed=seed)

    def __call__(self, sample: dict) -> dict:
        meg_mods = [m for m in MEG_RVQ_OUT_MODALITIES if m in sample]
        if meg_mods:
            grid, _valid = self._meg(sample[meg_mods[0]])  # (128, n_rvq); one shared pick
            for q, mod in enumerate(MEG_RVQ_OUT_MODALITIES):
                if mod in sample:
                    sample[mod] = np.ascontiguousarray(grid[:, q])
        if EEG_OUT_MODALITY in sample:
            tokens, _valid = self._eeg(sample[EEG_OUT_MODALITY])  # (17,)
            sample[EEG_OUT_MODALITY] = tokens
        return sample
