"""THINGS-EEG loaders for the LaBraM token cache.

Loads pre-tokenized trial arrays from the npz cache written by
modal_eeg_produce_tokens.py. No MNE dependency — the npz files contain
only numpy arrays.

NPZ schema (one file per (source, subject)):
  tokens    (N, 17) int16   8192-vocab codes, one per channel
  image_id  (N,)    <U9     9-digit zero-padded THINGS catalog id
  trial_idx (N,)    int16   repetition index within (subject, source, image)
  source    scalar  <U4     "eeg1" or "eeg2"
  subject   scalar  <U6     "sub-01" etc

Label mappings are shared with MEG (same THINGS catalog). Imported from
neural_tokenizers.meg.data lazily; candidate for refactor to
neural_tokenizers/labels/ once pipelines stabilize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .eeg_config import EEG_DATA


@dataclass(frozen=True)
class SubjectIndex:
    """Metadata for one (source, subject) cache file."""

    subject: str        # e.g. "sub-01"
    source: str         # "eeg1" or "eeg2"
    npz_path: str
    image_ids: np.ndarray   # (N,) str — 9-digit catalog ids, trial-aligned
    n_trials: int


def list_subjects(
    slug: str = EEG_DATA.default_slug,
    cache_root: str = EEG_DATA.cache_root,
    sources: tuple[str, ...] = ("eeg1", "eeg2"),
    subjects_eeg2: tuple[str, ...] = EEG_DATA.subjects_eeg2,
    subjects_eeg1: tuple[str, ...] = EEG_DATA.subjects_eeg1,
) -> list[SubjectIndex]:
    """Discover available npz cache files and eagerly load image_id arrays.

    image_ids are loaded immediately (tiny per-trial metadata); token arrays
    stay on disk until load_tokens_pooled is called.

    Returns only subjects whose npz files actually exist on disk.
    """
    root = Path(cache_root) / slug
    subject_map: dict[str, tuple[str, ...]] = {
        "eeg2": subjects_eeg2,
        "eeg1": subjects_eeg1,
    }
    out: list[SubjectIndex] = []
    for source in sources:
        for subj in subject_map.get(source, ()):
            p = root / f"{source}_{subj}.npz"
            if not p.exists():
                continue
            data = np.load(str(p), allow_pickle=True)
            image_ids: np.ndarray = data["image_id"]
            out.append(SubjectIndex(
                subject=subj,
                source=source,
                npz_path=str(p),
                image_ids=image_ids,
                n_trials=int(len(image_ids)),
            ))
    if not out:
        raise FileNotFoundError(
            f"No npz cache files found under {root}. "
            "Check that the volume is mounted and modal_eeg_produce_tokens.py "
            "has been run."
        )
    return out


def load_tokens_pooled(
    subjects: list[SubjectIndex],
    trial_indices: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load and concatenate token arrays for the given subjects.

    Args:
        subjects: from list_subjects().
        trial_indices: optional {(source, subject): int64 trial index array}.
            Pass None to load all trials from every subject.

    Returns:
        tokens:      (N_total, 17) long tensor.
        image_ids:   (N_total,) str array — 9-digit THINGS catalog ids.
        subject_ids: (N_total,) int64 array — 0-based index into `subjects`.
    """
    all_tokens: list[np.ndarray] = []
    all_image_ids: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []

    for i, s in enumerate(subjects):
        data = np.load(s.npz_path, allow_pickle=True)
        tokens = data["tokens"].astype(np.int64)   # (N, 17)
        image_ids = data["image_id"]               # (N,) str

        if trial_indices is not None:
            idx = trial_indices.get((s.source, s.subject))
            if idx is None or len(idx) == 0:
                continue
            tokens = tokens[idx]
            image_ids = image_ids[idx]
            n = len(idx)
        else:
            n = len(tokens)

        all_tokens.append(tokens)
        all_image_ids.append(image_ids)
        all_subject_ids.append(np.full(n, i, dtype=np.int64))

    if not all_tokens:
        return (
            torch.zeros(0, 17, dtype=torch.long),
            np.empty(0, dtype="<U9"),
            np.empty(0, dtype=np.int64),
        )

    return (
        torch.from_numpy(np.concatenate(all_tokens, axis=0)),
        np.concatenate(all_image_ids, axis=0),
        np.concatenate(all_subject_ids, axis=0),
    )


def image_ids_to_int(image_ids: np.ndarray) -> np.ndarray:
    """Convert 9-digit zero-padded string catalog ids to int64.

    The EEG npz stores image_id as '<U9' strings ("000001234"). The MEG
    concept and superordinate mappings use integer keys. This is the bridge.
    """
    return np.array([int(x) for x in image_ids], dtype=np.int64)


# ---------- cross-subject + cross-rep averaging ----------------------------

def average_waveforms_by_image(
    X: torch.Tensor,
    image_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Group trials by image_id and average waveforms within each group.

    EEG analogue of meg/data.py::average_trials_by_image. Operates on raw
    waveform tensors (N, C, T) — call this BEFORE tokenizing, not after.
    Use for Experiment 1.5 (averaged eval input, no retrain). Subject labels
    are invalid after averaging; the caller should set subject_ids to -1.

    Args:
        X: (N, C, T) float tensor of single-trial waveforms. For THINGS-EEG2
           this is (N, 17, 200) at 200 Hz, calibrated to µV.
        image_ids: (N,) str array of 9-digit THINGS catalog ids, trial-aligned.

    Returns:
        X_avg: (M, C, T) float32 tensor, M = number of unique image_ids.
        image_ids_unique: (M,) str array, sorted ascending.
    """
    X_np = X.detach().cpu().numpy()
    unique_ids, inverse = np.unique(image_ids, return_inverse=True)
    n_groups = len(unique_ids)
    C, T = X_np.shape[1], X_np.shape[2]

    sums = np.zeros((n_groups, C, T), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(sums, inverse, X_np)
    np.add.at(counts, inverse, 1)

    X_avg = sums / counts.reshape(n_groups, 1, 1)
    return torch.from_numpy(X_avg.astype(np.float32)), unique_ids

