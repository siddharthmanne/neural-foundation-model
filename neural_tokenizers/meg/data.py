"""THINGS-MEG loaders shared across all MEG tokenizer phases.

Design goal: keep tokenizer code agnostic to "where does the data live, how
do we read .fif, how do we pool across subjects." Every MEG tokenizer (μ,
Cho2026, BrainOmni) calls into this module to get a torch tensor it can
push to GPU.

Numpy / torch boundary:
  - .fif → numpy (MNE has no torch backend, always returns float64).
  - As soon as data is in RAM, convert to torch and let the caller pick
    device. Everything downstream is torch-on-GPU.
  - Image IDs and trial-index arrays stay as numpy. They are int label
    arrays used as Python indices into the data tensor — they never run
    on GPU and never have tensor math applied, so torch would only add
    .tolist() conversions.

Memory budget reminder (see meg/CLAUDE.md §3):
  4 subjects × 27,048 trials × 271 ch × 281 samples × 4 B ≈ 33 GB float32.
  → never materialize all at once. Materialize:
      * the test subset for eval (≤ ~5k trials ≈ 1.5 GB), AND
      * a small random subsample for calibration (≤ ~2k trials ≈ 0.6 GB).

This module imports MNE lazily so the rest of the package (and unit tests
on synthetic tensors) can be imported on a laptop without MNE installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
import json
import os

import numpy as np
import torch

from .meg_config import MEG_DATA


# Default location of the image_id → concept_id mapping. Produced by
# `modal/modal_download_things_labels.py`; checked into git as well as
# living on the Volume. The local path is what eval code reads; the Volume
# copy is the source of truth in case the local copy is missing.
DEFAULT_CONCEPT_MAP_LOCAL = "neural_tokenizers/meg/data/image_id_to_concept.json"
DEFAULT_CONCEPT_MAP_REMOTE = "/project/data/things-meg/labels/image_id_to_concept.json"


# ---------- types ---------------------------------------------------------

@dataclass(frozen=True)
class SubjectIndex:
    """One subject's epoch metadata + image-id array.

    Stores the path to the primary -epo.fif file (MNE auto-chains the
    -1.fif / -2.fif continuations). Image IDs are eagerly loaded — they
    come from `epochs.events[:, 2]` which is tiny — but the actual MEG
    tensor stays on disk until `load_trials` is called.
    """

    subject: str
    epo_path: str
    image_ids: np.ndarray   # (n_trials,) int64 — trial-aligned image codes


# ---------- subject discovery + metadata ----------------------------------

def list_subjects(data_dir: str = MEG_DATA.data_dir) -> list[SubjectIndex]:
    """Discover preprocessed THINGS-MEG subjects and load their image IDs.

    Returns one SubjectIndex per subject in MEG_DATA.subjects, in declared
    order. Raises FileNotFoundError if any subject's primary -epo.fif is
    missing.

    Only `epochs.events` is materialized here (tiny; one int triple per
    trial). The MEG data tensor stays on disk until `load_trials`.
    """
    import mne
    mne.set_log_level("ERROR")

    out: list[SubjectIndex] = []
    for subj in MEG_DATA.subjects:
        epo_path = os.path.join(data_dir, f"preprocessed_{subj}-epo.fif")
        if not os.path.exists(epo_path):
            available = sorted(glob(os.path.join(data_dir, "preprocessed_*-epo.fif")))
            raise FileNotFoundError(
                f"Missing primary epoch file: {epo_path}\n"
                f"Available primaries in {data_dir}: {available}"
            )
        epochs = mne.read_epochs(epo_path, preload=False, verbose="ERROR")
        out.append(
            SubjectIndex(
                subject=subj,
                epo_path=epo_path,
                image_ids=epochs.events[:, 2].astype(np.int64),
            )
        )
    return out


# ---------- materializing trials ------------------------------------------

def load_trials(
    subject_idx: SubjectIndex,
    trial_indices: np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, np.ndarray]:
    """Read selected trials for one subject into a torch tensor on CPU.

    The caller is responsible for moving X to GPU. Keeping the loader on
    CPU lets the same function feed both CPU smoke tests and GPU runs.

    Args:
        subject_idx: from list_subjects().
        trial_indices: int array, indices into subject's epoch list.
        dtype: torch dtype for the output (default float32; MNE returns f64).

    Returns:
        X:         (N, C, T) torch tensor on CPU (N = len(trial_indices)).
        image_ids: (N,) numpy int64 array.
    """
    import mne
    mne.set_log_level("ERROR")

    if len(trial_indices) == 0:
        C, T = MEG_DATA.trial_shape
        return torch.empty(0, C, T, dtype=dtype), np.empty(0, dtype=np.int64)

    epochs = mne.read_epochs(subject_idx.epo_path, preload=False, verbose="ERROR")
    data = epochs.get_data(item=trial_indices.tolist())   # numpy float64 (n, C, T)
    X = torch.from_numpy(np.asarray(data)).to(dtype)
    return X, subject_idx.image_ids[trial_indices]


def load_trials_pooled(
    subject_indices: list[SubjectIndex],
    indices_per_subject: dict[str, np.ndarray],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Concatenate trials from multiple subjects into one CPU tensor.

    Args:
        subject_indices: from list_subjects().
        indices_per_subject: {subject_name: trial_indices_array}.
        dtype: torch dtype for the output.

    Returns:
        X:           (N_total, C, T) torch tensor on CPU.
        image_ids:   (N_total,) int64 numpy — trial-aligned image codes.
        subject_id:  (N_total,) int64 numpy — index into MEG_DATA.subjects.

    The subject_id column lets a downstream caller (e.g. a per-subject probe)
    recover provenance without having to pass subject metadata around.
    """
    xs: list[torch.Tensor] = []
    img_chunks: list[np.ndarray] = []
    subj_chunks: list[np.ndarray] = []
    name_to_pos = {s.subject: i for i, s in enumerate(subject_indices)}

    for s in subject_indices:
        idx = indices_per_subject.get(s.subject)
        if idx is None or len(idx) == 0:
            continue
        X_s, img_s = load_trials(s, idx, dtype=dtype)
        xs.append(X_s)
        img_chunks.append(img_s)
        subj_chunks.append(np.full(len(img_s), name_to_pos[s.subject], dtype=np.int64))

    if not xs:
        C, T = MEG_DATA.trial_shape
        return (
            torch.empty(0, C, T, dtype=dtype),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return torch.cat(xs, dim=0), np.concatenate(img_chunks), np.concatenate(subj_chunks)


# ---------- subsampling for calibration -----------------------------------

def sample_train_trials(
    subject_indices: list[SubjectIndex],
    train_indices_per_subject: dict[str, np.ndarray],
    n_sample: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniformly subsample N trials from the pooled train split.

    Used by μ-transform calibration: percentile + max-abs estimates converge
    fast in sample size, so loading a few hundred MB of trials is enough to
    fit the calibration parameters to high accuracy. Subsampling sidesteps
    the streaming-quantile complexity entirely (CLAUDE.md §5 / discussion).

    Args:
        subject_indices: from list_subjects().
        train_indices_per_subject: per-subject train trial indices.
        n_sample: total trials to draw across all subjects, pooled.
        seed: rng seed.
        dtype: output torch dtype.

    Returns:
        X: (n_sample, C, T) torch tensor on CPU. Caller moves to GPU.
    """
    # Build a flat list of (subject_pos, trial_idx) pairs, then sample.
    pairs: list[tuple[int, int]] = []
    for pos, s in enumerate(subject_indices):
        idx = train_indices_per_subject.get(s.subject)
        if idx is None:
            continue
        for t in idx:
            pairs.append((pos, int(t)))
    if not pairs:
        raise ValueError("No trials available to subsample (train_indices empty).")

    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(pairs))
    chosen = rng.choice(len(pairs), size=n_sample, replace=False)

    # Group chosen pairs by subject for a single .fif read per subject.
    per_subj: dict[str, list[int]] = {}
    for k in chosen:
        pos, t = pairs[k]
        per_subj.setdefault(subject_indices[pos].subject, []).append(t)

    xs: list[torch.Tensor] = []
    for s in subject_indices:
        ts = per_subj.get(s.subject)
        if not ts:
            continue
        X_s, _ = load_trials(s, np.asarray(ts, dtype=np.int64), dtype=dtype)
        xs.append(X_s)
    return torch.cat(xs, dim=0)


# ---------- image_id → concept_id mapping (THINGS, 1854 concepts) ---------

@dataclass(frozen=True)
class ConceptMapping:
    """Lookup table from THINGS image trigger codes → dense concept indices.

    `image_id_to_concept_id[image_id]` gives a THINGS category number in
    [1, 1854] (the canonical THINGS concept ID). For the linear probe we
    additionally re-encode to dense [0, K) indices so cross-entropy works
    without holes in the label range.

    Attributes:
        image_id_to_concept_id: dict[int, int] — direct from events.tsv.
        dense_index:            dict[int, int] — concept_id → [0, K).
        concept_ids:            sorted unique THINGS concept IDs (length K).
        n_concepts:             K (= 1854 for the full THINGS pool).
    """

    image_id_to_concept_id: dict[int, int]
    dense_index: dict[int, int]
    concept_ids: np.ndarray
    n_concepts: int

    @classmethod
    def from_json(cls, payload: dict) -> "ConceptMapping":
        raw = payload["image_id_to_concept_id"]
        m = {int(k): int(v) for k, v in raw.items()}
        concept_ids = np.array(sorted(set(m.values())), dtype=np.int64)
        dense_index = {int(c): i for i, c in enumerate(concept_ids)}
        return cls(
            image_id_to_concept_id=m,
            dense_index=dense_index,
            concept_ids=concept_ids,
            n_concepts=len(concept_ids),
        )

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "ConceptMapping":
        """Read the mapping file. Tries the local checked-in copy first,
        then the Modal Volume path; raises if neither exists.
        """
        if path is not None:
            return cls.from_json(json.loads(open(path).read()))
        for candidate in (DEFAULT_CONCEPT_MAP_LOCAL, DEFAULT_CONCEPT_MAP_REMOTE):
            if os.path.exists(candidate):
                return cls.from_json(json.loads(open(candidate).read()))
        raise FileNotFoundError(
            f"image_id_to_concept.json not found at any of "
            f"({DEFAULT_CONCEPT_MAP_LOCAL!r}, {DEFAULT_CONCEPT_MAP_REMOTE!r}). "
            f"Run `modal run neural_tokenizers/meg/modal/modal_download_things_labels.py::download`."
        )

    def encode(self, image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translate (N,) image_ids → (N,) dense concept indices.

        Returns:
            dense_labels: (N,) int64 — index into `self.concept_ids`, suitable
                          for cross-entropy with `num_classes=n_concepts`.
            valid_mask:   (N,) bool — True where the image_id was found in
                          the mapping. Callers may want to filter unknown
                          codes (should be rare; logs a warning if any).
        """
        out = np.empty(len(image_ids), dtype=np.int64)
        valid = np.ones(len(image_ids), dtype=bool)
        for i, code in enumerate(image_ids):
            concept = self.image_id_to_concept_id.get(int(code))
            if concept is None:
                valid[i] = False
                out[i] = -1
            else:
                out[i] = self.dense_index[concept]
        return out, valid
