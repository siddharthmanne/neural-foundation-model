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

# 27-superordinate mapping is produced by
# `modal/modal_download_things_superordinate.py` from THINGS-data canonical
# metadata (`category_mat_manual.tsv`). Modality-agnostic — shared with EEG.
DEFAULT_SUPER_MAP_LOCAL = "neural_tokenizers/meg/data/concept_id_to_superordinate.json"
DEFAULT_SUPER_MAP_REMOTE = "/project/data/things/labels/concept_id_to_superordinate.json"


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


# ---------- cross-subject + cross-rep averaging ---------------------------

def average_trials_by_image(
    X: torch.Tensor | np.ndarray,
    image_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Group trials by `image_id` and average within each group.

    Produces one signal per unique image — the canonical "cross-subject +
    cross-rep" averaging used by Experiment 2 (see
    ``notes/meg_tokenization.md``). The averaging axis collapses both
    subject and within-subject repetitions, so the output has no subject
    label per row.

    Args:
        X: ``(N, C, T)`` torch or numpy array of single-trial signals.
        image_ids: ``(N,)`` int array, trial-aligned image codes.

    Returns:
        X_avg:     ``(M, C, T)`` torch.float32 tensor (M = unique image
                   count).
        image_ids_unique: ``(M,)`` int64 numpy array, sorted ascending.

    Memory-frugal: accumulates in float64 element-wise inside `np.add.at`
    but **never materializes a float64 copy of the input**. The 88k-trial
    training tensor (~27 GB float32) is the largest object in this
    pipeline; doubling it with an f64 cast would push the container OOM.
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()  # no dtype cast — preserve source f32
    else:
        X_np = np.asarray(X)

    image_ids = np.asarray(image_ids, dtype=np.int64)
    if X_np.shape[0] != image_ids.shape[0]:
        raise ValueError(
            f"X.shape[0]={X_np.shape[0]} must match image_ids length "
            f"{image_ids.shape[0]}"
        )

    unique_ids, inverse = np.unique(image_ids, return_inverse=True)
    n_groups = len(unique_ids)
    C, T = X_np.shape[1], X_np.shape[2]

    # Sums stay in f64 for accumulator precision when a single group has
    # 40+ contributors (test images × 4 subjects × 12 reps); numpy promotes
    # f32 source elements element-wise inside np.add.at — no global cast.
    sums = np.zeros((n_groups, C, T), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(sums, inverse, X_np)
    np.add.at(counts, inverse, 1)

    X_avg_np = sums / counts.reshape(n_groups, 1, 1)
    return torch.from_numpy(X_avg_np.astype(np.float32)), unique_ids.astype(np.int64)


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


# ---------- concept_id → 27 superordinate label (THINGS-wide) -------------

@dataclass(frozen=True)
class SuperordinateMapping:
    """Lookup table from THINGS concept_id → 27-class superordinate index.

    The §5.3 linear-probe target. Produced by
    `modal/modal_download_things_superordinate.py` from canonical THINGS
    metadata (`category_mat_manual.tsv` in the ViCCo-Group/THINGS-data
    repo) — modality-independent, so the same file is used for MEG, EEG,
    and future intracortical probes.

    Single-category-only: concepts with multi-category membership are
    excluded from the mapping (recorded in the source JSON for inspection
    but `encode_image_ids` marks them invalid). This keeps the probe a
    clean K-way classification rather than multi-label.

    Attributes:
        concept_id_to_super: {concept_id (1..1854): super_index (0..26)}
        category_names:      ordered list of 27 high-level category labels
                             (e.g. "animal", "food", "vehicle", ...)
        n_categories:        27 (sanity-checked at load).
    """

    concept_id_to_super: dict[int, int]
    category_names: tuple[str, ...]
    n_categories: int

    @classmethod
    def from_json(cls, payload: dict) -> "SuperordinateMapping":
        raw = payload["concept_id_to_superordinate_index"]
        m = {int(k): int(v) for k, v in raw.items()}
        names = tuple(payload["category_names"])
        n = int(payload["n_categories"])
        if n != len(names) or n != 27:
            raise ValueError(
                f"superordinate mapping must have 27 categories with matching "
                f"name list; got n_categories={n}, len(names)={len(names)}"
            )
        return cls(concept_id_to_super=m, category_names=names, n_categories=n)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "SuperordinateMapping":
        """Read the mapping. Tries the checked-in local copy, then the
        Volume; raises with a clear hint to the producer script if missing.
        """
        if path is not None:
            return cls.from_json(json.loads(open(path).read()))
        for candidate in (DEFAULT_SUPER_MAP_LOCAL, DEFAULT_SUPER_MAP_REMOTE):
            if os.path.exists(candidate):
                return cls.from_json(json.loads(open(candidate).read()))
        raise FileNotFoundError(
            f"concept_id_to_superordinate.json not found at "
            f"({DEFAULT_SUPER_MAP_LOCAL!r}, {DEFAULT_SUPER_MAP_REMOTE!r}). "
            f"Run `modal run neural_tokenizers/meg/modal/"
            f"modal_download_things_superordinate.py::download`."
        )

    def encode_image_ids(
        self, image_ids: np.ndarray, concept_map: "ConceptMapping"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Translate trial-level (N,) image_ids → (N,) superordinate labels.

        Composes two lookups: image_id → concept_id (via `concept_map`) →
        superordinate_index. A trial is dropped (valid=False) if either
        step fails — i.e. the image is unmapped OR its concept has
        multi-category membership and was excluded from the canonical
        mapping.

        Returns:
            super_labels: (N,) int64 in [0, 27) (or -1 where invalid).
            valid_mask:   (N,) bool — True where both lookups succeeded.
        """
        out = np.full(len(image_ids), -1, dtype=np.int64)
        valid = np.zeros(len(image_ids), dtype=bool)
        for i, code in enumerate(image_ids):
            iid = int(code)
            concept = concept_map.image_id_to_concept_id.get(iid)
            if concept is None:
                continue
            super_idx = self.concept_id_to_super.get(concept)
            if super_idx is None:
                continue
            out[i] = super_idx
            valid[i] = True
        return out, valid


# ---------- animacy: 2-class probe target (derived from superordinate) ----

# Strict animate set — derived from the 27-category THINGS scheme. The
# canonical THINGS animacy ratings (Hebart 2019, ratings_animacy.csv on
# OSF) are continuous per-concept scores; we don't pull them here because
# the category-derived heuristic is sharper (THINGS' "bird" / "insect" /
# "animal" are unambiguously animate regardless of any rater score) and
# avoids a second download/version-pinning step. If we later want the
# continuous animacy variable for regression-style probing, fetching
# ratings_animacy.csv is a small additional script.
#
# Body part is excluded: a "leg" or "ear" is anatomically animal-derived
# but the THINGS-MEG stimuli are isolated images of body parts, perceived
# as objects, not as living things. Plant is excluded: not animal-like
# behaviorally, and not typically considered "animate" in object-decoding
# papers.
ANIMATE_SUPER_NAMES: frozenset[str] = frozenset({"animal", "bird", "insect"})


@dataclass(frozen=True)
class AnimacyMapping:
    """Derived 2-class animate/inanimate target.

    Maps a concept_id → 0 (inanimate) or 1 (animate) by checking whether
    the concept's superordinate falls in `ANIMATE_SUPER_NAMES`. A concept
    is marked invalid only if its superordinate is itself unknown — i.e.
    when `SuperordinateMapping` would have dropped it.

    The 27-class superordinate target gives ~5,000 trials but only
    ~5% balanced accuracy on raw (a hard linear-decoding problem).
    Animacy is a much easier task with substantial published evidence
    of MEG decoding (Cichy et al., Hebart et al.), so it serves as a
    sanity check that the probe protocol *can* detect MEG signal when
    the task is appropriately difficult.

    Strong class imbalance is expected: only `animal/bird/insect` of the
    27 categories are animate → ~13% of concepts. Balanced accuracy + the
    class-weighted CE pipeline handle this.
    """

    super_map: "SuperordinateMapping"
    animate_super_indices: frozenset[int]
    n_categories: int = 2

    @classmethod
    def from_super_map(
        cls,
        super_map: "SuperordinateMapping",
        animate_super_names: frozenset[str] = ANIMATE_SUPER_NAMES,
    ) -> "AnimacyMapping":
        unknown = animate_super_names - set(super_map.category_names)
        if unknown:
            raise ValueError(
                f"Animate-set names not found in superordinate categories: "
                f"{sorted(unknown)}. Known: {sorted(super_map.category_names)}"
            )
        animate_idx = frozenset(
            super_map.category_names.index(n) for n in animate_super_names
        )
        return cls(super_map=super_map, animate_super_indices=animate_idx)

    @property
    def category_names(self) -> tuple[str, str]:
        return ("inanimate", "animate")

    def encode_image_ids(
        self, image_ids: np.ndarray, concept_map: "ConceptMapping"
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N,) image_ids → (N,) {0=inanimate, 1=animate}, with valid mask."""
        super_labels, valid = self.super_map.encode_image_ids(image_ids, concept_map)
        out = np.zeros(len(image_ids), dtype=np.int64)
        for i, (s, ok) in enumerate(zip(super_labels, valid)):
            if ok:
                out[i] = 1 if int(s) in self.animate_super_indices else 0
        return out, valid
