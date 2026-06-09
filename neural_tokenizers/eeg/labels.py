"""THINGS label mappings for the EEG eval probe.

These classes are copies of the equivalent classes in neural_tokenizers/meg/data.py.
They live here so the EEG eval container does not need to import the meg package
(which eagerly loads BrainOmniTokenizer -> einops, adding heavy deps to a
CPU-only container).

The JSON source files are the same files used by the MEG eval — modality-agnostic
THINGS metadata checked into git under neural_tokenizers/meg/data/.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# JSON files live next to the meg package's data/ folder.
# Use __file__-relative paths so the lookup works in both local and Modal
# container contexts regardless of working directory.
_LABELS_DIR = Path(__file__).resolve().parent.parent / "meg" / "data"

_CONCEPT_MAP_LOCAL = str(_LABELS_DIR / "image_id_to_concept.json")
_CONCEPT_MAP_REMOTE = "/project/data/things-meg/labels/image_id_to_concept.json"

_SUPER_MAP_LOCAL = str(_LABELS_DIR / "concept_id_to_superordinate.json")
_SUPER_MAP_REMOTE = "/project/data/things/labels/concept_id_to_superordinate.json"


def _read_json(local: str, remote: str, label: str) -> dict:
    for path in (local, remote):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"{label} not found at {local!r} or {remote!r}."
    )


# ---------- image_id → concept_id -----------------------------------------

@dataclass(frozen=True)
class ConceptMapping:
    """Lookup table from THINGS image_ids → dense concept indices."""

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
        if path is not None:
            with open(path) as f:
                return cls.from_json(json.load(f))
        return cls.from_json(
            _read_json(_CONCEPT_MAP_LOCAL, _CONCEPT_MAP_REMOTE, "image_id_to_concept.json")
        )

    def encode(self, image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,) image_ids → (N,) dense concept indices + valid mask."""
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


# ---------- concept_id → 27 superordinate labels --------------------------

@dataclass(frozen=True)
class SuperordinateMapping:
    """Lookup table from THINGS concept_id → 27-class superordinate index."""

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
                f"Expected 27 categories; got n_categories={n}, len(names)={len(names)}"
            )
        return cls(concept_id_to_super=m, category_names=names, n_categories=n)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "SuperordinateMapping":
        if path is not None:
            with open(path) as f:
                return cls.from_json(json.load(f))
        return cls.from_json(
            _read_json(_SUPER_MAP_LOCAL, _SUPER_MAP_REMOTE, "concept_id_to_superordinate.json")
        )

    def encode_image_ids(
        self, image_ids: np.ndarray, concept_map: ConceptMapping
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N,) image_ids → (N,) superordinate labels + valid mask."""
        out = np.full(len(image_ids), -1, dtype=np.int64)
        valid = np.zeros(len(image_ids), dtype=bool)
        for i, code in enumerate(image_ids):
            concept = concept_map.image_id_to_concept_id.get(int(code))
            if concept is None:
                continue
            super_idx = self.concept_id_to_super.get(concept)
            if super_idx is None:
                continue
            out[i] = super_idx
            valid[i] = True
        return out, valid


# ---------- animacy: 2-class derived from superordinate -------------------

ANIMATE_SUPER_NAMES: frozenset[str] = frozenset({"animal", "bird", "insect"})


@dataclass(frozen=True)
class AnimacyMapping:
    """Derived 2-class animate/inanimate target (from superordinate)."""

    super_map: SuperordinateMapping
    animate_super_indices: frozenset[int]
    n_categories: int = 2

    @classmethod
    def from_super_map(
        cls,
        super_map: SuperordinateMapping,
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
        self, image_ids: np.ndarray, concept_map: ConceptMapping
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N,) image_ids → (N,) {0=inanimate, 1=animate} + valid mask."""
        super_labels, valid = self.super_map.encode_image_ids(image_ids, concept_map)
        out = np.zeros(len(image_ids), dtype=np.int64)
        for i, (s, ok) in enumerate(zip(super_labels, valid)):
            if ok:
                out[i] = 1 if int(s) in self.animate_super_indices else 0
        return out, valid
