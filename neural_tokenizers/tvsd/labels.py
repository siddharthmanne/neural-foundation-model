"""TVSD → THINGS 27-way superordinate label resolver.

Pipeline:
  image_id (1-indexed, 0=catch)
  → things_imgs_<monkey>.mat :: train_imgs['class'][image_id - 1]  → concept_name
  → concept_id = 1-indexed alphabetical rank in TVSDStimulusSet/images/
  → superordinate index (0–26) via SuperordinateMapping

concept_id numbering matches the THINGS canonical convention used in
meg/data/concept_id_to_superordinate.json (1 = aardvark, alphabetical order).
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from neural_tokenizers.meg.data import SuperordinateMapping


STIMULUS_ROOT = "/scratch/users/liubr/cs375/TVSDStimulusSet"
_DEFAULT_SUPER_MAP = (
    "neural_tokenizers/meg/data/concept_id_to_superordinate.json"
)


def _load_concept_names(stimulus_mat: str) -> list[str]:
    """Return list of 22248 concept names, one per training image (0-indexed)."""
    names: list[str] = []
    with h5py.File(stimulus_mat, "r") as f:
        class_dataset = f["train_imgs"]["class"]
        for i in range(class_dataset.shape[0]):
            ref = class_dataset[i, 0]
            arr = f[ref][:]
            names.append("".join(chr(c) for c in arr.flat))
    return names


def _build_concept_id_lookup(images_dir: str) -> dict[str, int]:
    """Map concept_name → 1-indexed alphabetical rank (THINGS convention)."""
    concepts = sorted(os.listdir(images_dir))
    return {name: idx + 1 for idx, name in enumerate(concepts)}


def build_tvsd_label_map(
    monkey: str = "monkeyF",
    stimulus_root: str = STIMULUS_ROOT,
    super_map_path: str | None = None,
) -> np.ndarray:
    """Build a (22248,) int64 array mapping training image index → superordinate label.

    Args:
        monkey:          'monkeyF' or 'monkeyN'.
        stimulus_root:   Path to TVSDStimulusSet directory.
        super_map_path:  Path to concept_id_to_superordinate.json; defaults to
                         the checked-in copy under neural_tokenizers/meg/data/.

    Returns:
        labels: (22248,) int64 — superordinate index in [0, 26], or -1 for:
                  • catch trials (image_id == 0)
                  • concepts excluded from the single-category superordinate mapping
    """
    stimulus_mat = os.path.join(stimulus_root, f"things_imgs_{monkey}.mat")
    images_dir = os.path.join(stimulus_root, "images")

    concept_names = _load_concept_names(stimulus_mat)           # 22248 entries
    concept_id_lookup = _build_concept_id_lookup(images_dir)    # name → 1-based id
    super_map = SuperordinateMapping.load(super_map_path or _DEFAULT_SUPER_MAP)

    n = len(concept_names)
    labels = np.full(n, -1, dtype=np.int64)
    for i, name in enumerate(concept_names):
        cid = concept_id_lookup.get(name)
        if cid is None:
            continue
        sidx = super_map.concept_id_to_super.get(cid)
        if sidx is None:
            continue
        labels[i] = sidx
    return labels


def apply_image_ids(
    label_map: np.ndarray,
    image_ids: np.ndarray,
) -> np.ndarray:
    """Map per-trial image_ids (from ALLMAT) to superordinate labels.

    Args:
        label_map:  (22248,) output of build_tvsd_label_map().
        image_ids:  (B,) int64 from data.load_tvsd(); 1-indexed, 0=catch.

    Returns:
        (B,) int64 superordinate labels; -1 where image_id==0 or concept unmapped.
    """
    out = np.full(len(image_ids), -1, dtype=np.int64)
    for i, iid in enumerate(image_ids):
        if iid == 0:
            continue
        idx = int(iid) - 1  # convert 1-indexed to 0-indexed into label_map
        if 0 <= idx < len(label_map):
            out[i] = label_map[idx]
    return out
