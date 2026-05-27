"""Contract tests: THINGS on-disk shard formats must survive the 4M transform path.

These guard the single most damaging silent failure mode — vision pretokens
(``tok_rgb`` / ``tok_depth``) being reduced from a full ``(196,)`` grid to a
scalar. Stock ``TokTransform.image_augment`` does ``v[rand_aug_idx]``, which
assumes an on-disk augmentation axis ``(n_augs, n_tokens)``. THINGS shards are
flat ``(n_tokens,)`` (verified against the Modal ``project`` volume), so the
stock transform indexes away 195 of 196 tokens with no error.

The neural path already has shape assertions; the image path did not, which is
why a green suite still trained on one token per image. Every test here asserts
*token count*, not just dtype / vocab range.
"""

from __future__ import annotations

import io
import sys
import tarfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import fourm_neural_modalities  # noqa: F401 — registers tok_rgb/tok_depth/neural transforms
import webdataset as wds
from fourm.data.modality_info import MODALITY_TRANSFORMS
from fourm.data.modality_transforms import (
    IdentityTransform,
    UnifiedDataTransform,
)
from fourm.data.unified_datasets import (
    filter_metadata,
    map as keyless_map,
    multi_tarfile_samples,
    remove_extensions,
    tok_to_int64,
    wds_decoder,
)
from neural_constants import (
    EEG_TRIAL_SHAPE,
    MEG_TRIAL_SHAPE,
    THINGS_IMAGE_SIZE,
    TOK_DEPTH_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    TOK_RGB_VOCAB_SIZE,
)
from things_augmenter import ThingsImageAugmenter, ThingsTokTransform


# --- fixtures matching the real volume layout (flat (196,) vision tokens) ---


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _make_tar(path: Path, entries: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, data in entries:
            info = tarfile.TarInfo(name=f"{key}.npy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _write_things_shard(root: Path, key: str = "000000001") -> None:
    """Mirror the on-disk THINGS format verified on the Modal volume."""
    rng = np.random.default_rng(0)
    mods = {
        "tok_rgb": rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16),
        "tok_depth": rng.integers(0, TOK_DEPTH_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16),
        "tok_meg": rng.integers(0, 512, (3, *MEG_TRIAL_SHAPE), dtype=np.int16),
        "tok_eeg": rng.integers(0, 8192, (2, *EEG_TRIAL_SHAPE), dtype=np.int16),
        "meg_mask": np.array([1], dtype=np.uint8),
        "eeg_mask": np.array([1], dtype=np.uint8),
    }
    for mod, arr in mods.items():
        _make_tar(root / mod / "shard_000.tar", [(key, _npy_bytes(arr))])


def _decode_one(root: Path) -> dict:
    url = f"{root}/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{{000..000}}.tar"
    pipe = wds.DataPipeline(
        wds.SimpleShardList(url),
        partial(multi_tarfile_samples),
        wds.decode(wds_decoder),
        wds.map(remove_extensions),
        keyless_map(filter_metadata),
        keyless_map(tok_to_int64),
    )
    return next(iter(pipe))


def _unified_transform() -> UnifiedDataTransform:
    transforms = dict(MODALITY_TRANSFORMS)
    transforms["__key__"] = IdentityTransform()
    return UnifiedDataTransform(
        transforms_dict=transforms,
        image_augmenter=ThingsImageAugmenter(
            target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
        ),
    )


# --- direct unit tests on the transform ---


class TestThingsTokTransform:
    def test_flat_array_keeps_all_tokens(self):
        """The core fix: a flat (196,) array must come out as (196,), not a scalar."""
        flat = np.arange(TOK_RGB_TOKENS_PER_IMAGE, dtype=np.int64)
        out = ThingsTokTransform().image_augment(
            flat, crop_coords=(0, 0, 224, 224), flip=False,
            orig_size=None, target_size=(224, 224), rand_aug_idx=0,
        )
        assert tuple(out.shape) == (TOK_RGB_TOKENS_PER_IMAGE,)
        assert int(out[0]) == 0 and int(out[-1]) == TOK_RGB_TOKENS_PER_IMAGE - 1

    def test_augmentation_axis_still_selected(self):
        """CC12M-style (n_augs, 196) shards must still select one augmentation."""
        n_augs = 5
        batched = np.stack(
            [np.full(TOK_RGB_TOKENS_PER_IMAGE, i, dtype=np.int64) for i in range(n_augs)]
        )
        out = ThingsTokTransform().image_augment(
            batched, crop_coords=(0, 0, 224, 224), flip=False,
            orig_size=None, target_size=(224, 224), rand_aug_idx=2,
        )
        assert tuple(out.shape) == (TOK_RGB_TOKENS_PER_IMAGE,)
        assert int(out[0]) == 2  # selected augmentation index 2

    def test_registered_for_rgb_and_depth(self):
        from fourm.data.modality_transforms import get_transform_key

        for mod in ("tok_rgb", "tok_depth"):
            tx = MODALITY_TRANSFORMS[get_transform_key(mod)]
            assert isinstance(tx, ThingsTokTransform), f"{mod} -> {type(tx).__name__}"


# --- end-to-end through the production transform path ---


class TestVisionTokensSurvivePipeline:
    def test_rgb_keeps_196_tokens(self, tmp_path: Path):
        root = tmp_path / "things"
        _write_things_shard(root)
        out = _unified_transform()(_decode_one(root))
        assert tuple(out["tok_rgb"].shape) == (TOK_RGB_TOKENS_PER_IMAGE,)

    def test_depth_keeps_196_tokens(self, tmp_path: Path):
        root = tmp_path / "things"
        _write_things_shard(root)
        out = _unified_transform()(_decode_one(root))
        assert tuple(out["tok_depth"].shape) == (TOK_RGB_TOKENS_PER_IMAGE,)
