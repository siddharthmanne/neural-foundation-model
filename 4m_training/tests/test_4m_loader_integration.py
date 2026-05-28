"""End-to-end 4M data pipeline integration tests on synthetic shards.

Exercises the full 4M WebDataset chain that
``build_wds_fm_pretraining_dataloader`` runs on real training data, except
we substitute synthetic tars and stop before the masking / Dirichlet step
(which requires a text tokenizer we don't need for storage-format validation).

The pipeline under test:
    ResampledShards
    -> multi_tarfile_samples       (zips tars by __key__)
    -> wds.shuffle                  (intra-shard shuffle)
    -> wds.decode(wds_decoder)      (npy -> ndarray)
    -> remove_extensions            (strip ".npy")
    -> filter_metadata              (drop __key__ / __url__)
    -> tok_to_int64                 (tok_* npy int16 -> int64)
    -> rename_modalities            (folder name -> modality name)

These are the same primitives 4M chains internally; we just stop before
``UnifiedMasking``. The point of this test layer is to prove the
Option 1 storage layout traverses the entire 4M decode path cleanly.
"""

from __future__ import annotations

import io
import sys
import tarfile
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import webdataset as wds

from neural_constants import MEG_GRID_SHAPE, MEG_TRIAL_SHAPE, TOK_RGB_TOKENS_PER_IMAGE

from fourm.data.unified_datasets import (
    filter_metadata,
    multi_tarfile_samples,
    remove_extensions,
    rename_modalities,
    tok_to_int64,
    wds_decoder,
)
# fourm's local map shadows builtin; expose as keyless_map for clarity.
from fourm.data.unified_datasets import map as keyless_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neural_trial_transform import (  # noqa: E402
    MEG_TRIAL_SHAPE,
    MegTrialSampleTransform,
    is_placeholder,
)


# ---------- shard fixture helpers ----------------------------------------


def _make_tar(path: Path, entries: Iterable[tuple[str, str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, ext, data in entries:
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _synthesize_three_shards(root: Path, n_per_shard: int = 20) -> list[str]:
    """Write 3 shards with n_per_shard image ids each.

    Modalities written:
      tok_rgb, tok_depth, tok_meg, meg_mask, crop_settings

    Note: ``meg_mask`` deliberately does NOT start with ``tok_`` because 4M's
    ``tok_to_int64`` filter casts any ``tok_*`` value to int64 — which would
    silently convert our uint8 presence flag.

    Half the MEG entries (alternating ids) are sentinel placeholders.
    """
    all_ids: list[str] = []
    for shard_idx in range(3):
        ids = [
            f"{shard_idx * n_per_shard + j + 1:09d}" for j in range(n_per_shard)
        ]
        all_ids.extend(ids)

        # Dense tokenized vision modalities.
        _make_tar(
            root / f"tok_rgb/shard_{shard_idx:03d}.tar",
            [
                (
                    image_id,
                    "npy",
                    _npy_bytes(
                        np.full((TOK_RGB_TOKENS_PER_IMAGE,), int(image_id) % 100, dtype=np.int16)
                    ),
                )
                for image_id in ids
            ],
        )
        _make_tar(
            root / f"tok_depth/shard_{shard_idx:03d}.tar",
            [
                (image_id, "npy", _npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)))
                for image_id in ids
            ],
        )

        # MEG: alternate real / placeholder.
        meg_entries: list[tuple[str, str, bytes]] = []
        mask_entries: list[tuple[str, str, bytes]] = []
        for j, image_id in enumerate(ids):
            if j % 2 == 0:
                # Real: variable trial count.
                n_trials = 1 + (j % 4)
                arr = np.full(
                    (n_trials, *MEG_TRIAL_SHAPE),
                    int(image_id) % 256,
                    dtype=np.int16,
                )
                meg_entries.append((image_id, "npy", _npy_bytes(arr)))
                mask_entries.append(
                    (image_id, "npy", _npy_bytes(np.array([1], dtype=np.uint8)))
                )
            else:
                meg_entries.append(
                    (
                        image_id,
                        "npy",
                        _npy_bytes(np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)),
                    )
                )
                mask_entries.append(
                    (image_id, "npy", _npy_bytes(np.array([0], dtype=np.uint8)))
                )
        _make_tar(root / f"tok_meg/shard_{shard_idx:03d}.tar", meg_entries)
        _make_tar(root / f"meg_mask/shard_{shard_idx:03d}.tar", mask_entries)

        # Crop settings: not used here but typically required by 4M for tok_*.
        _make_tar(
            root / f"crop_settings/shard_{shard_idx:03d}.tar",
            [
                (
                    image_id,
                    "npy",
                    _npy_bytes(np.zeros((1, 5), dtype=np.float32)),
                )
                for image_id in ids
            ],
        )
    return all_ids


def _fail_handler(exc: Exception) -> bool:
    raise exc


@pytest.fixture
def shard_root(tmp_path: Path) -> Path:
    return tmp_path / "things"


# ---------- tests --------------------------------------------------------


class TestLoaderIntegration:
    """Real 4M wds primitives chained on synthetic Option-1 shards."""

    def test_loader_yields_batches_with_all_modalities(self, shard_root: Path):
        all_ids = _synthesize_three_shards(shard_root, n_per_shard=20)

        url_template = (
            f"{shard_root}/[tok_rgb,tok_depth,tok_meg,meg_mask]/"
            "shard_{000..002}.tar"
        )

        pipeline = wds.DataPipeline(
            wds.SimpleShardList(url_template),
            partial(multi_tarfile_samples, handler=_fail_handler),
            wds.decode(wds_decoder),
            wds.map(remove_extensions),
            keyless_map(filter_metadata),
            keyless_map(tok_to_int64),
        )

        seen_keys: list[str] = []
        for sample in pipeline:
            assert set(sample) == {
                "tok_rgb",
                "tok_depth",
                "tok_meg",
                "meg_mask",
            }, f"missing modality; got {sorted(sample)}"
            assert sample["tok_rgb"].dtype == np.int64
            assert sample["tok_depth"].dtype == np.int64
            assert sample["tok_meg"].dtype == np.int64  # 'tok_' triggers int64
            assert sample["meg_mask"].dtype == np.uint8
            # MEG trial array still has its (n_trials, 16, 8, 4) shape.
            assert sample["tok_meg"].ndim == 4
            assert sample["tok_meg"].shape[1:] == MEG_TRIAL_SHAPE
        # No explicit keys here since filter_metadata drops __key__.

    def test_loader_paired_by_image_id(self, shard_root: Path):
        """Pair tok_rgb and tok_meg by __key__ before filter_metadata drops it."""
        _synthesize_three_shards(shard_root, n_per_shard=5)

        url_template = (
            f"{shard_root}/[tok_rgb,tok_meg]/shard_{{000..002}}.tar"
        )
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(url_template),
            partial(multi_tarfile_samples, handler=_fail_handler),
            wds.decode(wds_decoder),
            wds.map(remove_extensions),
        )

        # Each sample MUST have a __key__ that matches the image_id encoded
        # in tok_rgb (we wrote tok_rgb values = int(image_id) % 100).
        for sample in pipeline:
            image_id = sample["__key__"]
            rgb_marker = sample["tok_rgb"]
            assert isinstance(rgb_marker, np.ndarray)
            assert int(rgb_marker[0]) == int(image_id) % 100

    def test_loader_handles_seven_modality_brace(self, shard_root: Path):
        """Brace expansion with 5 modalities (we synthesized 5) runs end-to-end."""
        _synthesize_three_shards(shard_root, n_per_shard=10)
        url_template = (
            f"{shard_root}/[tok_rgb,tok_depth,tok_meg,meg_mask,"
            f"crop_settings]/shard_{{000..002}}.tar"
        )
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(url_template),
            partial(multi_tarfile_samples, handler=_fail_handler),
            wds.decode(wds_decoder),
            wds.map(remove_extensions),
            keyless_map(filter_metadata),
        )
        count = 0
        for sample in pipeline:
            assert {
                "tok_rgb",
                "tok_depth",
                "tok_meg",
                "meg_mask",
                "crop_settings",
            } == set(sample)
            count += 1
        assert count == 30, f"expected 3 shards x 10 = 30 samples; got {count}"

    def test_sentinel_samples_have_zero_mask_flag(self, shard_root: Path):
        """Verify our 'placeholder' MEG is paired with mask=0 throughout."""
        _synthesize_three_shards(shard_root, n_per_shard=10)
        url_template = (
            f"{shard_root}/[tok_meg,meg_mask]/shard_{{000..002}}.tar"
        )
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(url_template),
            partial(multi_tarfile_samples, handler=_fail_handler),
            wds.decode(wds_decoder),
            wds.map(remove_extensions),
        )

        for sample in pipeline:
            meg = sample["tok_meg"]
            mask = int(sample["meg_mask"][0])
            sentinel = is_placeholder(meg, MEG_TRIAL_SHAPE)
            assert sentinel == (mask == 0), (
                f"sentinel/mask disagreement for {sample['__key__']}: "
                f"sentinel={sentinel}, mask={mask}"
            )

    def test_neural_output_heads_materialized_and_coherent(self, shard_root: Path):
        """Real pipeline: rename fan-out + splitter produce coherent per-head targets.

        Each MEG trial is a distinct constant, so we can assert all four RVQ heads land
        on the *same* trial (coherence) and have the right flat shapes/dtypes.
        """
        import fourm_neural_modalities  # noqa: F401  (registers output modalities)
        from fourm.data import unified_datasets as ud
        from fourm_dataloader import (
            _extend_modality_paths,
            patch_pretrain_utils,
            unpatch_pretrain_utils,
        )
        from neural_constants import (
            EEG_MODALITY,
            EEG_TOKENS_PER_TRIAL,
            EEG_TRIAL_SHAPE,
            MEG_POSITIONS_PER_TRIAL,
            MEG_RVQ_MODALITIES,
        )
        from train_4m import _build_modality_info

        root = shard_root
        ids = ["000000001", "000000002"]
        n_trials = 5
        for mod_dir, entries in (
            ("tok_rgb", [(i, _npy_bytes(np.full((TOK_RGB_TOKENS_PER_IMAGE,), 1, np.int16))) for i in ids]),
            # MEG trial t is the constant (t+1) everywhere -> identifies the picked trial.
            ("tok_meg", [(i, _npy_bytes(np.stack([np.full(MEG_TRIAL_SHAPE, t + 1, np.int16) for t in range(n_trials)]))) for i in ids]),
            ("tok_eeg", [(i, _npy_bytes(np.stack([np.full(EEG_TRIAL_SHAPE, t + 1, np.int16) for t in range(n_trials)]))) for i in ids]),
            ("meg_mask", [(i, _npy_bytes(np.array([1], np.uint8))) for i in ids]),
            ("eeg_mask", [(i, _npy_bytes(np.array([1], np.uint8))) for i in ids]),
        ):
            _make_tar(root / f"{mod_dir}/shard_000.tar", [(k, "npy", b) for k, b in entries])

        out_domains = ["tok_rgb", *MEG_RVQ_MODALITIES, EEG_MODALITY]
        mod_info = _build_modality_info(out_domains)
        paths = _extend_modality_paths(mod_info)

        patch_pretrain_utils()  # installs _rename_modalities + train-mode splitter
        try:
            pipeline = wds.DataPipeline(
                wds.SimpleShardList(
                    f"{root}/[tok_rgb,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{{000..000}}.tar"
                ),
                partial(multi_tarfile_samples, handler=_fail_handler),
                wds.decode(wds_decoder),
                wds.map(remove_extensions),
                keyless_map(filter_metadata),
                keyless_map(tok_to_int64),
                keyless_map(partial(ud.rename_modalities, modality_paths=paths)),
            )
            n = 0
            for sample in pipeline:
                n += 1
                for mod in MEG_RVQ_MODALITIES:
                    assert sample[mod].shape == (MEG_POSITIONS_PER_TRIAL,), mod
                assert sample[EEG_MODALITY].shape == (EEG_TOKENS_PER_TRIAL,)
                # All four MEG heads must come from one trial -> one shared constant.
                picked = {int(np.unique(sample[mod])[0]) for mod in MEG_RVQ_MODALITIES}
                assert len(picked) == 1, f"RVQ heads decohered across trials: {picked}"
        finally:
            unpatch_pretrain_utils()
        assert n == 2

    def test_meg_trial_transform_runs_on_pipeline_output(
        self, shard_root: Path
    ):
        """Hook MegTrialSampleTransform into the pipeline.map and verify shape."""
        _synthesize_three_shards(shard_root, n_per_shard=4)

        meg_transform = MegTrialSampleTransform(training=True, seed=0)

        def apply_meg_transform(sample):
            arr = sample["tok_meg"]
            tokens, valid = meg_transform(arr)
            sample["tok_meg"] = tokens
            sample["tok_meg_valid"] = valid
            return sample

        url_template = (
            f"{shard_root}/[tok_rgb,tok_meg,meg_mask]/"
            f"shard_{{000..002}}.tar"
        )
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(url_template),
            partial(multi_tarfile_samples, handler=_fail_handler),
            wds.decode(wds_decoder),
            wds.map(remove_extensions),
            keyless_map(filter_metadata),
            keyless_map(apply_meg_transform),
        )

        for sample in pipeline:
            tok_meg = sample["tok_meg"]
            valid = sample["tok_meg_valid"]
            mask = int(sample["meg_mask"][0])
            assert tok_meg.shape == MEG_GRID_SHAPE
            assert tok_meg.dtype == np.int32
            assert valid == bool(mask)
