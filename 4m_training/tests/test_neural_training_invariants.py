"""Regression tests for GPU training hazards under the symmetric-neural design.

The original Modal crash (``scatter gather kernel index out of bounds``) was caused by
``tok_meg`` / ``tok_eeg`` being ``seq_token``: stock ``sequence_token_mask`` injected
text-tokenizer sentinel ids (>30k) into small-vocab neural embeddings. That class of bug
is now **structurally impossible** — neural modalities are ``neural_grid`` and masked with
``image_mask`` (no text sentinels). These tests guard the remaining hazards:

  H1 — Neural masking keeps ids in [0, vocab) (no sentinel injection on the grid path).
  H2 — Raw int16 RVQ codes (sentinels / out-of-range) are clamped by the trial transforms.
  H3 — Vision pretoken ids stay in range for the img embeddings.
  H4 — The THINGS center-crop augmenter is wired by ``patch_pretrain_utils``.
  H5 — ``_build_modality_info`` sets img ``max_tokens`` and leaves neural grids (128/17).
  H6 — Presence flags are not ``tok_*`` prefixed (so ``tok_to_int64`` won't touch them).
  H7 — End-to-end: shard -> rename+splitter -> transform -> mask -> embedding lookup is safe.
  H8 — Shipped smoke config alphas match its domain counts.
"""

from __future__ import annotations

import io
import sys
import tarfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from repo_paths import REPO_ROOT as _REPO, TRAINING_DIR

sys.path.insert(0, str(TRAINING_DIR))

import fourm_neural_modalities  # noqa: F401
from fourm.data import pretrain_utils
from fourm.data import unified_datasets as ud
from fourm.data.masking import UnifiedMasking
from fourm.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from fourm.data.modality_transforms import (
    CropSettingsTransform,
    IdentityTransform,
    UnifiedDataTransform,
)
from fourm.data.pretrain_utils import setup_sampling_mod_info
from fourm.data.unified_datasets import (
    filter_metadata,
    map as keyless_map,
    multi_tarfile_samples,
    remove_extensions,
    tok_to_int64,
    wds_decoder,
)
from fourm_dataloader import (
    _extend_modality_paths,
    patch_pretrain_utils,
    unpatch_pretrain_utils,
)
from fourm_neural_transforms import EegTokTransform, MegTokTransform
from neural_constants import (
    EEG_MODALITY,
    EEG_TOKENS_PER_TRIAL,
    EEG_TRIAL_SHAPE,
    EEG_VOCAB_SIZE,
    MEG_POSITIONS_PER_TRIAL,
    MEG_RVQ_MODALITIES,
    MEG_TRIAL_SHAPE,
    MEG_VOCAB_SIZE,
    NEURAL_GRID_TYPE,
    THINGS_CENTER_CROP,
    THINGS_IMAGE_SIZE,
    TOK_DEPTH_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    TOK_RGB_VOCAB_SIZE,
)
from neural_masking import PresenceAwareUnifiedMasking
from things_augmenter import ThingsImageAugmenter
from train_4m import _build_modality_info

_TOK = _REPO / "external/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
_OOB_RGB_CODE = TOK_RGB_VOCAB_SIZE + 616  # deliberately invalid pretoken id for H3
_MEG0 = MEG_RVQ_MODALITIES[0]
_VISION_AND_NEURAL = ["tok_rgb", "tok_depth", *MEG_RVQ_MODALITIES, EEG_MODALITY]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(_TOK))


def _masking_modality_info(domains: list[str]) -> dict:
    """Modality dict with Dirichlet alphas set (required by ``UnifiedMasking``)."""
    cfg = {
        "in_domains": "-".join(sorted(domains)),
        "out_domains": "-".join(sorted(domains)),
        "input_alphas": "1.0",
        "target_alphas": "1.0",
    }
    full = _build_modality_info(domains, input_size=THINGS_IMAGE_SIZE)
    mod_info, _ = setup_sampling_mod_info(cfg, full)
    return mod_info


def _presence_masking(modality_info: dict, rng_range=(64, 64)) -> PresenceAwareUnifiedMasking:
    return PresenceAwareUnifiedMasking(
        modality_info=modality_info, text_tokenizer=_text_tokenizer(),
        input_tokens_range=rng_range, target_tokens_range=rng_range,
    )


def _make_tar(path: Path, entries: list[tuple[str, str, bytes]]) -> None:
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


def _assert_vocab_safe(tensor: torch.Tensor, vocab_size: int, label: str) -> None:
    t = tensor.detach().long()
    assert t.numel() > 0, label
    assert int(t.min()) >= 0, f"{label}: negative id {int(t.min())}"
    assert int(t.max()) < vocab_size, f"{label}: max id {int(t.max())} >= vocab {vocab_size}"


def _synth_things_shard(root: Path, meg_codes: np.ndarray, eeg_codes: np.ndarray) -> None:
    """One image with configurable raw MEG/EEG trial arrays + vision + masks."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    depth = rng.integers(0, TOK_DEPTH_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    meg = meg_codes.reshape(1, *MEG_TRIAL_SHAPE).astype(np.int16)
    eeg = eeg_codes.reshape(1, *EEG_TRIAL_SHAPE).astype(np.int16)
    one = np.array([1], dtype=np.uint8)
    for mod, arr in (
        ("tok_rgb", rgb), ("tok_depth", depth), ("tok_meg", meg), ("tok_eeg", eeg),
        ("meg_mask", one), ("eeg_mask", one),
    ):
        _make_tar(root / f"{mod}/shard_000.tar", [("000000001", "npy", _npy_bytes(arr))])


def _decode_with_splitter(root: Path) -> dict:
    """Full pipeline incl. rename + neural splitter -> per-modality tensors (one sample)."""
    paths = _extend_modality_paths(_build_modality_info(_VISION_AND_NEURAL))
    pipe = __import__("webdataset").DataPipeline(
        __import__("webdataset").SimpleShardList(
            f"{root}/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{{000..000}}.tar"
        ),
        partial(multi_tarfile_samples),
        __import__("webdataset").decode(wds_decoder),
        __import__("webdataset").map(remove_extensions),
        keyless_map(filter_metadata),
        keyless_map(tok_to_int64),
        keyless_map(partial(ud.rename_modalities, modality_paths=paths)),
    )
    return next(iter(pipe))


# ---------------------------------------------------------------------------
# H1 — neural_grid masking keeps ids in vocab (no sentinel injection)
# ---------------------------------------------------------------------------


class TestHypothesis1NeuralGridMaskingVocabSafe:
    @pytest.mark.parametrize("seed", range(20))
    def test_meg_grid_masking_stays_in_vocab(self, seed: int):
        rng = np.random.default_rng(seed)
        mod_info = _masking_modality_info([_MEG0, "tok_rgb"])
        meg = torch.from_numpy(rng.integers(0, MEG_VOCAB_SIZE, (MEG_POSITIONS_PER_TRIAL,))).long()
        out = _presence_masking(mod_info)({
            _MEG0: meg,
            "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE),
            "meg_mask": torch.tensor([1]),
        })
        _assert_vocab_safe(out[_MEG0]["tensor"], MEG_VOCAB_SIZE, _MEG0)

    @pytest.mark.parametrize("seed", range(20))
    def test_eeg_grid_masking_stays_in_vocab(self, seed: int):
        rng = np.random.default_rng(seed)
        mod_info = _masking_modality_info([EEG_MODALITY])
        eeg = torch.from_numpy(rng.integers(0, EEG_VOCAB_SIZE, (EEG_TOKENS_PER_TRIAL,))).long()
        out = _presence_masking(mod_info)({EEG_MODALITY: eeg, "eeg_mask": torch.tensor([1])})
        _assert_vocab_safe(out[EEG_MODALITY]["tensor"], EEG_VOCAB_SIZE, EEG_MODALITY)

    def test_absent_neural_tensor_still_vocab_safe(self):
        mod_info = _masking_modality_info([_MEG0])
        out = _presence_masking(mod_info)({
            _MEG0: torch.zeros(MEG_POSITIONS_PER_TRIAL, dtype=torch.long),
            "meg_mask": torch.tensor([0]),
        })
        _assert_vocab_safe(out[_MEG0]["tensor"], MEG_VOCAB_SIZE, f"absent {_MEG0}")

    def test_neural_modalities_are_grid_not_seq_token(self):
        """The fix itself: neural must never be seq_token (that path injects sentinels)."""
        for mod in (*MEG_RVQ_MODALITIES, EEG_MODALITY):
            assert MODALITY_INFO[mod]["type"] == NEURAL_GRID_TYPE


# ---------------------------------------------------------------------------
# H2 — trial transform clamps raw int16 codes
# ---------------------------------------------------------------------------


class TestHypothesis2TrialTransformClamping:
    def test_placeholder_never_emits_negative_ids(self):
        sentinel = np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)
        tokens, valid = MegTokTransform(training=False).sampler(sentinel)
        assert valid is False
        assert tokens.min() >= 0 and tokens.max() < MEG_VOCAB_SIZE

    def test_meg_transform_clamps_hot_int16_codes(self):
        arr = np.full(MEG_TRIAL_SHAPE, 9000, dtype=np.int16).reshape(1, *MEG_TRIAL_SHAPE)
        assert int(MegTokTransform(training=False).preprocess(arr).max()) < MEG_VOCAB_SIZE

    def test_eeg_transform_clamps_hot_int16_codes(self):
        arr = np.full(EEG_TRIAL_SHAPE, 99999, dtype=np.int16).reshape(1, *EEG_TRIAL_SHAPE)
        assert int(EegTokTransform(training=False).preprocess(arr).max()) < EEG_VOCAB_SIZE

    @pytest.mark.parametrize("n_trials", [1, 2, 4])
    def test_multi_trial_sampling_in_vocab(self, n_trials: int):
        arr = np.random.randint(0, MEG_VOCAB_SIZE, (n_trials, *MEG_TRIAL_SHAPE), dtype=np.int16)
        out = MegTokTransform(training=True, seed=42).preprocess(arr)
        _assert_vocab_safe(out, MEG_VOCAB_SIZE, "multi-trial meg")


# ---------------------------------------------------------------------------
# H3 — vision pretoken range
# ---------------------------------------------------------------------------


class TestHypothesis3VisionTokenRange:
    @pytest.mark.parametrize("seed", range(15))
    def test_img_masking_rgb_depth_in_vocab(self, seed: int):
        rng = np.random.default_rng(seed)
        mod_info = _masking_modality_info(["tok_rgb", "tok_depth"])
        out = _presence_masking(mod_info)({
            "tok_rgb": torch.from_numpy(rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,))).long(),
            "tok_depth": torch.from_numpy(rng.integers(0, TOK_DEPTH_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,))).long(),
        })
        _assert_vocab_safe(out["tok_rgb"]["tensor"], TOK_RGB_VOCAB_SIZE, "tok_rgb")
        _assert_vocab_safe(out["tok_depth"]["tensor"], TOK_DEPTH_VOCAB_SIZE, "tok_depth")

    def test_oob_rgb_codes_crash_embedding(self):
        """Secondary risk: int16 shard values above vocab must be caught upstream."""
        mod_info = _masking_modality_info(["tok_rgb"])
        out = _presence_masking(mod_info)({
            "tok_rgb": torch.full((TOK_RGB_TOKENS_PER_IMAGE,), _OOB_RGB_CODE, dtype=torch.long)
        })
        with pytest.raises(IndexError):
            nn.Embedding(TOK_RGB_VOCAB_SIZE, 8)(out["tok_rgb"]["tensor"])


# ---------------------------------------------------------------------------
# H4 — augmenter patch
# ---------------------------------------------------------------------------


class TestHypothesis4AugmenterPatch:
    def test_pretrain_utils_augmenter_is_things_when_patched(self):
        patch_pretrain_utils()
        assert pretrain_utils.PreTokenizedImageAugmenter is ThingsImageAugmenter

    def test_things_augmenter_none_crop_settings(self):
        aug = ThingsImageAugmenter(target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb")
        coords, flip, _orig, _target, _idx = aug({}, None)
        assert coords == THINGS_CENTER_CROP[:4]
        assert flip == 0


# ---------------------------------------------------------------------------
# H5 — modality_info max_tokens (img square rule must skip neural grids)
# ---------------------------------------------------------------------------


class TestHypothesis5MaxTokensSetup:
    def test_build_modality_info_sets_img_and_keeps_grid_tokens(self):
        info = _build_modality_info(_VISION_AND_NEURAL, input_size=THINGS_IMAGE_SIZE)
        assert info["tok_rgb"]["max_tokens"] == TOK_RGB_TOKENS_PER_IMAGE
        assert info["tok_depth"]["max_tokens"] == TOK_RGB_TOKENS_PER_IMAGE
        assert info[_MEG0]["max_tokens"] == MEG_POSITIONS_PER_TRIAL  # 128, not a square
        assert info[EEG_MODALITY]["max_tokens"] == EEG_TOKENS_PER_TRIAL  # 17

    def test_build_modality_info_does_not_mutate_global_registry(self):
        import copy

        before = copy.deepcopy(MODALITY_INFO["tok_rgb"])
        _build_modality_info(_VISION_AND_NEURAL)
        assert MODALITY_INFO["tok_rgb"]["max_tokens"] == before["max_tokens"]


# ---------------------------------------------------------------------------
# H6 — presence flag naming / dtype
# ---------------------------------------------------------------------------


class TestHypothesis6PresenceFlagNaming:
    def test_meg_mask_not_tok_prefixed(self):
        assert "meg_mask" in MODALITY_INFO
        assert not MODALITY_INFO["meg_mask"]["path"].startswith("tok_")

    def test_tok_to_int64_skips_meg_mask(self):
        out = tok_to_int64({"meg_mask": np.array([1], dtype=np.uint8), "tok_rgb": np.array([1])})
        assert out["meg_mask"].dtype == np.uint8
        assert out["tok_rgb"].dtype == np.int64


# ---------------------------------------------------------------------------
# H7 — end-to-end pipeline -> mask -> embedding lookup
# ---------------------------------------------------------------------------


class TestHypothesis7PipelineEmbeddingSafety:
    def test_pipeline_materializes_neural_modalities(self, tmp_path: Path):
        root = tmp_path / "things"
        _synth_things_shard(
            root,
            np.random.randint(0, MEG_VOCAB_SIZE, MEG_TRIAL_SHAPE),
            np.random.randint(0, EEG_VOCAB_SIZE, EEG_TRIAL_SHAPE),
        )
        patch_pretrain_utils()
        try:
            sample = _decode_with_splitter(root)
        finally:
            unpatch_pretrain_utils()
        for mod in MEG_RVQ_MODALITIES:
            assert sample[mod].shape == (MEG_POSITIONS_PER_TRIAL,)
        assert sample[EEG_MODALITY].shape == (EEG_TOKENS_PER_TRIAL,)

    @pytest.mark.parametrize("seed", range(8))
    def test_full_mask_then_embedding_lookup(self, seed: int, tmp_path: Path):
        rng = np.random.default_rng(seed)
        root = tmp_path / f"things_{seed}"
        # Pathological raw int16 values; the trial transforms must clamp them.
        _synth_things_shard(
            root,
            rng.integers(-5, 2000, MEG_TRIAL_SHAPE, dtype=np.int16),
            rng.integers(-3, 9000, EEG_TRIAL_SHAPE, dtype=np.int16),
        )
        patch_pretrain_utils()
        try:
            sample = _decode_with_splitter(root)
            transforms = dict(MODALITY_TRANSFORMS)
            transforms["crop_settings"] = CropSettingsTransform()
            transforms["__key__"] = IdentityTransform()
            tensors = UnifiedDataTransform(
                transforms_dict=transforms,
                image_augmenter=ThingsImageAugmenter(
                    target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
                ),
            )(sample)
            mod_info = _masking_modality_info(_VISION_AND_NEURAL)
            masked = _presence_masking(mod_info)(tensors)
        finally:
            unpatch_pretrain_utils()

        nn.Embedding(MEG_VOCAB_SIZE, 8)(masked[_MEG0]["tensor"])
        nn.Embedding(EEG_VOCAB_SIZE, 8)(masked[EEG_MODALITY]["tensor"])
        nn.Embedding(TOK_RGB_VOCAB_SIZE, 8)(masked["tok_rgb"]["tensor"])
        nn.Embedding(TOK_DEPTH_VOCAB_SIZE, 8)(masked["tok_depth"]["tensor"])


# ---------------------------------------------------------------------------
# H8 — smoke config alphas match domain counts
# ---------------------------------------------------------------------------


class TestHypothesis8SamplingModInfo:
    def test_smoke_things_alphas_match_domain_count(self):
        import yaml

        ds = yaml.safe_load((TRAINING_DIR / "configs" / "4m_smoke_things_data.yaml").read_text())
        ds = ds["train"]["datasets"]["things"]
        n_in = len(ds["in_domains"].split("-"))
        n_out = len(ds["out_domains"].split("-"))
        assert len(ds["input_alphas"].split("-")) in (1, n_in)
        assert len(ds["target_alphas"].split("-")) in (1, n_out)
        mod_info, weights = setup_sampling_mod_info(ds, _build_modality_info(_VISION_AND_NEURAL))
        assert weights is None
        assert len(mod_info) == n_in
