"""Regression tests for Modal GPU training failures and weakly covered integration paths.

The Modal ``prod_things`` smoke hit ``scatter gather kernel index out of bounds`` on the
first forward pass. Likely causes (each has tests below):

  H1 — Text sentinel leakage (primary):
      ``tok_meg`` / ``tok_eeg`` are ``seq_token`` but stock ``sequence_token_mask`` injects
      text-tokenizer sentinel ids (often >30k) into tensors fed to small-vocab embeddings.

  H2 — Raw RVQ codes out of range:
      int16 shard values <0 (sentinel -1) or >= vocab_size before / after trial sampling.

  H3 — Vision pretoken ids out of range:
      ``tok_rgb`` / ``tok_depth`` int16 codes must stay in [0, vocab_size) for img embeddings.

  H4 — Augmenter patch not applied:
      ``get_train_dataloader`` binds ``PreTokenizedImageAugmenter`` at import; patch must
      replace ``pretrain_utils.PreTokenizedImageAugmenter`` or THINGS shards without
      ``crop_settings`` crash.

  H5 — ``max_tokens`` unset for img modalities:
      ``MODALITY_INFO`` leaves ``max_tokens: None``; training must call ``_build_modality_info``.

  H6 — Presence flags vs ``tok_to_int64``:
      ``meg_mask`` must not use a ``tok_*`` prefix (would break uint8 semantics).

  H7 — End-to-end index safety:
      Synthetic WDS sample → transforms → masking → ``nn.Embedding`` lookup on CPU.
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
from fourm.data.image_augmenter import PreTokenizedImageAugmenter
from fourm.data.masking import UnifiedMasking
from fourm.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from fourm.data.modality_transforms import (
    CaptionTransform,
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
from fourm_dataloader import patch_pretrain_utils, unpatch_pretrain_utils
from fourm_neural_transforms import EegTokTransform, MegTokTransform
from neural_constants import (
    EEG_CODE_MAX,
    EEG_TOKENS_PER_TRIAL,
    EEG_TRIAL_SHAPE,
    EEG_VOCAB_SIZE,
    MEG_CODE_MAX,
    MEG_GRID_SHAPE,
    MEG_POSITIONS_PER_TRIAL,
    MEG_TOKENS_PER_TRIAL,
    MEG_TRIAL_SHAPE,
    MEG_VOCAB_SIZE,
    THINGS_CENTER_CROP,
    THINGS_IMAGE_SIZE,
    TOK_DEPTH_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    TOK_RGB_VOCAB_SIZE,
)
from neural_masking import PresenceAwareUnifiedMasking, _NEURAL_FLAT_TOKEN_MODS
from things_augmenter import DEFAULT_CROP_SETTINGS, ThingsImageAugmenter
from train_4m import _build_modality_info

_TOK = _REPO / "external/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
# Deliberately invalid pretoken id for H3 hazard tests (above TOK_RGB vocab).
_OOB_RGB_CODE = TOK_RGB_VOCAB_SIZE + 616


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(_TOK))


def _things_modality_info() -> dict:
    domains = ["tok_rgb", "tok_depth", "tok_meg", "tok_eeg"]
    return _build_modality_info(domains, input_size=THINGS_IMAGE_SIZE)


def _masking_modality_info(
    domains: list[str],
    input_alphas: str = "1.0",
    target_alphas: str = "1.0",
) -> dict:
    """Modality dict with Dirichlet alphas set (required by ``UnifiedMasking``)."""
    cfg = {
        "in_domains": "-".join(sorted(domains)),
        "out_domains": "-".join(sorted(domains)),
        "input_alphas": input_alphas,
        "target_alphas": target_alphas,
    }
    full = _build_modality_info(domains, input_size=THINGS_IMAGE_SIZE)
    mod_info, _ = setup_sampling_mod_info(cfg, full)
    return mod_info


def _smoke_dataset_config(root: Path) -> dict:
    return {
        "type": "multimodal",
        "use_wds": True,
        "data_path": (
            f"{root}/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/"
            "shard_{000..000}.tar"
        ),
        "in_domains": "tok_rgb-tok_depth-tok_meg-tok_eeg",
        "out_domains": "tok_rgb-tok_depth-tok_meg-tok_eeg",
        "main_augment_domain": "tok_rgb",
        "tok_train_aug": False,
        "input_alphas": "1.0-1.0-1.0-1.0",
        "target_alphas": "1.0-1.0-1.0-1.0",
        "aligned_captions": False,
    }


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
    """All indices must lie in [0, vocab_size)."""
    t = tensor.detach().long()
    assert t.numel() > 0, label
    assert int(t.min()) >= 0, f"{label}: negative id {int(t.min())}"
    assert int(t.max()) < vocab_size, (
        f"{label}: max id {int(t.max())} >= vocab_size {vocab_size}"
    )


def _presence_masking(
    modality_info: dict,
    input_range: tuple[int, int] = (64, 64),
    target_range: tuple[int, int] = (64, 64),
) -> PresenceAwareUnifiedMasking:
    return PresenceAwareUnifiedMasking(
        modality_info=modality_info,
        text_tokenizer=_text_tokenizer(),
        input_tokens_range=input_range,
        target_tokens_range=target_range,
    )


def _stock_masking(modality_info: dict) -> UnifiedMasking:
    return UnifiedMasking(
        modality_info=modality_info,
        text_tokenizer=_text_tokenizer(),
        input_tokens_range=(64, 64),
        target_tokens_range=(64, 64),
    )


def _synth_things_shard(root: Path, meg_codes: np.ndarray, eeg_codes: np.ndarray) -> None:
    """One image with configurable MEG/EEG RVQ arrays."""
    rgb = np.random.randint(0, TOK_RGB_VOCAB_SIZE, size=(TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    depth = np.random.randint(0, TOK_DEPTH_VOCAB_SIZE, size=(TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    meg = meg_codes.reshape(1, *MEG_TRIAL_SHAPE).astype(np.int16)
    eeg = eeg_codes.reshape(1, *EEG_TRIAL_SHAPE).astype(np.int16)
    _make_tar(root / "tok_rgb/shard_000.tar", [("000000001", "npy", _npy_bytes(rgb))])
    _make_tar(root / "tok_depth/shard_000.tar", [("000000001", "npy", _npy_bytes(depth))])
    _make_tar(root / "tok_meg/shard_000.tar", [("000000001", "npy", _npy_bytes(meg))])
    _make_tar(root / "tok_eeg/shard_000.tar", [("000000001", "npy", _npy_bytes(eeg))])
    _make_tar(
        root / "meg_mask/shard_000.tar",
        [("000000001", "npy", _npy_bytes(np.array([1], dtype=np.uint8)))],
    )
    _make_tar(
        root / "eeg_mask/shard_000.tar",
        [("000000001", "npy", _npy_bytes(np.array([1], dtype=np.uint8)))],
    )


# ---------------------------------------------------------------------------
# H1 — masking / sentinel leakage (Modal root cause)
# ---------------------------------------------------------------------------


class TestHypothesis1MaskingSentinelLeakage:
    def test_stock_sequence_token_mask_injects_text_vocab_ids(self):
        """Documents the bug: stock path mixes in text sentinels / pad ids."""
        stock = _stock_masking(
            {
                "tok_meg": {
                    "type": "seq_token",
                    "min_tokens": 0,
                    "max_tokens": MEG_VOCAB_SIZE,
                    "input_alphas": [1.0],
                    "target_alphas": [1.0],
                }
            }
        )
        meg_codes = torch.arange(MEG_VOCAB_SIZE, dtype=torch.long)
        unsafe = False
        for _ in range(30):
            out = stock.sequence_token_mask(
                meg_codes, MEG_VOCAB_SIZE, 32, 32, "random", vocab_offset=0
            )
            ids = {int(x) for x in out["tensor"].tolist()}
            if ids & stock.sentinel_ids or any(i >= MEG_VOCAB_SIZE for i in ids):
                unsafe = True
                break
        assert unsafe, "stock sequence_token_mask never produced OOB/sentinel ids in 30 tries"

    @pytest.mark.parametrize("seed", range(20))
    def test_neural_masking_meg_stays_in_vocab_many_seeds(self, seed: int):
        rng = np.random.default_rng(seed)
        mod_info = _masking_modality_info(["tok_meg", "tok_rgb"])
        masking = _presence_masking(mod_info)
        meg = torch.from_numpy(rng.integers(0, MEG_VOCAB_SIZE, size=(MEG_VOCAB_SIZE,))).long()
        out = masking(
            {
                "tok_meg": meg,
                "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE),
                "meg_mask": torch.tensor([1]),
            }
        )
        _assert_vocab_safe(out["tok_meg"]["tensor"], MEG_VOCAB_SIZE, "tok_meg")

    @pytest.mark.parametrize("seed", range(20))
    def test_neural_masking_eeg_stays_in_vocab_many_seeds(self, seed: int):
        rng = np.random.default_rng(seed)
        mod_info = _masking_modality_info(["tok_eeg"])
        masking = _presence_masking(mod_info)
        eeg = torch.from_numpy(rng.integers(0, EEG_VOCAB_SIZE, size=(EEG_TOKENS_PER_TRIAL,))).long()
        out = masking({"tok_eeg": eeg, "eeg_mask": torch.tensor([1])})
        _assert_vocab_safe(out["tok_eeg"]["tensor"], EEG_VOCAB_SIZE, "tok_eeg")

    def test_neural_flat_mods_declared(self):
        assert _NEURAL_FLAT_TOKEN_MODS == frozenset({"tok_meg", "tok_eeg"})

    def test_absent_meg_tensor_still_vocab_safe(self):
        mod_info = _masking_modality_info(["tok_meg"])
        out = _presence_masking(mod_info)(
            {"tok_meg": torch.zeros(MEG_VOCAB_SIZE, dtype=torch.long), "meg_mask": torch.tensor([0])}
        )
        _assert_vocab_safe(out["tok_meg"]["tensor"], MEG_VOCAB_SIZE, "absent tok_meg")


# ---------------------------------------------------------------------------
# H2 — trial transform / raw codes
# ---------------------------------------------------------------------------


class TestHypothesis2TrialTransformAndClamping:
    def test_placeholder_never_emits_negative_ids(self):
        sentinel = np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)
        tokens, valid = MegTokTransform(training=False).sampler(sentinel)
        assert valid is False
        assert tokens.min() >= 0
        assert tokens.max() < MEG_VOCAB_SIZE

    def test_meg_transform_clamps_hot_int16_codes(self):
        hot = np.full(MEG_TRIAL_SHAPE, 9000, dtype=np.int16)
        arr = hot.reshape(1, *MEG_TRIAL_SHAPE)
        out = MegTokTransform(training=False).preprocess(arr)
        assert int(out.max()) < MEG_VOCAB_SIZE

    def test_eeg_transform_clamps_hot_int16_codes(self):
        hot = np.full(EEG_TRIAL_SHAPE, 99999, dtype=np.int16)
        arr = hot.reshape(1, *EEG_TRIAL_SHAPE)
        out = EegTokTransform(training=False).preprocess(arr)
        assert int(out.max()) < EEG_VOCAB_SIZE

    @pytest.mark.parametrize("trial_idx", [0, 1, 3])
    def test_multi_trial_sampling_in_vocab(self, trial_idx: int):
        n = trial_idx + 1
        arr = np.random.randint(0, MEG_VOCAB_SIZE, size=(n, *MEG_TRIAL_SHAPE), dtype=np.int16)
        out = MegTokTransform(training=True, seed=42).preprocess(arr)
        _assert_vocab_safe(out, MEG_VOCAB_SIZE, "multi-trial meg")


# ---------------------------------------------------------------------------
# H3 — vision pretoken range + img masking
# ---------------------------------------------------------------------------


class TestHypothesis3VisionTokenRange:
    @pytest.mark.parametrize("seed", range(15))
    def test_img_masking_rgb_depth_in_vocab(self, seed: int):
        rng = np.random.default_rng(seed)
        mod_info = _masking_modality_info(["tok_rgb", "tok_depth"])
        masking = _presence_masking(mod_info)
        out = masking(
            {
                "tok_rgb": torch.from_numpy(
                    rng.integers(0, TOK_RGB_VOCAB_SIZE, size=(TOK_RGB_TOKENS_PER_IMAGE,))
                ).long(),
                "tok_depth": torch.from_numpy(
                    rng.integers(0, TOK_DEPTH_VOCAB_SIZE, size=(TOK_RGB_TOKENS_PER_IMAGE,))
                ).long(),
            }
        )
        _assert_vocab_safe(out["tok_rgb"]["tensor"], TOK_RGB_VOCAB_SIZE, "tok_rgb")
        _assert_vocab_safe(out["tok_depth"]["tensor"], TOK_DEPTH_VOCAB_SIZE, "tok_depth")

    def test_int16_rgb_codes_at_or_above_vocab_crash_embedding(self):
        """Secondary Modal risk: int16 shard values can exceed 16384 / 8192."""
        mod_info = _masking_modality_info(["tok_rgb"])
        out = _presence_masking(mod_info)(
            {"tok_rgb": torch.full((TOK_RGB_TOKENS_PER_IMAGE,), _OOB_RGB_CODE, dtype=torch.long)}
        )
        emb = nn.Embedding(TOK_RGB_VOCAB_SIZE, 8)
        with pytest.raises(IndexError):
            emb(out["tok_rgb"]["tensor"])


# ---------------------------------------------------------------------------
# H4 — augmenter patch
# ---------------------------------------------------------------------------


class TestHypothesis4AugmenterPatch:
    def test_pretrain_utils_augmenter_is_things_when_patched(self):
        """``train_4m`` / ``patch_pretrain_utils`` must wire THINGS center-crop augmenter."""
        patch_pretrain_utils()
        assert pretrain_utils.PreTokenizedImageAugmenter is ThingsImageAugmenter
        assert pretrain_utils.PreTokenizedImageAugmenter.__name__ == "ThingsImageAugmenter"

    def test_things_augmenter_none_crop_settings(self):
        aug = ThingsImageAugmenter(target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb")
        coords, flip, orig, target, idx = aug({}, None)
        assert coords == (THINGS_CENTER_CROP[0], THINGS_CENTER_CROP[1], THINGS_CENTER_CROP[2], THINGS_CENTER_CROP[3])
        assert flip == 0


# ---------------------------------------------------------------------------
# H5 — modality_info max_tokens
# ---------------------------------------------------------------------------


class TestHypothesis5MaxTokensSetup:
    def test_build_modality_info_sets_img_max_tokens(self):
        info = _things_modality_info()
        assert info["tok_rgb"]["max_tokens"] == TOK_RGB_TOKENS_PER_IMAGE
        assert info["tok_depth"]["max_tokens"] == TOK_RGB_TOKENS_PER_IMAGE
        assert info["tok_meg"]["max_tokens"] == MEG_POSITIONS_PER_TRIAL  # 128 grid cells

    def test_build_modality_info_does_not_mutate_global_registry(self):
        import copy

        before = copy.deepcopy(MODALITY_INFO["tok_rgb"])
        _build_modality_info(["tok_rgb", "tok_depth", "tok_meg", "tok_eeg"])
        assert MODALITY_INFO["tok_rgb"]["max_tokens"] == before["max_tokens"]

    def test_raw_modality_info_rgb_alias_inherits_none_max_tokens(self):
        assert MODALITY_INFO["tok_rgb@224"]["max_tokens"] is None
        assert MODALITY_INFO["tok_rgb"]["max_tokens"] is None


# ---------------------------------------------------------------------------
# H6 — meg_mask naming / dtype
# ---------------------------------------------------------------------------


class TestHypothesis6PresenceFlagNaming:
    def test_meg_mask_not_tok_prefixed_in_modality_info(self):
        assert "meg_mask" in MODALITY_INFO
        assert not MODALITY_INFO["meg_mask"]["path"].startswith("tok_")

    def test_tok_to_int64_skips_meg_mask(self):
        sample = {"meg_mask": np.array([1], dtype=np.uint8), "tok_rgb": np.array([1])}
        out = tok_to_int64(sample)
        assert out["meg_mask"].dtype == np.uint8
        assert out["tok_rgb"].dtype == np.int64


# ---------------------------------------------------------------------------
# H7 — pipeline + embedding lookup (CPU surrogate for CUDA scatter)
# ---------------------------------------------------------------------------


class TestHypothesis7PipelineEmbeddingSafety:
    @pytest.fixture
    def decoded_sample(self, tmp_path: Path):
        root = tmp_path / "things"
        codes = np.random.randint(0, MEG_VOCAB_SIZE, size=MEG_TRIAL_SHAPE, dtype=np.int16)
        _synth_things_shard(root, codes, np.random.randint(0, EEG_VOCAB_SIZE, size=EEG_TRIAL_SHAPE))
        pipe = __import__("webdataset").DataPipeline(
            __import__("webdataset").SimpleShardList(
                _smoke_dataset_config(root)["data_path"]
            ),
            partial(multi_tarfile_samples),
            __import__("webdataset").decode(wds_decoder),
            __import__("webdataset").map(remove_extensions),
            keyless_map(filter_metadata),
            keyless_map(tok_to_int64),
        )
        return next(iter(pipe))

    def test_unified_transform_produces_neural_tensors(self, decoded_sample):
        transforms = dict(MODALITY_TRANSFORMS)
        transforms["crop_settings"] = CropSettingsTransform()
        transforms["__key__"] = IdentityTransform()
        aug = ThingsImageAugmenter(target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb")
        udt = UnifiedDataTransform(transforms_dict=transforms, image_augmenter=aug)
        out = udt(decoded_sample)
        assert out["tok_meg"].shape == MEG_GRID_SHAPE  # (128, 4) cells x RVQ
        assert out["tok_eeg"].shape == (EEG_TOKENS_PER_TRIAL,)
        _assert_vocab_safe(out["tok_meg"], MEG_VOCAB_SIZE, "pipeline meg")
        _assert_vocab_safe(out["tok_eeg"], EEG_VOCAB_SIZE, "pipeline eeg")

    def test_full_mask_then_embedding_lookup(self, decoded_sample):
        transforms = dict(MODALITY_TRANSFORMS)
        transforms["__key__"] = IdentityTransform()
        udt = UnifiedDataTransform(
            transforms_dict=transforms,
            image_augmenter=ThingsImageAugmenter(
                target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
            ),
        )
        tensors = udt(decoded_sample)
        mod_info = _masking_modality_info(
            ["tok_rgb", "tok_depth", "tok_meg", "tok_eeg"],
            input_alphas="1.0-1.0-1.0-1.0",
            target_alphas="1.0-1.0-1.0-1.0",
        )
        masked = _presence_masking(mod_info, (64, 64), (64, 64))(tensors)

        meg_emb = nn.Embedding(MEG_VOCAB_SIZE, 32, padding_idx=0)
        eeg_emb = nn.Embedding(EEG_VOCAB_SIZE, 32, padding_idx=0)
        rgb_emb = nn.Embedding(TOK_RGB_VOCAB_SIZE, 32)
        depth_emb = nn.Embedding(TOK_DEPTH_VOCAB_SIZE, 32)

        _assert_vocab_safe(masked["tok_meg"]["tensor"], MEG_VOCAB_SIZE, "masked meg")
        meg_emb(masked["tok_meg"]["tensor"])
        eeg_emb(masked["tok_eeg"]["tensor"])
        rgb_emb(masked["tok_rgb"]["tensor"])
        depth_emb(masked["tok_depth"]["tensor"])

    @pytest.mark.parametrize("seed", range(10))
    def test_stress_random_codes_through_mask_and_embed(self, seed: int, tmp_path: Path):
        rng = np.random.default_rng(seed)
        root = tmp_path / f"things_{seed}"
        # Include pathological int16 values; transforms must clamp.
        meg_raw = rng.integers(-5, 2000, size=MEG_TRIAL_SHAPE, dtype=np.int16)
        eeg_raw = rng.integers(-3, 9000, size=EEG_TRIAL_SHAPE, dtype=np.int16)
        _synth_things_shard(root, meg_raw, eeg_raw)

        pipe = __import__("webdataset").DataPipeline(
            __import__("webdataset").SimpleShardList(
                _smoke_dataset_config(root)["data_path"]
            ),
            partial(multi_tarfile_samples),
            __import__("webdataset").decode(wds_decoder),
            __import__("webdataset").map(remove_extensions),
            keyless_map(filter_metadata),
            keyless_map(tok_to_int64),
        )
        sample = next(iter(pipe))
        transforms = dict(MODALITY_TRANSFORMS)
        transforms["__key__"] = IdentityTransform()
        tensors = UnifiedDataTransform(
            transforms_dict=transforms,
            image_augmenter=ThingsImageAugmenter(
                target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
            ),
        )(sample)

        cfg = _smoke_dataset_config(root)
        mod_info = _masking_modality_info(
            ["tok_rgb", "tok_depth", "tok_meg", "tok_eeg"],
            input_alphas="1.0-1.0-1.0-1.0",
            target_alphas="1.0-1.0-1.0-1.0",
        )
        masked = _presence_masking(mod_info)(tensors)

        nn.Embedding(MEG_VOCAB_SIZE, 8)(masked["tok_meg"]["tensor"])
        nn.Embedding(EEG_VOCAB_SIZE, 8)(masked["tok_eeg"]["tensor"])


# ---------------------------------------------------------------------------
# H8 — setup_sampling_mod_info + alphas (smoke config shape)
# ---------------------------------------------------------------------------


class TestHypothesis8SamplingModInfo:
    def test_smoke_things_alphas_match_domain_count(self):
        import yaml

        cfg_path = TRAINING_DIR / "configs" / "4m_smoke_things_data.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        ds = cfg["train"]["datasets"]["things"]
        n_in = len(ds["in_domains"].split("-"))
        n_out = len(ds["out_domains"].split("-"))
        mod_info, weights = setup_sampling_mod_info(
            ds, _things_modality_info()
        )
        assert len(mod_info) == n_in
        assert weights is None
        assert mod_info["tok_meg"]["input_alphas"] == [1.0]
        assert len(ds["input_alphas"].split("-")) in (1, n_in)
        assert len(ds["target_alphas"].split("-")) in (1, n_out)
