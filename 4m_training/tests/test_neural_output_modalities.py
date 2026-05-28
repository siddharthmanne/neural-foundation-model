"""Contract + integration tests for neural (MEG/EEG) OUTPUT modalities.

These lock the design in notes/4m_neural_modality_design.md §6 as realized:

  * MEG is predicted with 4 parallel heads (one per RVQ layer) — registered as
    4 ``neural_grid`` modalities ``tok_meg_rvq0..3`` (vocab 512, 128-cell grid).
  * EEG is predicted with one head — ``tok_eeg_out`` (vocab 8192, 17 tokens).
  * The output modalities use a custom ``neural_grid`` type so they route through
    the *parallel* decoder branch (not the autoregressive seq_token path) and are
    *not* clobbered by the trainer's square ``max_tokens`` rule for ``img``.
  * The 4 MEG heads come from the SAME sampled trial (coherent split).
  * The existing input-only ``tok_meg`` / ``tok_eeg`` modalities are untouched.

Fast + CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourm_neural_modalities  # noqa: F401 — registers modalities at import
from fourm_neural_embeddings import EegDecoderEmbedding, MegRVQDecoderEmbedding
from neural_constants import (
    EEG_OUT_MODALITY,
    EEG_TOKENS_PER_TRIAL,
    EEG_TRIAL_SHAPE,
    EEG_VOCAB_SIZE,
    MEG_GRID_SHAPE,
    MEG_N_RVQ,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_POSITIONS_PER_TRIAL,
    MEG_RVQ_OUT_MODALITIES,
    MEG_TRIAL_SHAPE,
    MEG_VOCAB_SIZE,
    NEURAL_GRID_TYPE,
)

_DIM = 32


# ---------------------------------------------------------------------------
# Decoder embeddings (parallel heads)
# ---------------------------------------------------------------------------


class TestMegRVQDecoderEmbedding:
    def _emb(self) -> MegRVQDecoderEmbedding:
        torch.manual_seed(0)
        e = MegRVQDecoderEmbedding(vocab_size=MEG_VOCAB_SIZE, n_sources=MEG_N_SOURCES, n_time=MEG_N_TIME)
        e.init(dim_tokens=_DIM)
        return e.eval()

    def _ids(self, batch: int = 2) -> torch.Tensor:
        g = torch.Generator().manual_seed(1)
        return torch.randint(0, MEG_VOCAB_SIZE, (batch, MEG_POSITIONS_PER_TRIAL), generator=g)

    def test_forward_embed_shapes(self):
        e = self._emb()
        out = e.forward_embed({"tensor": self._ids()})
        assert out["x"].shape == (2, MEG_POSITIONS_PER_TRIAL, _DIM)
        assert out["emb"].shape == (2, MEG_POSITIONS_PER_TRIAL, _DIM)
        assert out["ids"].shape == (2, MEG_POSITIONS_PER_TRIAL)

    def test_forward_logits_maps_to_vocab(self):
        e = self._emb()
        y = torch.randn(7, _DIM)  # 7 selected decoder tokens
        logits = e.forward_logits(y)
        assert logits.shape == (7, MEG_VOCAB_SIZE)

    def test_axial_positions_source_and_time_both_matter(self):
        e = self._emb()
        ids = torch.full((1, MEG_POSITIONS_PER_TRIAL), 3, dtype=torch.long)
        emb = e.forward_embed({"tensor": ids})["emb"][0]  # (P, D), content identical
        same_source_diff_time = not torch.allclose(emb[0], emb[1])          # (s0,t0) vs (s0,t1)
        same_time_diff_source = not torch.allclose(emb[0], emb[MEG_N_TIME])  # (s0,t0) vs (s1,t0)
        assert same_source_diff_time and same_time_diff_source

    def test_lazy_init_required_before_forward(self):
        e = MegRVQDecoderEmbedding()
        with pytest.raises(AssertionError):
            e.forward_embed({"tensor": self._ids()})

    def test_accepts_share_embedding_kwarg(self):
        # The HF FM wrapper passes share_embedding=False to non-img decoder factories.
        MegRVQDecoderEmbedding(vocab_size=MEG_VOCAB_SIZE, share_embedding=False)


class TestEegDecoderEmbedding:
    def _emb(self) -> EegDecoderEmbedding:
        torch.manual_seed(0)
        e = EegDecoderEmbedding(vocab_size=EEG_VOCAB_SIZE, max_length=EEG_TOKENS_PER_TRIAL)
        e.init(dim_tokens=_DIM)
        return e.eval()

    def test_forward_embed_shapes(self):
        e = self._emb()
        ids = torch.randint(0, EEG_VOCAB_SIZE, (2, EEG_TOKENS_PER_TRIAL))
        out = e.forward_embed({"tensor": ids})
        assert out["x"].shape == (2, EEG_TOKENS_PER_TRIAL, _DIM)
        assert out["emb"].shape == (2, EEG_TOKENS_PER_TRIAL, _DIM)
        assert out["ids"].shape == (2, EEG_TOKENS_PER_TRIAL)

    def test_forward_logits_maps_to_vocab(self):
        e = self._emb()
        logits = e.forward_logits(torch.randn(5, _DIM))
        assert logits.shape == (5, EEG_VOCAB_SIZE)

    def test_positions_vary_along_sequence(self):
        e = self._emb()
        ids = torch.full((1, EEG_TOKENS_PER_TRIAL), 4, dtype=torch.long)
        emb = e.forward_embed({"tensor": ids})["emb"][0]
        assert not torch.allclose(emb[0], emb[1])


# ---------------------------------------------------------------------------
# Modality registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def _info(self):
        from fourm.data.modality_info import MODALITY_INFO

        return MODALITY_INFO

    def test_four_meg_rvq_modalities_registered(self):
        info = self._info()
        assert len(MEG_RVQ_OUT_MODALITIES) == MEG_N_RVQ
        for mod in MEG_RVQ_OUT_MODALITIES:
            assert mod in info, mod
            m = info[mod]
            assert m["type"] == NEURAL_GRID_TYPE
            assert m["vocab_size"] == MEG_VOCAB_SIZE
            assert m["max_tokens"] == MEG_POSITIONS_PER_TRIAL
            assert m["path"] == "tok_meg"           # reads the same on-disk folder
            assert m.get("encoder_embedding") is None  # output-only
            assert m.get("decoder_embedding") is not None
            assert m.get("pretokenized") is True

    def test_eeg_out_modality_registered(self):
        m = self._info()[EEG_OUT_MODALITY]
        assert m["type"] == NEURAL_GRID_TYPE
        assert m["vocab_size"] == EEG_VOCAB_SIZE
        assert m["max_tokens"] == EEG_TOKENS_PER_TRIAL
        assert m["path"] == "tok_eeg"
        assert m.get("encoder_embedding") is None
        assert m.get("decoder_embedding") is not None

    def test_input_only_neural_modalities_unchanged(self):
        info = self._info()
        # The original input-only modalities must keep no decoder head.
        assert info["tok_meg"].get("decoder_embedding") is None
        assert info["tok_eeg"].get("decoder_embedding") is None

    def test_transforms_registered_for_output_modalities(self):
        from fourm.data.modality_info import MODALITY_TRANSFORMS

        for mod in (*MEG_RVQ_OUT_MODALITIES, EEG_OUT_MODALITY):
            assert mod in MODALITY_TRANSFORMS, mod


# ---------------------------------------------------------------------------
# Coherent trial split (the 4 MEG heads share one trial)
# ---------------------------------------------------------------------------


class TestCoherentTrialSplit:
    def _arr_per_trial_constant(self, n_trials: int = 5) -> np.ndarray:
        """arr[t] is entirely the constant (t+1) so we can identify the picked trial."""
        arr = np.zeros((n_trials, *MEG_TRIAL_SHAPE), dtype=np.int16)
        for t in range(n_trials):
            arr[t] = t + 1
        return arr

    def test_all_four_meg_layers_from_same_trial(self):
        from neural_trial_transform import NeuralTargetSplitter

        splitter = NeuralTargetSplitter(training=True, seed=0)
        arr = self._arr_per_trial_constant()
        # rename aliases all four rvq keys to the same source array object.
        sample = {mod: arr for mod in MEG_RVQ_OUT_MODALITIES}
        splitter(sample)

        layers = [sample[mod] for mod in MEG_RVQ_OUT_MODALITIES]
        for layer in layers:
            assert layer.shape == (MEG_POSITIONS_PER_TRIAL,)
        # Every layer is a single constant (one trial) and they all agree.
        constants = {int(np.unique(layer)[0]) for layer in layers}
        assert len(constants) == 1, f"layers came from different trials: {constants}"

    def test_eval_mode_picks_first_trial(self):
        from neural_trial_transform import NeuralTargetSplitter

        splitter = NeuralTargetSplitter(training=False)
        arr = self._arr_per_trial_constant()
        sample = {mod: arr for mod in MEG_RVQ_OUT_MODALITIES}
        splitter(sample)
        assert int(np.unique(sample[MEG_RVQ_OUT_MODALITIES[0]])[0]) == 1  # trial 0 -> constant 1

    def test_eeg_out_split_single_trial(self):
        from neural_trial_transform import NeuralTargetSplitter

        splitter = NeuralTargetSplitter(training=False)
        arr = np.stack([np.full(EEG_TRIAL_SHAPE, t + 1, dtype=np.int16) for t in range(3)])
        sample = {EEG_OUT_MODALITY: arr}
        splitter(sample)
        assert sample[EEG_OUT_MODALITY].shape == (EEG_TOKENS_PER_TRIAL,)
        assert int(np.unique(sample[EEG_OUT_MODALITY])[0]) == 1

    def test_sentinel_meg_becomes_placeholder(self):
        from neural_trial_transform import NeuralTargetSplitter

        splitter = NeuralTargetSplitter(training=True, seed=0)
        sentinel = np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)
        sample = {mod: sentinel for mod in MEG_RVQ_OUT_MODALITIES}
        splitter(sample)
        for mod in MEG_RVQ_OUT_MODALITIES:
            assert sample[mod].shape == (MEG_POSITIONS_PER_TRIAL,)
            assert (sample[mod] >= 0).all()  # sentinel -1 clipped to placeholder 0

    def test_split_is_noop_without_neural_keys(self):
        from neural_trial_transform import NeuralTargetSplitter

        splitter = NeuralTargetSplitter(training=True, seed=0)
        sample = {"tok_rgb": np.zeros((196,), dtype=np.int64)}
        splitter(sample)
        assert set(sample) == {"tok_rgb"}


# ---------------------------------------------------------------------------
# Masking: neural_grid routes through the parallel image_mask path
# ---------------------------------------------------------------------------


class TestNeuralGridMasking:
    @pytest.fixture
    def masking(self):
        from tokenizers import Tokenizer

        from repo_paths import REPO_ROOT
        from neural_masking import PresenceAwareUnifiedMasking

        tok = Tokenizer.from_file(
            str(
                REPO_ROOT
                / "external/ml-4m/fourm/utils/tokenizer/trained/"
                "text_tokenizer_4m_wordpiece_30k.json"
            )
        )
        modality_info = {
            "tok_rgb": {
                "type": "seq_token", "min_tokens": 0, "max_tokens": 196,
                "input_alphas": [1.0], "target_alphas": [1.0], "vocab_offset": 0,
            },
            "tok_meg_rvq0": {
                "type": NEURAL_GRID_TYPE, "min_tokens": 0,
                "max_tokens": MEG_POSITIONS_PER_TRIAL,
                "input_alphas": [0.0], "target_alphas": [1.0],
            },
        }
        return PresenceAwareUnifiedMasking(
            modality_info=modality_info, text_tokenizer=tok,
            input_tokens_range=(64, 64), target_tokens_range=(64, 64),
        )

    def test_present_output_has_targets_and_no_sentinel_ids(self, masking):
        out = masking({
            "tok_rgb": torch.arange(196, dtype=torch.long),
            "tok_meg_rvq0": torch.randint(0, MEG_VOCAB_SIZE, (MEG_POSITIONS_PER_TRIAL,)),
            "meg_mask": torch.tensor([1]),
        })
        meg = out["tok_meg_rvq0"]
        assert (~meg["target_mask"]).sum() > 0                 # produces decoder targets
        assert int(meg["tensor"].max()) < MEG_VOCAB_SIZE       # no text-sentinel injection
        assert int(meg["tensor"].min()) >= 0
        assert meg["tensor"].shape[0] == MEG_POSITIONS_PER_TRIAL  # parallel grid, not span-padded

    def test_absent_output_zeroed_via_presence_flag(self, masking):
        out = masking({
            "tok_rgb": torch.arange(196, dtype=torch.long),
            "tok_meg_rvq0": torch.zeros(MEG_POSITIONS_PER_TRIAL, dtype=torch.long),
            "meg_mask": torch.tensor([0]),
        })
        assert (~out["tok_meg_rvq0"]["target_mask"]).sum() == 0


# ---------------------------------------------------------------------------
# Config validation: neural outputs are now allowed as targets
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def _base_ds(self) -> dict:
        return {
            "type": "multimodal",
            "use_wds": True,
            "data_path": (
                "/project/data/train/things/"
                "[tok_rgb,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{000..000}.tar"
            ),
            "in_domains": "tok_rgb",
            "out_domains": "-".join(["tok_rgb", *MEG_RVQ_OUT_MODALITIES, EEG_OUT_MODALITY]),
            "main_augment_domain": "tok_rgb",
            "input_alphas": "1.0",
            "target_alphas": "1.0",
        }

    def test_neural_outputs_allowed_in_out_domains(self):
        from repo_paths import REPO_ROOT
        from config_validate import validate_dataset_config
        from fourm.data.modality_info import MODALITY_INFO

        errors = validate_dataset_config("things", self._base_ds(), REPO_ROOT, MODALITY_INFO)
        assert errors == [], errors

    def test_tok_meg_still_rejected_as_target(self):
        from repo_paths import REPO_ROOT
        from config_validate import validate_dataset_config
        from fourm.data.modality_info import MODALITY_INFO

        ds = self._base_ds()
        ds["out_domains"] = "tok_rgb-tok_meg"
        errors = validate_dataset_config("things", ds, REPO_ROOT, MODALITY_INFO)
        assert any("tok_meg" in e and "input-only" in e for e in errors)

    def test_rvq_output_requires_meg_folder_in_bracket(self):
        from repo_paths import REPO_ROOT
        from config_validate import validate_dataset_config
        from fourm.data.modality_info import MODALITY_INFO

        ds = self._base_ds()
        ds["data_path"] = "/project/data/train/things/[tok_rgb]/shard_{000..000}.tar"
        errors = validate_dataset_config("things", ds, REPO_ROOT, MODALITY_INFO)
        assert any("tok_meg_rvq0" in e for e in errors)


# ---------------------------------------------------------------------------
# Integration: neural heads produce a finite loss and receive gradient
# ---------------------------------------------------------------------------


class TestNeuralOutputGradientFlow:
    def _batch_and_model(self):
        from tokenizers import Tokenizer

        from fourm.data.modality_info import MODALITY_TRANSFORMS
        from fourm.data.modality_transforms import IdentityTransform, UnifiedDataTransform
        from fourm.data.pretrain_utils import setup_sampling_mod_info
        from fourm.data.unified_datasets import default_collate
        from fourm.utils import create_model

        from neural_masking import PresenceAwareUnifiedMasking
        from neural_trial_transform import NeuralTargetSplitter
        from repo_paths import TEXT_TOKENIZER
        from things_augmenter import ThingsImageAugmenter
        from train_4m import _build_modality_info
        from neural_constants import THINGS_IMAGE_SIZE, TOK_RGB_TOKENS_PER_IMAGE, TOK_RGB_VOCAB_SIZE

        in_domains = ["tok_rgb"]
        out_domains = ["tok_rgb", *MEG_RVQ_OUT_MODALITIES, EEG_OUT_MODALITY]
        all_d = sorted(set(in_domains) | set(out_domains))
        full = _build_modality_info(all_d, input_size=THINGS_IMAGE_SIZE)

        ds_cfg = {
            "in_domains": "-".join(sorted(in_domains)),
            "out_domains": "-".join(sorted(out_domains)),
            "input_alphas": "-".join("1.0" for _ in in_domains),
            "target_alphas": "-".join("1.0" for _ in out_domains),
        }
        mask_info, _ = setup_sampling_mod_info(ds_cfg, full)

        text_tokenizer = Tokenizer.from_file(str(TEXT_TOKENIZER))
        transforms = dict(MODALITY_TRANSFORMS)
        transforms["__key__"] = IdentityTransform()
        udt = UnifiedDataTransform(
            transforms_dict=transforms,
            image_augmenter=ThingsImageAugmenter(
                target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
            ),
        )
        splitter = NeuralTargetSplitter(training=True, seed=0)

        def decoded(rng):
            arr_meg = rng.integers(0, MEG_VOCAB_SIZE, (3, *MEG_TRIAL_SHAPE)).astype(np.int16)
            arr_eeg = rng.integers(0, EEG_VOCAB_SIZE, (3, *EEG_TRIAL_SHAPE)).astype(np.int16)
            sample = {
                "tok_rgb": rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,)).astype(np.int64),
                "meg_mask": np.array([1], dtype=np.int64),
                "eeg_mask": np.array([1], dtype=np.int64),
                "__key__": "x",
            }
            # rename fan-out: all 4 rvq keys alias the MEG source; eeg_out aliases EEG source.
            for mod in MEG_RVQ_OUT_MODALITIES:
                sample[mod] = arr_meg
            sample[EEG_OUT_MODALITY] = arr_eeg
            splitter(sample)
            return sample

        # Build a batch where every neural head has at least one target.
        for attempt in range(50):
            torch.manual_seed(attempt)
            masker = PresenceAwareUnifiedMasking(
                modality_info=mask_info, text_tokenizer=text_tokenizer,
                input_tokens_range=(96, 96), target_tokens_range=(96, 96),
            )
            rng = np.random.default_rng(attempt)
            samples = [masker(udt(decoded(rng))) for _ in range(2)]
            batch = default_collate(samples)
            neural = [*MEG_RVQ_OUT_MODALITIES, EEG_OUT_MODALITY]
            if all(int((~batch[m]["target_mask"]).sum()) > 0 for m in neural):
                break
        else:
            pytest.skip("could not assemble a batch with all neural targets present")

        enc = {m: full[m]["encoder_embedding"](patch_size=full[m].get("patch_size", 16),
                                               image_size=THINGS_IMAGE_SIZE)
               if full[m]["type"] == "img" else full[m]["encoder_embedding"]()
               for m in in_domains}
        dec = {m: full[m]["decoder_embedding"]() for m in out_domains}
        model = create_model(
            "fm_tiny_6e_6d_swiglu_nobias", encoder_embeddings=enc, decoder_embeddings=dec,
            modality_info=full, num_register_tokens=0,
        )
        return model, batch, [*MEG_RVQ_OUT_MODALITIES, EEG_OUT_MODALITY]

    def test_loss_finite_and_gradient_reaches_meg_head(self):
        from fourm_dataloader import patch_pretrain_utils

        patch_pretrain_utils()
        model, batch, neural = self._batch_and_model()
        model.train()
        # Generous caps so top-k selection never truncates a head's targets.
        n_targets = sum(int((~batch[m]["target_mask"]).sum()) for m in batch)
        loss, mod_loss = model(
            batch, num_encoder_tokens=256, num_decoder_tokens=n_targets + 256,
            loss_type="mod",
        )
        assert torch.isfinite(loss)
        for m in neural:
            assert m in mod_loss and torch.isfinite(mod_loss[m]).all()
            assert float(mod_loss[m].detach()) > 0
        loss.backward()
        head = model.decoder_embeddings[MEG_RVQ_OUT_MODALITIES[0]].to_logits.weight
        assert head.grad is not None and head.grad.abs().sum() > 0

    def test_neural_heads_overfit_one_batch(self):
        """The strongest wiring check: each neural head drives its loss down on one batch."""
        from fourm_dataloader import patch_pretrain_utils

        patch_pretrain_utils()
        model, batch, neural = self._batch_and_model()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95))
        n_targets = sum(int((~batch[m]["target_mask"]).sum()) for m in batch)

        def step():
            opt.zero_grad()
            loss, mod_loss = model(
                batch, num_encoder_tokens=256, num_decoder_tokens=n_targets + 256,
                loss_type="mod",
            )
            loss.backward()
            opt.step()
            return {m: float(mod_loss[m].detach()) for m in neural}

        first = step()
        for _ in range(60):
            last = step()
        for m in neural:
            assert last[m] < first[m] - 0.5, f"{m}: {first[m]:.3f} -> {last[m]:.3f} (not learning)"
