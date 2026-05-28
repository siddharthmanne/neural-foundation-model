"""Tests for presence-aware Dirichlet budget zeroing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from neural_constants import (
    EEG_OUT_MODALITY,
    MEG_RVQ_OUT_MODALITIES,
    MEG_TOKENS_PER_TRIAL,
    MEG_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
)
from neural_masking import (
    PresenceAwareUnifiedMasking,
    extract_presence_flags,
    zero_absent_budgets,
)


class TestZeroAbsentBudgets:
    def test_zeros_meg_when_absent(self):
        modality_info = {
            "tok_rgb": {"type": "seq_token"},
            "tok_meg": {"type": "seq_token"},
        }
        input_b, target_b = zero_absent_budgets(
            modality_info,
            [10, 20],
            [5, 15],
            {"tok_meg": False, "tok_rgb": True},
        )
        assert input_b == [10, 0]
        assert target_b == [5, 0]

    def test_zeros_eeg_when_absent(self):
        modality_info = {
            "tok_eeg": {"type": "seq_token"},
        }
        input_b, target_b = zero_absent_budgets(
            modality_info,
            [30],
            [12],
            {"tok_eeg": False},
        )
        assert input_b == [0]
        assert target_b == [0]


class TestExtractPresenceFlags:
    def test_pops_flags_from_dict(self):
        mod_dict = {
            "tok_rgb": np.zeros(3),
            "meg_mask": np.array([0], dtype=np.uint8),
            "eeg_mask": torch.tensor([1]),
        }
        presence = extract_presence_flags(mod_dict)
        assert "meg_mask" not in mod_dict
        assert "eeg_mask" not in mod_dict
        # Every token modality gated by meg_mask is absent; those gated by eeg_mask present.
        # The output heads share the same flags as their input-only counterparts.
        assert presence["tok_meg"] is False
        assert presence["tok_eeg"] is True
        assert all(presence[m] is False for m in MEG_RVQ_OUT_MODALITIES)
        assert presence[EEG_OUT_MODALITY] is True

    def test_defaults_present_when_flag_missing(self):
        presence = extract_presence_flags({"tok_rgb": 1})
        assert all(v is True for v in presence.values())
        assert presence["tok_meg"] and presence["tok_eeg"]


class TestPresenceAwareUnifiedMasking:
    @pytest.fixture
    def masking(self):
        from repo_paths import REPO_ROOT

        tok_path = (
            REPO_ROOT
            / "external/ml-4m/fourm/utils/tokenizer/trained/"
            "text_tokenizer_4m_wordpiece_30k.json"
        )
        text_tokenizer = Tokenizer.from_file(str(tok_path))
        modality_info = {
            "tok_rgb": {
                "type": "seq_token",
                "min_tokens": 0,
                "max_tokens": TOK_RGB_TOKENS_PER_IMAGE,
                "input_alphas": [1.0],
                "target_alphas": [1.0],
                "vocab_offset": 0,
            },
            "tok_meg": {
                "type": "seq_token",
                "min_tokens": 0,
                "max_tokens": MEG_TOKENS_PER_TRIAL,
                "input_alphas": [1.0],
                "target_alphas": [1.0],
                "vocab_offset": 0,
            },
        }
        return PresenceAwareUnifiedMasking(
            modality_info=modality_info,
            text_tokenizer=text_tokenizer,
            input_tokens_range=(32, 32),
            target_tokens_range=(32, 32),
        )

    def test_absent_meg_produces_no_target_tokens(self, masking):
        mod_dict = {
            "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE, dtype=torch.long),
            "tok_meg": torch.zeros(MEG_TOKENS_PER_TRIAL, dtype=torch.long),
            "meg_mask": torch.tensor([0]),
        }
        out = masking(mod_dict)
        # When budget is 0, sequence_token_mask leaves target_mask all True (ignored).
        meg_logits_budget = (~out["tok_meg"]["target_mask"]).sum()
        assert meg_logits_budget == 0

    def test_present_meg_has_target_tokens(self, masking):
        mod_dict = {
            "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE, dtype=torch.long),
            "tok_meg": torch.arange(MEG_TOKENS_PER_TRIAL, dtype=torch.long),
            "meg_mask": torch.tensor([1]),
        }
        out = masking(mod_dict)
        meg_logits_budget = (~out["tok_meg"]["target_mask"]).sum()
        assert meg_logits_budget > 0

    def test_masked_meg_tensor_ids_stay_below_vocab(self, masking):
        """Regression: stock sequence_token_mask would inject text sentinel ids."""
        out = masking(
            {
                "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE, dtype=torch.long),
                "tok_meg": torch.randint(0, MEG_VOCAB_SIZE, (MEG_TOKENS_PER_TRIAL,)),
                "meg_mask": torch.tensor([1]),
            }
        )
        assert int(out["tok_meg"]["tensor"].max()) < MEG_VOCAB_SIZE
        assert int(out["tok_meg"]["tensor"].min()) >= 0
