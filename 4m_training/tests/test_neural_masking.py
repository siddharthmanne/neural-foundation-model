"""Tests for presence-aware Dirichlet budget zeroing + the leak-free neural split.

Neural modalities are ``neural_grid`` and SYMMETRIC (in both in/out domains). The key
correctness guarantee is that ``image_mask`` partitions a modality's cells into input vs
target *disjointly*, so a cell is never both an encoder input and a decoder target — no
input->target leakage. See notes/4m_neural_modality_design.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from neural_constants import (
    EEG_MODALITY,
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_POSITIONS_PER_TRIAL,
    MEG_RVQ_MODALITIES,
    MEG_VOCAB_SIZE,
    NEURAL_GRID_TYPE,
    TOK_RGB_TOKENS_PER_IMAGE,
)
from neural_masking import (
    PRESENCE_FLAGS,
    PresenceAwareUnifiedMasking,
    extract_presence_flags,
    zero_absent_budgets,
)

_MEG0 = MEG_RVQ_MODALITIES[0]


class TestZeroAbsentBudgets:
    def test_zeros_meg_when_absent(self):
        modality_info = {
            "tok_rgb": {"type": "seq_token"},
            _MEG0: {"type": NEURAL_GRID_TYPE},
        }
        input_b, target_b = zero_absent_budgets(
            modality_info,
            [10, 20],
            [5, 15],
            {_MEG0: False, "tok_rgb": True},
        )
        assert input_b == [10, 0]
        assert target_b == [5, 0]

    def test_zeros_eeg_when_absent(self):
        modality_info = {EEG_MODALITY: {"type": NEURAL_GRID_TYPE}}
        input_b, target_b = zero_absent_budgets(
            modality_info, [30], [12], {EEG_MODALITY: False}
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
        # All four MEG RVQ modalities gate on meg_mask; EEG gates on eeg_mask.
        assert all(presence[m] is False for m in MEG_RVQ_MODALITIES)
        assert presence[EEG_MODALITY] is True

    def test_defaults_present_when_flag_missing(self):
        presence = extract_presence_flags({"tok_rgb": 1})
        assert all(v is True for v in presence.values())

    def test_presence_map_has_no_folder_only_names(self):
        # "tok_meg" is a folder, not a modality, so it must not be a presence key.
        assert "tok_meg" not in PRESENCE_FLAGS
        assert set(PRESENCE_FLAGS) == {*MEG_RVQ_MODALITIES, EEG_MODALITY}


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
        # tok_rgb (seq_token) + one symmetric neural modality (neural_grid, in AND out).
        modality_info = {
            "tok_rgb": {
                "type": "seq_token", "min_tokens": 0, "max_tokens": TOK_RGB_TOKENS_PER_IMAGE,
                "input_alphas": [1.0], "target_alphas": [1.0], "vocab_offset": 0,
            },
            _MEG0: {
                "type": NEURAL_GRID_TYPE, "min_tokens": 0,
                "max_tokens": MEG_POSITIONS_PER_TRIAL,
                "input_alphas": [1.0], "target_alphas": [1.0],
            },
        }
        return PresenceAwareUnifiedMasking(
            modality_info=modality_info, text_tokenizer=text_tokenizer,
            input_tokens_range=(64, 64), target_tokens_range=(64, 64),
        )

    def _present_sample(self) -> dict:
        return {
            "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE, dtype=torch.long),
            _MEG0: torch.randint(0, MEG_VOCAB_SIZE, (MEG_POSITIONS_PER_TRIAL,)),
            "meg_mask": torch.tensor([1]),
        }

    def test_present_neural_has_input_and_target(self, masking):
        out = masking(self._present_sample())[_MEG0]
        assert (~out["input_mask"]).sum() > 0   # some cells given to the encoder
        assert (~out["target_mask"]).sum() > 0  # some cells predicted by the decoder

    def test_input_and_target_cells_are_disjoint_no_leak(self, masking):
        """THE core guarantee: a cell is never both an encoder input and a decoder target."""
        for _ in range(40):
            out = masking(self._present_sample())[_MEG0]
            both = (~out["input_mask"]) & (~out["target_mask"])
            assert int(both.sum()) == 0, "a neural cell leaked from input into target"

    def test_tensor_ids_stay_below_vocab(self, masking):
        """Regression: neural_grid masking must not inject text-tokenizer sentinel ids."""
        out = masking(self._present_sample())[_MEG0]
        assert int(out["tensor"].max()) < MEG_VOCAB_SIZE
        assert int(out["tensor"].min()) >= 0
        assert out["tensor"].shape[0] == MEG_POSITIONS_PER_TRIAL  # full parallel grid

    def test_absent_neural_zeroed(self, masking):
        out = masking({
            "tok_rgb": torch.arange(TOK_RGB_TOKENS_PER_IMAGE, dtype=torch.long),
            _MEG0: torch.zeros(MEG_POSITIONS_PER_TRIAL, dtype=torch.long),
            "meg_mask": torch.tensor([0]),
        })[_MEG0]
        assert (~out["target_mask"]).sum() == 0
        assert (~out["input_mask"]).sum() == 0
