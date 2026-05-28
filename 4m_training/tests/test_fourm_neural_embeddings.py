"""Contract tests for the neural ENCODER embeddings (symmetric modalities).

Each RVQ layer is its own modality with a single codebook; ``MegRVQEncoderEmbedding``
embeds ``(B, 128)`` single codes and adds axial positions (learned source, sincos time).
``EegEncoderEmbedding`` embeds ``(B, 17)`` codes with sincos positions. Both share their
positional layout with the matching decoder embedding via the position mixins, so the two
sides of a modality cannot drift in *scheme* (the learned source weights are per-module).
See notes/4m_neural_modality_design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourm_neural_embeddings import (
    EegDecoderEmbedding,
    EegEncoderEmbedding,
    MegRVQDecoderEmbedding,
    MegRVQEncoderEmbedding,
    _AxialPositions,
    _SincosPositions,
)
from neural_constants import (
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_N_TIME,
    MEG_POSITIONS_PER_TRIAL,
    MEG_VOCAB_SIZE,
)

_DIM = 32


class TestMegRVQEncoderEmbedding:
    def _emb(self) -> MegRVQEncoderEmbedding:
        torch.manual_seed(0)
        e = MegRVQEncoderEmbedding(vocab_size=MEG_VOCAB_SIZE)
        e.init(dim_tokens=_DIM)
        return e.eval()

    def _ids(self, batch: int = 2) -> torch.Tensor:
        g = torch.Generator().manual_seed(1)
        return torch.randint(0, MEG_VOCAB_SIZE, (batch, MEG_POSITIONS_PER_TRIAL), generator=g)

    def test_lazy_init_then_forward(self):
        e = MegRVQEncoderEmbedding()  # no dim yet
        with pytest.raises(AssertionError):
            e.forward({"tensor": self._ids()})
        e.init(_DIM)
        out = e.forward({"tensor": self._ids()})
        assert out["x"].shape == (2, MEG_POSITIONS_PER_TRIAL, _DIM)
        assert out["emb"].shape == (2, MEG_POSITIONS_PER_TRIAL, _DIM)

    def test_content_is_single_codebook_lookup(self):
        e = self._emb()
        ids = self._ids()
        x = e.forward({"tensor": ids})["x"]
        assert torch.allclose(x, e.token_emb(ids), atol=1e-6)

    def test_axial_positions_source_and_time_both_matter(self):
        e = self._emb()
        ids = torch.full((1, MEG_POSITIONS_PER_TRIAL), 3, dtype=torch.long)
        emb = e.forward({"tensor": ids})["emb"][0]  # (P, D), content identical
        same_source_diff_time = not torch.allclose(emb[0], emb[1])          # (s0,t0) vs (s0,t1)
        same_time_diff_source = not torch.allclose(emb[0], emb[MEG_N_TIME])  # (s0,t0) vs (s1,t0)
        assert same_source_diff_time and same_time_diff_source

    def test_source_emb_excluded_from_weight_decay(self):
        assert "source_emb" in self._emb().no_weight_decay()

    def test_accepts_share_embedding_kwarg(self):
        # The HF FM wrapper passes share_embedding=False to non-img embedding factories.
        MegRVQEncoderEmbedding(vocab_size=MEG_VOCAB_SIZE, share_embedding=False)


class TestEegEncoderEmbedding:
    def _emb(self) -> EegEncoderEmbedding:
        torch.manual_seed(0)
        e = EegEncoderEmbedding(vocab_size=EEG_VOCAB_SIZE, max_length=EEG_TOKENS_PER_TRIAL)
        e.init(dim_tokens=_DIM)
        return e.eval()

    def test_forward_shapes(self):
        e = self._emb()
        ids = torch.randint(0, EEG_VOCAB_SIZE, (2, EEG_TOKENS_PER_TRIAL))
        out = e.forward({"tensor": ids})
        assert out["x"].shape == (2, EEG_TOKENS_PER_TRIAL, _DIM)
        assert out["emb"].shape == (2, EEG_TOKENS_PER_TRIAL, _DIM)

    def test_positions_vary_along_sequence(self):
        e = self._emb()
        ids = torch.full((1, EEG_TOKENS_PER_TRIAL), 4, dtype=torch.long)
        emb = e.forward({"tensor": ids})["emb"][0]
        assert not torch.allclose(emb[0], emb[1])


class TestEncoderDecoderShareScheme:
    """Encoder and decoder of a modality use the SAME positional scheme (shared mixin)."""

    def test_meg_uses_axial_on_both_sides(self):
        assert issubclass(MegRVQEncoderEmbedding, _AxialPositions)
        assert issubclass(MegRVQDecoderEmbedding, _AxialPositions)

    def test_eeg_uses_sincos_on_both_sides(self):
        assert issubclass(EegEncoderEmbedding, _SincosPositions)
        assert issubclass(EegDecoderEmbedding, _SincosPositions)

    def test_eeg_positions_deterministic_match(self):
        """EEG positions are fixed sincos (no learned part), so the two sides are identical."""
        torch.manual_seed(0)
        enc = EegEncoderEmbedding(); enc.init(_DIM)
        dec = EegDecoderEmbedding(); dec.init(_DIM)
        assert torch.allclose(enc._position_embedding(), dec._position_embedding())
