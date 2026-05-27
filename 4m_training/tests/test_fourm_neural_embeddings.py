"""Contract tests for MegEncoderEmbedding (RVQ-sum + axial positions).

These lock the structural promises from notes/4m_neural_modality_design.md:
the 4 RVQ layers are summed via per-layer codebooks, and the 16x8 grid gets
axial positions (learned source, sincos time).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourm_neural_embeddings import MegEncoderEmbedding
from neural_constants import (
    MEG_N_RVQ,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_POSITIONS_PER_TRIAL,
    MEG_VOCAB_SIZE,
)

_DIM = 32


def _emb() -> MegEncoderEmbedding:
    torch.manual_seed(0)
    e = MegEncoderEmbedding(vocab_size=MEG_VOCAB_SIZE, n_rvq=MEG_N_RVQ,
                            n_sources=MEG_N_SOURCES, n_time=MEG_N_TIME)
    e.init(dim_tokens=_DIM)
    return e.eval()


def _ids(batch: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, MEG_VOCAB_SIZE, (batch, MEG_POSITIONS_PER_TRIAL, MEG_N_RVQ), generator=g)


class TestShapeAndLifecycle:
    def test_lazy_init_then_forward(self):
        e = MegEncoderEmbedding()  # no dim yet
        with pytest.raises(AssertionError):
            e.forward({"tensor": _ids()})
        e.init(_DIM)
        out = e.forward({"tensor": _ids()})
        assert out["x"].shape == (2, MEG_POSITIONS_PER_TRIAL, _DIM)
        assert out["emb"].shape == (2, MEG_POSITIONS_PER_TRIAL, _DIM)


class TestRvqSummation:
    def test_content_is_sum_of_per_layer_codebooks(self):
        e = _emb()
        ids = _ids()
        x = e.forward({"tensor": ids})["x"]
        expected = sum(e.codebooks[q](ids[..., q]) for q in range(MEG_N_RVQ))
        assert torch.allclose(x, expected, atol=1e-6)

    def test_layers_use_distinct_codebooks(self):
        """Same code value in different RVQ layers must embed differently."""
        e = _emb()
        same = torch.zeros(1, MEG_POSITIONS_PER_TRIAL, MEG_N_RVQ, dtype=torch.long)
        same[..., 0] = 5  # code 5 only in layer 0
        only0 = e.forward({"tensor": same})["x"]
        same2 = torch.zeros_like(same)
        same2[..., 1] = 5  # code 5 only in layer 1
        only1 = e.forward({"tensor": same2})["x"]
        assert not torch.allclose(only0, only1), "layers 0 and 1 share a codebook"

    def test_every_layer_contributes(self):
        e = _emb()
        base = torch.zeros(1, MEG_POSITIONS_PER_TRIAL, MEG_N_RVQ, dtype=torch.long)
        x0 = e.forward({"tensor": base})["x"].clone()
        for q in range(MEG_N_RVQ):
            bumped = base.clone()
            bumped[..., q] = 7
            xq = e.forward({"tensor": bumped})["x"]
            assert not torch.allclose(x0, xq), f"layer {q} did not affect the embedding"


class TestAxialPositions:
    def test_same_codes_different_cell_differ_in_emb(self):
        e = _emb()
        ids = torch.full((1, MEG_POSITIONS_PER_TRIAL, MEG_N_RVQ), 3, dtype=torch.long)
        emb = e.forward({"tensor": ids})["emb"][0]  # (P, D)
        # identical content everywhere -> any position difference is purely positional
        assert not torch.allclose(emb[0], emb[1]), "adjacent cells share a position embedding"

    def test_source_and_time_axes_both_matter(self):
        e = _emb()
        ids = torch.full((1, MEG_POSITIONS_PER_TRIAL, MEG_N_RVQ), 3, dtype=torch.long)
        emb = e.forward({"tensor": ids})["emb"][0]
        # cell p -> source=p//n_time, time=p%n_time
        same_source_diff_time = not torch.allclose(emb[0], emb[1])          # (s0,t0) vs (s0,t1)
        same_time_diff_source = not torch.allclose(emb[0], emb[MEG_N_TIME])  # (s0,t0) vs (s1,t0)
        assert same_source_diff_time and same_time_diff_source
