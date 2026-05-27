"""Encoder embeddings for neural modalities that stay loyal to token structure.

MEG (BrainOmni) tokens are a ``16 x 8`` spatiotemporal grid with 4 RVQ codes per
cell. ``MegEncoderEmbedding`` keeps that structure: it sums the 4 residual codebook
embeddings per cell (the RVQ-faithful collapse) and adds **axial** positions — a
learned embedding for the 16 unordered latent sources, a sincos embedding for the 8
ordered time steps. EEG is already 1D and uses stock ``SequenceEncoderEmbedding``.

These are **encoder-only**: MEG/EEG are model inputs, never targets, so no decoder
embedding is needed. See ``notes/4m_neural_modality_design.md``.

The module mirrors ``ImageTokenEncoderEmbedding``'s ``init`` / ``forward`` contract
(fixed per-cell positions; outputs ``d['x']`` content + ``d['emb']`` pos+modality).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from einops import repeat
from fourm.models.fm_utils import build_1d_sincos_posemb

from neural_constants import MEG_N_RVQ, MEG_N_SOURCES, MEG_N_TIME, MEG_VOCAB_SIZE


class MegEncoderEmbedding(nn.Module):
    """Embed a ``(B, n_sources*n_time, n_rvq)`` MEG token grid into ``(B, P, D)``.

    Args:
        vocab_size: Per-RVQ-layer codebook size.
        n_rvq: Number of residual quantizer layers (summed).
        n_sources: Latent source variables (spatial axis; learned positions).
        n_time: Temporal latent steps (time axis; sincos positions).
        dim_tokens: Model dimension; set lazily via ``init`` when 4M is built.
    """

    def __init__(
        self,
        vocab_size: int = MEG_VOCAB_SIZE,
        n_rvq: int = MEG_N_RVQ,
        n_sources: int = MEG_N_SOURCES,
        n_time: int = MEG_N_TIME,
        dim_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_rvq = n_rvq
        self.n_sources = n_sources
        self.n_time = n_time
        self.n_positions = n_sources * n_time
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)

    def init(self, dim_tokens: int = 768, init_std: float = 0.02) -> None:
        """Build dimension-dependent parameters (called once when 4M is set up)."""
        self.dim_tokens = dim_tokens

        # One codebook per RVQ layer — code c in layer q is distinct from layer q'.
        # No padding_idx: code 0 is a real RVQ code; absent MEG is handled upstream
        # by presence masking (zeroed token budget), not by a padding embedding.
        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.vocab_size, dim_tokens) for _ in range(self.n_rvq)]
        )

        # Axial positions: sources are unordered (learned), time is ordered (sincos).
        self.source_emb = nn.Parameter(torch.zeros(self.n_sources, dim_tokens))
        nn.init.normal_(self.source_emb, std=init_std)
        time_emb = build_1d_sincos_posemb(max_len=self.n_time, embed_dim=dim_tokens)[0]
        self.register_buffer("time_emb", time_emb)  # (n_time, D)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Cell p (row-major over the 16x8 grid) -> source = p // n_time, time = p % n_time.
        cells = torch.arange(self.n_positions)
        self.register_buffer("src_idx", cells // self.n_time)
        self.register_buffer("time_idx", cells % self.n_time)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"source_emb"}

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``d['tensor']``: ``(B, P, n_rvq)`` ids -> adds ``d['x']`` and ``d['emb']``."""
        assert self.dim_tokens is not None, "Call init(dim_tokens) before forward."
        ids = d["tensor"]
        B = ids.shape[0]

        # RVQ collapse: sum each layer's codebook lookup (residual reconstruction).
        x = self.codebooks[0](ids[..., 0])
        for q in range(1, self.n_rvq):
            x = x + self.codebooks[q](ids[..., q])

        # Fixed axial position per cell + modality embedding (broadcast over batch).
        pos = self.source_emb[self.src_idx] + self.time_emb[self.time_idx]  # (P, D)
        x_emb = repeat(pos.unsqueeze(0) + self.mod_emb, "() n d -> b n d", b=B)

        d["x"] = x
        d["emb"] = x_emb
        return d
