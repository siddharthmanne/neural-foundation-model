"""Encoder embeddings for neural modalities that stay loyal to token structure.

MEG (BrainOmni) tokens are a ``16 x 8`` spatiotemporal grid with 4 RVQ codes per
cell. ``MegEncoderEmbedding`` keeps that structure: it sums the 4 residual codebook
embeddings per cell (the RVQ-faithful collapse) and adds **axial** positions — a
learned embedding for the 16 unordered latent sources, a sincos embedding for the 8
ordered time steps. EEG is already 1D and uses stock ``SequenceEncoderEmbedding``.

``MegEncoderEmbedding`` is the encoder (input) side. For the **output** side — predicting
MEG/EEG as a reconstruction regularizer — ``MegRVQDecoderEmbedding`` (one parallel head
per RVQ layer) and ``EegDecoderEmbedding`` mirror the same positional logic and add a
``forward_logits`` projection. See ``notes/4m_neural_modality_design.md`` §6.

The encoder module mirrors ``ImageTokenEncoderEmbedding``'s ``init`` / ``forward``
contract; the decoder modules mirror ``ImageTokenDecoderEmbedding``'s
``forward_embed`` / ``forward_logits`` (fixed per-cell positions; parallel decoding).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from einops import repeat
from fourm.models.fm_utils import build_1d_sincos_posemb

from neural_constants import (
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_N_RVQ,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_VOCAB_SIZE,
)


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


class _NeuralGridDecoderEmbedding(nn.Module):
    """Parallel decoder head for a fixed-size neural token grid (output modalities).

    Mirrors ``ImageTokenDecoderEmbedding``'s ``forward_embed`` / ``forward_logits``
    contract so 4M routes these through its **parallel** decoder branch: target ids are
    embedded (``d['x']``), per-cell positional + modality embeddings are added
    (``d['emb']``), and ``forward_logits`` projects decoder outputs to vocab logits.
    The decoder replaces ``d['x']`` with its mask token, so the model must *predict*
    each target — there is no leakage. Subclasses define the positional layout.

    Subclasses set ``self.vocab_size`` / ``self.n_positions`` / ``self.dim_tokens`` in
    ``__init__`` then call ``self.init(dim_tokens)`` (4M calls it lazily otherwise).
    """

    def init(self, dim_tokens: int = 768, init_std: float = 0.02) -> None:
        self.dim_tokens = dim_tokens
        # No padding_idx: code 0 is a real token; absence is handled by presence masking.
        self.token_emb = nn.Embedding(self.vocab_size, dim_tokens)
        self.to_logits = nn.Linear(dim_tokens, self.vocab_size, bias=False)
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)
        self._build_positions(dim_tokens, init_std)

    def _build_positions(self, dim_tokens: int, init_std: float) -> None:
        raise NotImplementedError

    def _position_embedding(self) -> torch.Tensor:
        """Fixed per-cell positional embedding of shape ``(n_positions, D)``."""
        raise NotImplementedError

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.dim_tokens is not None, "Call init(dim_tokens) before forward_embed."
        ids = d["tensor"]
        B = ids.shape[0]
        ids = ids.reshape(B, -1)
        x = self.token_emb(ids)
        x_emb = repeat(
            self._position_embedding().unsqueeze(0) + self.mod_emb, "() n d -> b n d", b=B
        )
        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = ids
        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_logits(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()


class MegRVQDecoderEmbedding(_NeuralGridDecoderEmbedding):
    """One MEG RVQ-layer head: ``(B, 128)`` ids of a 16x8 grid, vocab 512.

    Positions mirror ``MegEncoderEmbedding``: a **learned** embedding for the 16
    unordered latent sources and **sincos** for the 8 ordered time steps, summed
    per cell. Four of these (one per RVQ layer) reconstruct a MEG token in parallel.
    """

    def __init__(
        self,
        vocab_size: int = MEG_VOCAB_SIZE,
        n_sources: int = MEG_N_SOURCES,
        n_time: int = MEG_N_TIME,
        dim_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_sources = n_sources
        self.n_time = n_time
        self.n_positions = n_sources * n_time
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)

    def _build_positions(self, dim_tokens: int, init_std: float) -> None:
        self.source_emb = nn.Parameter(torch.zeros(self.n_sources, dim_tokens))
        nn.init.normal_(self.source_emb, std=init_std)
        time_emb = build_1d_sincos_posemb(max_len=self.n_time, embed_dim=dim_tokens)[0]
        self.register_buffer("time_emb", time_emb)  # (n_time, D)
        cells = torch.arange(self.n_positions)
        self.register_buffer("src_idx", cells // self.n_time)
        self.register_buffer("time_idx", cells % self.n_time)

    def _position_embedding(self) -> torch.Tensor:
        return self.source_emb[self.src_idx] + self.time_emb[self.time_idx]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"source_emb"}


class EegDecoderEmbedding(_NeuralGridDecoderEmbedding):
    """EEG head: ``(B, 17)`` ids of a 1D sequence, vocab 8192, sincos positions."""

    def __init__(
        self,
        vocab_size: int = EEG_VOCAB_SIZE,
        max_length: int = EEG_TOKENS_PER_TRIAL,
        dim_tokens: int | None = None,
        sincos_pos_emb: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = max_length
        self.sincos_pos_emb = sincos_pos_emb
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)

    def _build_positions(self, dim_tokens: int, init_std: float) -> None:
        if self.sincos_pos_emb:
            pos = build_1d_sincos_posemb(max_len=self.n_positions, embed_dim=dim_tokens)[0]
            self.register_buffer("pos_emb", pos)  # (n_positions, D)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(self.n_positions, dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

    def _position_embedding(self) -> torch.Tensor:
        return self.pos_emb
