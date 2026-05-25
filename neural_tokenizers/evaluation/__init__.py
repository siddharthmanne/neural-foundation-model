"""Tokenizer evaluation harness — four-axis correctness gate for any tokenizer.

Public entry point: `evaluate(tokenizer, signal, labels, config) -> TokenizerReport`.

Each metric module is independently importable so you can run a single axis
during development without paying for the others.
"""

from __future__ import annotations

import torch

from .codebook import (
    compute_codebook_metrics,
    compute_codebook_metrics_from_tokens,
    tokenize_all,
)
from .probe import compute_probe_metrics
from .protocol import (
    DEFAULT_BANDS,
    EvalConfig,
    MetricResult,
    Tokenizer,
    TokenizerReport,
    has_token_embeddings,
)
from .reconstruction import compute_reconstruction_metrics
from .sequence import compute_sequence_metrics


__all__ = [
    "DEFAULT_BANDS",
    "EvalConfig",
    "MetricResult",
    "Tokenizer",
    "TokenizerReport",
    "compute_codebook_metrics",
    "compute_codebook_metrics_from_tokens",
    "compute_probe_metrics",
    "compute_reconstruction_metrics",
    "compute_sequence_metrics",
    "evaluate",
    "has_token_embeddings",
    "tokenize_all",
]


def evaluate(
    tokenizer: Tokenizer,
    signal: torch.Tensor,
    labels: torch.Tensor | None,
    config: EvalConfig,
) -> TokenizerReport:
    """Run the four-axis tokenizer evaluation harness.

    Tokens are computed once and shared between the codebook and sequence
    axes; the probe can also accept the cached tokens to avoid retokenizing.

    Args:
        tokenizer: object satisfying the Tokenizer protocol.
        signal: (B, C, T) float tensor — any shape, harness adapts.
        labels: (B,) long tensor of class indices, or None to skip the probe.
        config: per-modality knobs (sample rate, bands, probe epochs, etc.).

    Note: not decorated with @torch.no_grad() because the probe trains a linear
    head. Each individual metric scopes its own no-grad regions internally.
    """
    report = TokenizerReport()

    if config.run_reconstruction:
        report.reconstruction = compute_reconstruction_metrics(tokenizer, signal, config)

    tokens: torch.Tensor | None = None
    if config.run_codebook or config.run_sequence or (config.run_probe and labels is not None):
        with torch.no_grad():
            tokens = tokenize_all(tokenizer, signal, config)

    if config.run_codebook and tokens is not None:
        report.codebook = compute_codebook_metrics_from_tokens(tokens, tokenizer.codebook_size)

    if config.run_sequence and tokens is not None:
        report.sequence = compute_sequence_metrics(tokens, tokenizer.codebook_size)

    if config.run_probe and labels is not None and tokens is not None:
        report.probe = compute_probe_metrics(tokenizer, signal, labels, config, tokens=tokens)

    return report
