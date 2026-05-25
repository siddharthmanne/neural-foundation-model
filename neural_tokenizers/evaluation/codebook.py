"""§5.2 Codebook utilization.

How many of the codebook's V entries does the tokenizer actually use?

  - perplexity = exp(H(p))   p = empirical token distribution.
        perplexity ranges in [1, V]. Equal to V iff codes are used uniformly.
        More diagnostic than utilization% because a handful of dominant codes
        don't trick it.
  - dead_code_fraction       fraction of codes that never appear on the data.
  - utilization              codes_used / V (cheap, less informative).

ml-4m provides one related utility — fourm/vq/vq_utils.py::compute_codebook_usage
— which counts unique tokens in fixed-size windows. We don't call it because
the per-window unique-count is less diagnostic than perplexity, but a hot
tokenizer should agree with both.
"""

from __future__ import annotations

import math

import torch

from .protocol import EvalConfig, MetricResult, Tokenizer


@torch.no_grad()
def compute_codebook_metrics(
    tokenizer: Tokenizer,
    signal: torch.Tensor,
    config: EvalConfig,
) -> MetricResult:
    """Tokenize the full signal, then compute codebook stats from the tokens."""
    tokens = tokenize_all(tokenizer, signal, config)
    return compute_codebook_metrics_from_tokens(tokens, tokenizer.codebook_size)


def compute_codebook_metrics_from_tokens(
    tokens: torch.Tensor,
    codebook_size: int,
) -> MetricResult:
    """Cheaper entry point when tokens have already been computed elsewhere.

    Args:
        tokens: integer tensor of any shape; will be flattened.
        codebook_size: V, the maximum number of distinct token IDs.
    """
    flat = tokens.reshape(-1).to(torch.long).cpu()
    counts = torch.bincount(flat, minlength=codebook_size).to(torch.float64)
    total = counts.sum().clamp_min(1.0)
    probs = counts / total

    nonzero = probs > 0
    entropy_nats = float(-(probs[nonzero] * probs[nonzero].log()).sum().item())
    perplexity = math.exp(entropy_nats)

    codes_used = int(nonzero.sum().item())
    return MetricResult(
        name="codebook",
        values={
            "perplexity": float(perplexity),
            "codes_used": float(codes_used),
            "codebook_size": float(codebook_size),
            "utilization": codes_used / codebook_size,
            "dead_code_fraction": 1.0 - codes_used / codebook_size,
            "entropy_nats": entropy_nats,
        },
    )


@torch.no_grad()
def tokenize_all(
    tokenizer: Tokenizer,
    signal: torch.Tensor,
    config: EvalConfig,
) -> torch.Tensor:
    """Tokenize the full signal in batches; return concatenated tokens on CPU.

    Used by the orchestrator to compute tokens once and feed both the codebook
    and the sequence metric. Output shape is (B_total, ...) — the trailing
    dims are whatever the tokenizer chose.
    """
    chunks: list[torch.Tensor] = []
    for start in range(0, signal.shape[0], config.batch_size):
        x = signal[start : start + config.batch_size].to(config.device)
        chunks.append(tokenizer.tokenize(x).cpu())
    return torch.cat(chunks, dim=0)
