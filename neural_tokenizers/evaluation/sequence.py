"""§5.4 Token-sequence statistics.

A tokenizer can pass reconstruction, codebook utilization, AND probe and still
break 4M training — if its token sequences are too redundant for masked-token
prediction to learn anything (e.g. emits `[42, 42, ..., 42]` per trial, varying
only by stimulus). This module is the gate that catches that failure mode.

Three numbers:
  - unigram_entropy             H(t). The "vocabulary" entropy.
  - bigram_conditional_entropy  H(t_i | t_{i-1}). The key number.
  - entropy_gap_pct             100 * (unigram - bigram) / unigram. Healthy ~0.
  - mean_run_length             how long are runs of the same code?
  - frac_runs_ge_2              fraction of trials with any run of length >= 2.

A healthy tokenizer has bigram ~= unigram (transitions are nearly independent
of the previous token) and short runs. The orchestrator should print a warning
when entropy_gap_pct exceeds ~20%.
"""

from __future__ import annotations

import math

import torch

from .protocol import MetricResult


def compute_sequence_metrics(
    tokens: torch.Tensor,
    codebook_size: int,
) -> MetricResult:
    """Compute sequence-level entropy and run-length stats.

    Args:
        tokens: long tensor of shape (B, ...) — flattened to (B, L) where the
            second dim is the natural reading order. For 4M image tokens
            (B, H, W), that flattens row-major, which is what masked-token
            training also does.
        codebook_size: V.
    """
    if tokens.ndim < 2:
        raise ValueError(
            f"tokens must have at least (B, L); got shape {tuple(tokens.shape)}"
        )
    flat = tokens.reshape(tokens.shape[0], -1).to(torch.long).cpu()

    unigram_H = _entropy(_marginal_probs(flat.reshape(-1), codebook_size))
    bigram_H = _bigram_conditional_entropy(flat, codebook_size)
    gap = _entropy_gap_pct(unigram_H, bigram_H)
    mean_rl, frac_ge2 = _run_length_stats(flat)

    return MetricResult(
        name="sequence",
        values={
            "unigram_entropy": float(unigram_H),
            "bigram_conditional_entropy": float(bigram_H),
            "entropy_gap_pct": float(gap),
            "mean_run_length": float(mean_rl),
            "frac_runs_ge_2": float(frac_ge2),
            "max_possible_entropy": float(math.log(codebook_size)),
        },
    )


def _marginal_probs(flat: torch.Tensor, codebook_size: int) -> torch.Tensor:
    counts = torch.bincount(flat, minlength=codebook_size).to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _entropy(probs: torch.Tensor) -> float:
    nonzero = probs > 0
    return float(-(probs[nonzero] * probs[nonzero].log()).sum().item())


def _bigram_conditional_entropy(tokens: torch.Tensor, codebook_size: int) -> float:
    """H(t_i | t_{i-1}) — entropy of next token given the previous one.

    For each trial we read pairs (t_0, t_1), (t_1, t_2), ..., (t_{L-2}, t_{L-1})
    then pool across trials. If L < 2 we return the unigram entropy (no bigrams
    available to condition on).
    """
    B, L = tokens.shape
    if L < 2:
        flat = tokens.reshape(-1)
        return _entropy(_marginal_probs(flat, codebook_size))

    prev = tokens[:, :-1].reshape(-1)
    curr = tokens[:, 1:].reshape(-1)
    pair_id = prev * codebook_size + curr
    pair_counts = torch.bincount(
        pair_id, minlength=codebook_size * codebook_size
    ).reshape(codebook_size, codebook_size).to(torch.float64)
    prev_counts = pair_counts.sum(dim=1).clamp_min(1.0)
    cond_probs = pair_counts / prev_counts.unsqueeze(1)
    prev_probs = prev_counts / prev_counts.sum().clamp_min(1.0)

    row_H = torch.where(
        cond_probs > 0, -cond_probs * cond_probs.log(), torch.zeros_like(cond_probs)
    ).sum(dim=1)
    return float((prev_probs * row_H).sum().item())


def _entropy_gap_pct(unigram_H: float, bigram_H: float) -> float:
    if unigram_H <= 0:
        return 0.0
    return 100.0 * (unigram_H - bigram_H) / unigram_H


def _run_length_stats(tokens: torch.Tensor) -> tuple[float, float]:
    """Mean run length across all trials, and fraction of trials containing
    any run of length >= 2.

    A run is a maximal stretch of identical consecutive tokens.
    """
    B, L = tokens.shape
    if L == 1:
        return 1.0, 0.0

    same = (tokens[:, 1:] == tokens[:, :-1]).to(torch.int64)
    n_breaks = (1 - same).sum(dim=1)
    runs_per_trial = n_breaks + 1
    mean_rl = float((L / runs_per_trial.to(torch.float64)).mean().item())
    has_run_ge_2 = (same.sum(dim=1) > 0).to(torch.float64)
    frac_ge2 = float(has_run_ge_2.mean().item())
    return mean_rl, frac_ge2
