"""Interface contract for the tokenizer evaluation harness.

Defines the minimal Tokenizer protocol any object must satisfy to be evaluated,
the EvalConfig dataclass that carries modality-specific knobs, and the
MetricResult / TokenizerReport types each metric module returns.

Protocol method names and shapes mirror fourm.vq.vqvae.VQ (see
external/ml-4m/fourm/vq/vqvae.py L39-394) so a 4M VQVAE / DiVAE satisfies this
protocol with zero adapter code. We deliberately exclude the training-time
methods (`encode` 3-tuple, `forward`, `autoencode`) — they are not needed to
evaluate a tokenizer and would force every eval-only stub to implement them.

Recommended pattern for concrete tokenizers (e.g. EEGTokenizer):

    class EEGTokenizer(nn.Module):
        codebook_size: int

        def __init__(self, codebook_size: int = 1024, ...):
            super().__init__()
            self.codebook_size = codebook_size
            ...

        def tokenize(self, x: torch.Tensor) -> torch.Tensor: ...
        def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor: ...

Do NOT inherit from Tokenizer — it's a Protocol, satisfied structurally. The
whole point is that 4M's VQ class (which doesn't know about us) drops in by
shape alone; same applies to your own tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


@runtime_checkable
class Tokenizer(Protocol):
    """The minimal interface a tokenizer must satisfy.

    Tokens may have any shape `(B, ...)`. The harness flattens to `(B, L)`
    internally where it needs a sequence (e.g. for bigram entropy reading
    row-major). This is what makes 4M image tokenizers — which return
    `(B, H_q, W_q)` — work alongside our EEG/MEG tokenizers that return
    `(B, L)`.

    Optional method (checked via `has_token_embeddings` below):
        tokens_to_embedding(tokens) -> float tensor with one extra trailing
            dim of size D. The probe uses these continuous codebook embeddings
            as features when available — strictly more informative than the
            bag-of-codes fallback.
    """

    codebook_size: int

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of signals to discrete token IDs.

        Args:
            x: (B, C, T) float tensor.

        Returns:
            (B, ...) long tensor with values in [0, codebook_size).
        """
        ...

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct signals from token IDs.

        Args:
            tokens: long tensor of the same shape `tokenize` returned.

        Returns:
            (B, C, T) float tensor matching the original signal shape.
        """
        ...


def has_token_embeddings(tokenizer: Tokenizer) -> bool:
    """Whether the tokenizer exposes the optional tokens_to_embedding method."""
    return hasattr(tokenizer, "tokens_to_embedding") and callable(
        tokenizer.tokens_to_embedding
    )


@dataclass
class EvalConfig:
    """Modality-specific evaluation knobs.

    Nothing about signal shape lives here — the harness reads C, T, L from the
    tensors at runtime. Only modality-level constants (sample rate, frequency
    bands, class count) and harness-level knobs (device, seed) go here.
    """

    sample_rate_hz: float
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )
    device: str = "cpu"
    batch_size: int = 64
    seed: int = 0

    run_reconstruction: bool = True
    run_codebook: bool = True
    run_probe: bool = True
    run_sequence: bool = True
    run_retrieval: bool = True

    probe_epochs: int = 100
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4
    probe_top_k: tuple[int, ...] = (1, 5)
    probe_test_frac: float = 0.2
    # Class-balanced cross-entropy (per-fold weights = N / (n_classes * count_c)).
    # On (real) imbalanced THINGS labels the unweighted optimum collapses to
    # majority prediction even when features carry signal in minority classes
    # — weighting equalizes gradient mass across classes so the head is given
    # a fair shot at finding signal everywhere. Same data, lower-variance than
    # subsampling-to-balance.
    probe_class_weighted: bool = True
    # Probe classifier head. "linear" (default) is the §5.3 gate that
    # mirrors what 4M's input embedding can read off. "mlp" is a diagnostic
    # escape hatch — if MLP succeeds where linear fails, the tokens carry
    # *some* nonlinear information but 4M still can't use it cleanly.
    probe_classifier: str = "linear"  # "linear" | "mlp" | "cnn"
    probe_mlp_hidden: int = 256
    probe_mlp_dropout: float = 0.5
    # CNN head — temporal-spatial inductive bias for token / raw features.
    # Dispatches to Conv1d (for 3D inputs (B, C, T)) or Conv2d (for 4D inputs
    # (B, S1, S2, D=embedding_dim)) based on feature shape.
    probe_cnn_hidden: int = 64
    # K-fold CV for the probe. n_folds=1 falls back to a single random
    # (1 - probe_test_frac, probe_test_frac) split — back-compat with the
    # original single-split behavior. n_folds>=2 runs proper k-fold and the
    # output keys gain `_mean` / `_std` suffixes.
    probe_n_folds: int = 5
    # Per-RVQ-layer probing for tokenizers that expose a layered
    # `tokens_to_embedding(tokens, layers=...)`. Each element is either:
    #   - None  → sum every available codebook layer (current default)
    #   - a tuple of layer indices to sum, e.g. (0,) for RVQ0 alone
    # Tokenizers without layered embeddings ignore this entirely and just
    # emit a single "tokens" entry.
    probe_rvq_layers: tuple[tuple[int, ...] | None, ...] = (None, (0,))

    psd_nperseg: int | None = None

    # Cap probe and retrieval to avoid O(B) and O(B²) memory blowup.
    # 0 = no cap. The codebook and sequence axes always use the full dataset.
    probe_max_samples: int = 50000
    retrieval_max_samples: int = 5000


@dataclass
class MetricResult:
    """Output of a single metric module (one of the four §5 axes)."""

    name: str
    values: dict[str, float]

    def __str__(self) -> str:
        lines = [f"  [{self.name}]"]
        for k, v in self.values.items():
            lines.append(f"    {k:<32s} {v:.4f}")
        return "\n".join(lines)


@dataclass
class TokenizerReport:
    """Aggregated multi-axis report. Any axis may be None if disabled in config."""

    reconstruction: MetricResult | None = None
    codebook: MetricResult | None = None
    probe: MetricResult | None = None
    sequence: MetricResult | None = None
    retrieval: MetricResult | None = None

    def axes(self) -> list[MetricResult]:
        return [
            m
            for m in (
                self.reconstruction,
                self.codebook,
                self.probe,
                self.sequence,
                self.retrieval,
            )
            if m is not None
        ]

    def __str__(self) -> str:
        return "TokenizerReport\n" + "\n".join(str(m) for m in self.axes())
