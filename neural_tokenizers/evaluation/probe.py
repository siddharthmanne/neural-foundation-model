"""§5.3 Downstream linear probe.

Does the token sequence still carry the information we actually care about?
Train a linear classifier (cross-entropy logistic regression in torch) on
three feature sets, all on the same train/test split:

  - tokens : features derived from the tokenizer's output
            (codebook embeddings if tokens_to_embedding exists, else
             bag-of-codes histograms — see _featurize_tokens)
  - raw    : flattened signal x (upper-bound: all info is there)
  - random : bag-of-codes of uniformly random tokens of the same shape
            (lower-bound: no info)

A good tokenizer's `top_k_tokens` score lands close to `top_k_raw`. If it sits
near `top_k_random`, the tokenizer threw the task-relevant signal away — even
if reconstruction looks fine.

The probe is intentionally weak (one linear layer): if a linear model can read
the stimulus off the tokens, the information is linearly decodable, which is
what 4M's transformer needs from its input embeddings. A strong nonlinear
classifier could rescue tokens that are entangled in ways 4M cannot cheaply
undo, and would therefore overstate quality.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .codebook import tokenize_all
from .protocol import EvalConfig, MetricResult, Tokenizer, has_token_embeddings


def compute_probe_metrics(
    tokenizer: Tokenizer,
    signal: torch.Tensor,
    labels: torch.Tensor,
    config: EvalConfig,
    tokens: torch.Tensor | None = None,
) -> MetricResult:
    """Run the three-way linear probe comparison.

    NOT decorated with @torch.no_grad() — the linear head needs gradients to
    train. Featurization (tokenize, tokens_to_embedding) is wrapped in no_grad
    inside its helpers; the gradient-bearing scope is limited to _train_and_score_probe.
    """
    _assert_probe_inputs(signal, labels)

    if tokens is None:
        with torch.no_grad():
            tokens = tokenize_all(tokenizer, signal, config)

    train_idx, test_idx = _train_test_split(
        n=signal.shape[0], test_frac=config.probe_test_frac, seed=config.seed
    )
    y = labels.to(torch.long).cpu()

    with torch.no_grad():
        feat_tokens = _featurize_tokens(tokenizer, tokens, config)
        feat_raw = signal.reshape(signal.shape[0], -1).to(torch.float32).cpu()
        feat_random = _bag_of_codes(
            _random_tokens_like(tokens, tokenizer.codebook_size, config.seed),
            tokenizer.codebook_size,
        )

    n_classes = int(labels.max().item()) + 1
    values: dict[str, float] = {}
    for name, feats in (("tokens", feat_tokens), ("raw", feat_raw), ("random", feat_random)):
        accs = _train_and_score_probe(feats, y, train_idx, test_idx, n_classes, config)
        for k, acc in accs.items():
            values[f"top{k}_{name}"] = acc
    return MetricResult(name="probe", values=values)


def _assert_probe_inputs(signal: torch.Tensor, labels: torch.Tensor) -> None:
    if signal.ndim != 3:
        raise ValueError(f"signal must be (B, C, T); got {tuple(signal.shape)}")
    if labels.ndim != 1 or labels.shape[0] != signal.shape[0]:
        raise ValueError(
            f"labels must be (B,) matching signal[0]; got {tuple(labels.shape)} vs B={signal.shape[0]}"
        )


def _train_test_split(n: int, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(round(test_frac * n)))
    return perm[n_test:], perm[:n_test]


def _featurize_tokens(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    config: EvalConfig,
) -> torch.Tensor:
    """Embed-and-mean if the tokenizer exposes tokens_to_embedding, else BoW.

    Embeddings are strictly more informative — they preserve geometry across
    codebook entries — but require the optional protocol method.
    """
    if has_token_embeddings(tokenizer):
        return _embed_and_mean(tokenizer, tokens, config)
    return _bag_of_codes(tokens, tokenizer.codebook_size)


@torch.no_grad()
def _embed_and_mean(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    config: EvalConfig,
) -> torch.Tensor:
    """Pool the per-token embeddings to a single (B, D) feature per trial."""
    pieces: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], config.batch_size):
        chunk = tokens[start : start + config.batch_size].to(config.device)
        emb = tokenizer.tokens_to_embedding(chunk)  # type: ignore[attr-defined]
        emb_flat = emb.reshape(emb.shape[0], -1, emb.shape[-1])
        pieces.append(emb_flat.mean(dim=1).cpu().to(torch.float32))
    return torch.cat(pieces, dim=0)


def _bag_of_codes(tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """Histogram of code occurrences per trial: (B, V) float."""
    flat = tokens.reshape(tokens.shape[0], -1).to(torch.long).cpu()
    B, L = flat.shape
    out = torch.zeros(B, codebook_size, dtype=torch.float32)
    out.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    return out / max(L, 1)


def _random_tokens_like(tokens: torch.Tensor, codebook_size: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, codebook_size, tokens.shape, generator=g, dtype=torch.long)


def _train_and_score_probe(
    feats: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    n_classes: int,
    config: EvalConfig,
) -> dict[int, float]:
    """Fit a single nn.Linear with full-batch gradient descent; return top-k acc."""
    device = config.device
    X_train = feats[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    X_test = feats[test_idx].to(device)
    y_test = labels[test_idx].to(device)

    feat_dim = feats.shape[1]
    torch.manual_seed(config.seed)
    head = torch.nn.Linear(feat_dim, n_classes).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=config.probe_lr, weight_decay=config.probe_weight_decay
    )

    head.train()
    for _ in range(config.probe_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(X_train), y_train)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        logits = head(X_test)
    return _topk_accuracy(logits, y_test, config.probe_top_k)


def _topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...]
) -> dict[int, float]:
    """Top-k accuracy. Falls back to top-min(k, n_classes) silently."""
    n_classes = logits.shape[1]
    out: dict[int, float] = {}
    for k in ks:
        k_eff = min(k, n_classes)
        top = logits.topk(k_eff, dim=1).indices
        correct = (top == labels.unsqueeze(1)).any(dim=1).to(torch.float64)
        out[k] = float(correct.mean().item())
    return out
