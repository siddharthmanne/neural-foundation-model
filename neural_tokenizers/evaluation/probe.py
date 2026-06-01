"""§5.3 Downstream linear probe.

Does the token sequence still carry the information we actually care about?
Train a linear classifier (cross-entropy logistic regression in torch) on
feature sets that always include `raw` and `random` as bracketing anchors,
plus one or more token-derived feature sets:

  - tokens_<subset> : per-trial features pooled from the tokenizer's output.
        If the tokenizer exposes a layered `tokens_to_embedding(tokens, layers)`,
        one entry is produced per `EvalConfig.probe_rvq_layers` element
        (e.g. `tokens_all`, `tokens_rvq0`). Otherwise a single `tokens`
        entry is emitted (codebook embeddings if available, else
        bag-of-codes).
  - raw    : flattened signal x (upper bound — all info is there)
  - random : bag-of-codes of uniformly random tokens of the same shape
             (lower bound — no info)

K-fold cross-validation:
  Set `EvalConfig.probe_n_folds >= 2` to run k-fold CV; the harness reports
  `top{k}_<set>_mean` and `top{k}_<set>_std` across folds. With n_folds=1 we
  fall back to a single random (train/test) split for back-compat.

The probe is intentionally weak (one linear layer): if a linear model can
read the stimulus off the tokens, the information is linearly decodable,
which is what 4M's transformer needs from its input embeddings. A strong
nonlinear classifier would rescue tokens that are entangled in ways 4M
cannot cheaply undo, and would therefore overstate quality.
"""

from __future__ import annotations

import inspect
import statistics

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
    """Run the bracketed linear probe with optional k-fold CV.

    NOT decorated with @torch.no_grad() — the linear head needs gradients.
    Featurization is wrapped in no_grad inside its helpers; the gradient-
    bearing scope is limited to _train_and_score_probe.
    """
    _assert_probe_inputs(signal, labels)

    if tokens is None:
        with torch.no_grad():
            tokens = tokenize_all(tokenizer, signal, config)

    y = labels.to(torch.long).cpu()

    with torch.no_grad():
        feature_sets = _build_feature_sets(tokenizer, tokens, signal, config)

    n_classes = int(labels.max().item()) + 1
    folds = _kfold_indices(n=signal.shape[0], n_folds=config.probe_n_folds, config=config)

    values: dict[str, float] = {}
    for name, feats in feature_sets.items():
        per_fold: dict[str, list[float]] = {}
        for train_idx, test_idx in folds:
            accs = _train_and_score_probe(feats, y, train_idx, test_idx, n_classes, config)
            for metric_name, acc in accs.items():
                per_fold.setdefault(metric_name, []).append(acc)
        for metric_name, vals in per_fold.items():
            values[f"{metric_name}_{name}_mean"] = float(statistics.mean(vals))
            values[f"{metric_name}_{name}_std"] = (
                float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
            )
    return MetricResult(name="probe", values=values)


# ---------- feature construction -----------------------------------------

def _build_feature_sets(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    signal: torch.Tensor,
    config: EvalConfig,
) -> dict[str, torch.Tensor]:
    """Return {feature_set_name: feature tensor on CPU, float32}.

    Shape varies by `config.probe_classifier`:
      - "linear" / "mlp": (B, D_flat) — mean-pooled / bag-of-codes / flattened.
      - "cnn"           : (B, *natural_spatial[, D]) — structured for conv heads.
                          See _embed_keep_structure for the natural-shape rules.

    Order matters only for stable JSON output: tokens variants first, then
    raw, then random.
    """
    use_cnn = config.probe_classifier == "cnn"

    feats: dict[str, torch.Tensor] = {}
    if has_token_embeddings(tokenizer):
        embed_fn = _embed_keep_structure if use_cnn else _embed_and_mean
        if _supports_layered_embedding(tokenizer):
            for subset in config.probe_rvq_layers:
                name = f"tokens_{_layer_subset_name(subset)}"
                feats[name] = embed_fn(tokenizer, tokens, config, layers=subset)
        else:
            feats["tokens"] = embed_fn(tokenizer, tokens, config, layers=None)
    else:
        if use_cnn:
            raise ValueError(
                "CNN probe requires tokenizer.tokens_to_embedding; "
                f"{type(tokenizer).__name__} only exposes bag-of-codes features."
            )
        feats["tokens"] = _bag_of_codes(tokens, tokenizer.codebook_size)

    random_tokens = _random_tokens_like(tokens, tokenizer.codebook_size, config.seed)
    if use_cnn:
        feats["raw"] = signal.to(torch.float32).cpu()
        if has_token_embeddings(tokenizer):
            feats["random"] = _embed_keep_structure(
                tokenizer, random_tokens, config, layers=None
            )
        else:
            # Defensive — earlier branch already raised for this case.
            feats["random"] = _bag_of_codes(random_tokens, tokenizer.codebook_size)
    else:
        feats["raw"] = signal.reshape(signal.shape[0], -1).to(torch.float32).cpu()
        feats["random"] = _bag_of_codes(random_tokens, tokenizer.codebook_size)
    return feats


def _supports_layered_embedding(tokenizer: Tokenizer) -> bool:
    """True if `tokens_to_embedding` accepts a `layers` kwarg.

    We don't import the BrainOmni adapter here — keep `probe.py` modality-
    agnostic. Signature sniff is the cheapest way to opt in without a flag.
    """
    fn = getattr(tokenizer, "tokens_to_embedding", None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "layers" in sig.parameters


def _layer_subset_name(layers: tuple[int, ...] | None) -> str:
    """Stable name for a layer subset (used as JSON key suffix)."""
    if layers is None:
        return "all"
    return "rvq" + "".join(str(int(layer)) for layer in layers)


@torch.no_grad()
def _embed_and_mean(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    config: EvalConfig,
    layers: tuple[int, ...] | None,
) -> torch.Tensor:
    """Pool the per-token embeddings to a single (B, D) feature per trial.

    `layers=None` means "use whatever the tokenizer's default is" (current
    behavior: sum all RVQ layers for BrainOmni). A tuple selects a subset.
    """
    pieces: list[torch.Tensor] = []
    layered = _supports_layered_embedding(tokenizer)
    for start in range(0, tokens.shape[0], config.batch_size):
        chunk = tokens[start : start + config.batch_size].to(config.device)
        if layered:
            emb = tokenizer.tokens_to_embedding(chunk, layers=layers)  # type: ignore[call-arg]
        else:
            emb = tokenizer.tokens_to_embedding(chunk)  # type: ignore[attr-defined]
        emb_flat = emb.reshape(emb.shape[0], -1, emb.shape[-1])
        pieces.append(emb_flat.mean(dim=1).cpu().to(torch.float32))
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def _embed_keep_structure(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    config: EvalConfig,
    layers: tuple[int, ...] | None,
) -> torch.Tensor:
    """Like `_embed_and_mean` but **preserves the natural spatial/temporal
    layout** instead of mean-pooling. Used by the CNN probe head.

    Shape conventions (assumes tokens shape is `(B, *natural, [RVQ?])`):
      - RVQ tokenizers (e.g. BrainOmni, tokens (B, 16, 8, 4)):
          → emb (B, 16, 8, D)
      - Single-codebook tokenizers (e.g. μ-transform, tokens (B, 271, 281)):
          → emb (B, 271, 281, D); D=1 is squeezed so CNN1D sees (B, 271, 281).

    Why D=1 squeeze: μ-transform's `tokens_to_embedding` returns scalar
    bin-center values per (channel, time-sample) position. Keeping the
    trailing D=1 would force the probe to use Conv2d when Conv1d over
    (channels, time) is the natural inductive bias.
    """
    layered = _supports_layered_embedding(tokenizer)
    if layered:
        # Last token axis is the RVQ-layer index; the spatial layout
        # is everything between the batch dim and the RVQ dim.
        natural_shape = tokens.shape[1:-1]
    else:
        natural_shape = tokens.shape[1:]

    pieces: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], config.batch_size):
        chunk = tokens[start : start + config.batch_size].to(config.device)
        if layered:
            emb = tokenizer.tokens_to_embedding(chunk, layers=layers)  # type: ignore[call-arg]
        else:
            emb = tokenizer.tokens_to_embedding(chunk)  # type: ignore[attr-defined]
        D = emb.shape[-1]
        emb = emb.reshape(emb.shape[0], *natural_shape, D)
        if D == 1:
            emb = emb.squeeze(-1)
        pieces.append(emb.cpu().to(torch.float32))
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


# ---------- splits & training -------------------------------------------

def _assert_probe_inputs(signal: torch.Tensor, labels: torch.Tensor) -> None:
    if signal.ndim != 3:
        raise ValueError(f"signal must be (B, C, T); got {tuple(signal.shape)}")
    if labels.ndim != 1 or labels.shape[0] != signal.shape[0]:
        raise ValueError(
            f"labels must be (B,) matching signal[0]; got {tuple(labels.shape)} vs B={signal.shape[0]}"
        )


def _kfold_indices(
    n: int, n_folds: int, config: EvalConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """K-fold permutation-based split.

    n_folds=1 → one (train, test) split sized by `config.probe_test_frac`
    (preserves old single-split behavior). n_folds>=2 → equal-sized folds;
    the last fold absorbs the remainder so all samples are tested exactly
    once across folds.
    """
    if n_folds < 1:
        raise ValueError(f"probe_n_folds must be >= 1, got {n_folds}")
    g = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(n, generator=g)
    if n_folds == 1:
        n_test = max(1, int(round(config.probe_test_frac * n)))
        return [(perm[n_test:], perm[:n_test])]
    if n_folds > n:
        raise ValueError(f"probe_n_folds={n_folds} exceeds sample count n={n}")
    fold_size = n // n_folds
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n
        test_idx = perm[start:end]
        train_idx = torch.cat([perm[:start], perm[end:]])
        out.append((train_idx, test_idx))
    return out


def _train_and_score_probe(
    feats: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    n_classes: int,
    config: EvalConfig,
) -> dict[str, float]:
    """Fit a single nn.Linear with full-batch gradient descent.

    Returns a dict of named metrics: `top{k}` for each k in probe_top_k, and
    `bal_acc` (balanced accuracy = mean of per-class recalls). Balanced acc
    is the imbalance-invariant diagnostic: chance is exactly 1/n_classes
    regardless of class frequencies, whereas top-1 floor is `max(class_freq)`
    once a classifier collapses to majority prediction.
    """
    device = config.device
    X_train = feats[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    X_test = feats[test_idx].to(device)
    y_test = labels[test_idx].to(device)

    feat_shape = tuple(feats.shape[1:])
    torch.manual_seed(config.seed)
    head = _build_head(feat_shape, n_classes, config).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=config.probe_lr, weight_decay=config.probe_weight_decay
    )

    class_weights = _class_weights(y_train, n_classes, device) if config.probe_class_weighted else None

    head.train()
    for _ in range(config.probe_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(X_train), y_train, weight=class_weights)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        logits = head(X_test)
    out: dict[str, float] = {
        f"top{k}": acc
        for k, acc in _topk_accuracy(logits, y_test, config.probe_top_k).items()
    }
    out["bal_acc"] = _balanced_accuracy(logits, y_test, n_classes)
    return out


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


def _build_head(
    feat_shape: tuple[int, ...], n_classes: int, config: EvalConfig
) -> torch.nn.Module:
    """Construct the probe classifier per `config.probe_classifier`.

    Args:
        feat_shape: feature shape **excluding** the batch dim. e.g.
            `(D,)` for flat features; `(C, T)` for 1-D time-series;
            `(S1, S2, D)` for 2-D structured embeddings.
        n_classes: classification target size.
        config: EvalConfig.

    Heads:
        "linear" — §5.3 gate (what 4M's input embedding effectively does).
                   Flatten + Linear.
        "mlp"    — 2-layer ReLU+dropout. Tests "info is there but nonlinear."
        "cnn"    — Conv1d head (3-D inputs) or Conv2d head (4-D inputs).
                   Tests "info needs the right inductive bias to surface."
    """
    feat_dim = int(torch.tensor(feat_shape).prod().item())
    clf = config.probe_classifier
    if clf == "linear":
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feat_dim, n_classes),
        )
    if clf == "mlp":
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feat_dim, config.probe_mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.probe_mlp_dropout),
            torch.nn.Linear(config.probe_mlp_hidden, n_classes),
        )
    if clf == "cnn":
        return _build_cnn_head(feat_shape, n_classes, hidden=config.probe_cnn_hidden)
    raise ValueError(
        f"probe_classifier must be 'linear', 'mlp', or 'cnn'; got {clf!r}"
    )


def _build_cnn_head(
    feat_shape: tuple[int, ...], n_classes: int, hidden: int
) -> torch.nn.Module:
    """Shape-dispatched CNN head.

      - (C, T)      → CNN1D: channels-as-input, conv over time.
      - (S1, S2, D) → CNN2D: D-as-input, conv over (S1, S2).

    Other shapes (e.g. flat or 4-D non-token) are rejected — they don't have
    a natural conv interpretation and the user should switch to linear/mlp.
    """
    if len(feat_shape) == 2:
        in_channels, _t = feat_shape
        return _CNN1DHead(in_channels=in_channels, n_classes=n_classes, hidden=hidden)
    if len(feat_shape) == 3:
        _s1, _s2, d_embed = feat_shape
        return _CNN2DHead(in_channels=d_embed, n_classes=n_classes, hidden=hidden)
    raise ValueError(
        f"CNN head expects features with shape (C, T) or (S1, S2, D); "
        f"got {feat_shape!r}"
    )


class _CNN1DHead(torch.nn.Module):
    """Conv1d → BN → ReLU ×2 → AdaptiveAvgPool1d → Linear.

    Designed for (B, C, T) — raw MEG (C=271, T=281) or μ-transform
    dequantized tokens (same shape).
    """

    def __init__(self, in_channels: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) → (B, n_classes)
        return self.net(x)


class _CNN2DHead(torch.nn.Module):
    """Conv2d → BN → ReLU ×2 → AdaptiveAvgPool2d → Linear.

    Designed for (B, S1, S2, D) — BrainOmni tokens decoded through
    `tokens_to_embedding` and kept structured: S1=16 latent, S2=8 time,
    D=embedding dim. The forward pass permutes D into the channel slot.
    """

    def __init__(self, in_channels: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S1, S2, D)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.net(x)


def _class_weights(
    y_train: torch.Tensor, n_classes: int, device: torch.device | str
) -> torch.Tensor:
    """sklearn-style balanced weights: w_c = N_total / (n_classes * count_c).

    Computed per-fold from the training labels (not globally), so each fold's
    weights match its own class distribution. Classes absent from this fold
    fall back to the unweighted-uniform default (N_total / n_classes) — a
    defensive choice that prevents `0 * +inf` if any class happens to be
    missing on a small fold.
    """
    counts = torch.bincount(y_train, minlength=n_classes).to(torch.float64)
    n_total = float(y_train.shape[0])
    default = n_total / n_classes
    weights = torch.where(
        counts > 0,
        n_total / (n_classes * counts.clamp(min=1.0)),
        torch.full_like(counts, default),
    )
    return weights.to(device=device, dtype=torch.float32)


def _balanced_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> float:
    """Mean of per-class recalls. Invariant to class imbalance.

    Classes absent from the test set contribute nothing (averaged over
    only the classes that *do* appear). This matters when a fold happens
    to miss a tiny class — the alternative (count 0 recall for missing
    classes) penalizes folds for sampling variance, not classifier quality.
    """
    preds = logits.argmax(dim=1)
    per_class_recall: list[float] = []
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            per_class_recall.append((preds[mask] == c).to(torch.float64).mean().item())
    return float(sum(per_class_recall) / len(per_class_recall)) if per_class_recall else 0.0
