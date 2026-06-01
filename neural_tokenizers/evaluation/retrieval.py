"""§5.5 model-free retrieval probe.

Does the token *representation* place trials of the same class nearer to each
other than to trials of other classes? This is asked **without** training a
classifier, so it cleanly separates "token quality is bad" from "the trained
probe (linear / MLP) was the bottleneck."

For each query trial we compute the K nearest neighbors in feature space and
measure **precision@K**: the fraction of retrieved neighbors that share the
query's class label. We compute it three ways for parity with the §5.3 probe:

  - tokens : same featurization the §5.3 probe uses on this tokenizer
             (mean of `tokens_to_embedding` if available; bag-of-codes else).
             For BrainOmni: dense embeddings → cosine similarity.
             For μ-transform with the new `tokens_to_embedding`: bin-center
             values → cosine similarity in (channel × time) feature space.
  - raw    : flattened signal x — same upper-bound bracket as §5.3.
  - random : uniformly random tokens, same shape — same lower bracket.

Also reported for BrainOmni-style sparse codecs:
  - tokens_jaccard : Jaccard similarity on token *sets* (vocab-subset used per
                     trial). Position-blind by design, but matches the user's
                     mental model of "do similar trials share codewords?"

Chance precision = class_frequency averaged over queries, exactly. With class
imbalance we report **balanced precision@K**: average of per-class precision
contributions weighted equally per class.

Why this is the right escape valve from the §5.3 result:
  If `tokens` retrieval precision is at chance, no trained classifier (linear,
  MLP, transformer) can pull signal out of features that don't have it — the
  feature geometry simply doesn't separate classes. If retrieval precision is
  well above chance but §5.3 sits at chance, the bottleneck is the §5.3
  classifier (architecture / capacity / regularization), not the tokens.
"""

from __future__ import annotations

import inspect

import torch

from .codebook import tokenize_all
from .protocol import EvalConfig, MetricResult, Tokenizer, has_token_embeddings


def compute_retrieval_metrics(
    tokenizer: Tokenizer,
    signal: torch.Tensor,
    labels: torch.Tensor,
    config: EvalConfig,
    tokens: torch.Tensor | None = None,
) -> MetricResult:
    """k-NN retrieval precision@K, model-free, on the same featurization the
    classifier-based §5.3 probe uses."""
    if signal.ndim != 3:
        raise ValueError(f"signal must be (B, C, T); got {tuple(signal.shape)}")
    if labels.ndim != 1 or labels.shape[0] != signal.shape[0]:
        raise ValueError(
            f"labels must be (B,) matching signal[0]; got "
            f"{tuple(labels.shape)} vs B={signal.shape[0]}"
        )

    if tokens is None:
        with torch.no_grad():
            tokens = tokenize_all(tokenizer, signal, config)

    y = labels.to(torch.long).cpu()
    n_classes = int(y.max().item()) + 1
    Ks = tuple(int(k) for k in config.probe_top_k)

    with torch.no_grad():
        feature_sets = _build_retrieval_features(tokenizer, tokens, signal, config)
        token_sets_for_jaccard = _token_sets_per_trial(tokens, tokenizer.codebook_size)

    values: dict[str, float] = {}
    for name, feats in feature_sets.items():
        precisions = _cosine_precision_at_k(feats, y, Ks, n_classes)
        for k, p in precisions.items():
            values[f"prec@{k}_{name}"] = p

    # Jaccard precision is meaningful only when token sets vary meaningfully
    # across trials. For dense per-sample codecs (μ-transform), every trial
    # uses ~all of the small vocab, so set sizes are saturated; we skip it
    # there and rely on the cosine retrieval on bin-center features instead.
    if _jaccard_makes_sense(token_sets_for_jaccard, tokenizer.codebook_size):
        jaccard_prec = _jaccard_precision_at_k(token_sets_for_jaccard, y, Ks, n_classes)
        for k, p in jaccard_prec.items():
            values[f"prec@{k}_tokens_jaccard"] = p

    return MetricResult(name="retrieval", values=values)


# ---------- featurization (mirrors probe.py) ------------------------------


def _build_retrieval_features(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    signal: torch.Tensor,
    config: EvalConfig,
) -> dict[str, torch.Tensor]:
    """Same featurization the classifier probe uses, so the comparison is
    apples-to-apples: any quality gap between classifier and retrieval is
    attributable to the classifier, not the features."""
    feats: dict[str, torch.Tensor] = {}
    if has_token_embeddings(tokenizer):
        fn = getattr(tokenizer, "tokens_to_embedding")
        try:
            layered = "layers" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            layered = False
        feats["tokens"] = _embed_and_mean(tokenizer, tokens, config, layered=layered)
    else:
        feats["tokens"] = _bag_of_codes(tokens, tokenizer.codebook_size)
    feats["raw"] = signal.reshape(signal.shape[0], -1).to(torch.float32).cpu()
    feats["random"] = _bag_of_codes(
        _random_tokens_like(tokens, tokenizer.codebook_size, config.seed),
        tokenizer.codebook_size,
    )
    return feats


@torch.no_grad()
def _embed_and_mean(
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    config: EvalConfig,
    layered: bool,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], config.batch_size):
        chunk = tokens[start : start + config.batch_size].to(config.device)
        if layered:
            emb = tokenizer.tokens_to_embedding(chunk, layers=None)  # type: ignore[call-arg]
        else:
            emb = tokenizer.tokens_to_embedding(chunk)  # type: ignore[attr-defined]
        emb_flat = emb.reshape(emb.shape[0], -1, emb.shape[-1])
        pieces.append(emb_flat.mean(dim=1).cpu().to(torch.float32))
    return torch.cat(pieces, dim=0)


def _bag_of_codes(tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    flat = tokens.reshape(tokens.shape[0], -1).to(torch.long).cpu()
    B, L = flat.shape
    out = torch.zeros(B, codebook_size, dtype=torch.float32)
    out.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    return out / max(L, 1)


def _random_tokens_like(tokens: torch.Tensor, codebook_size: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, codebook_size, tokens.shape, generator=g, dtype=torch.long)


# ---------- Jaccard machinery --------------------------------------------


def _token_sets_per_trial(tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """One row of a binary `(B, V)` membership matrix per trial. Cheap to
    compute and to AND/OR for Jaccard."""
    flat = tokens.reshape(tokens.shape[0], -1).to(torch.long).cpu()
    B, _ = flat.shape
    out = torch.zeros(B, codebook_size, dtype=torch.bool)
    ones = torch.ones_like(flat, dtype=torch.bool)
    out.scatter_(1, flat, ones)
    return out


def _jaccard_makes_sense(token_sets: torch.Tensor, codebook_size: int) -> bool:
    """Heuristic: skip Jaccard when token sets are saturated.

    If the median trial uses >80% of the vocab, Jaccard ≈ 1 for every pair
    and the metric collapses to chance regardless of token quality. This is
    the μ-transform regime (dense per-(channel, time) codecs) where the
    cosine retrieval on the *value* featurization is the right diagnostic.
    """
    used_per_trial = token_sets.to(torch.int64).sum(dim=1).to(torch.float64)
    median_frac = float(used_per_trial.median().item()) / max(1, codebook_size)
    return median_frac <= 0.80


@torch.no_grad()
def _jaccard_precision_at_k(
    token_sets: torch.Tensor,  # (B, V) bool
    labels: torch.Tensor,
    Ks: tuple[int, ...],
    n_classes: int,
) -> dict[int, float]:
    """Pairwise Jaccard similarity, k-NN retrieval, balanced precision@K."""
    A = token_sets.to(torch.float32)
    # |A ∩ B| = A @ B^T (bool counts)
    intersection = A @ A.T
    set_size = A.sum(dim=1, keepdim=True)
    # |A ∪ B| = |A| + |B| - |A ∩ B|
    union = set_size + set_size.T - intersection
    jaccard = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return _precision_at_k_from_similarity(jaccard, labels, Ks, n_classes)


# ---------- cosine machinery ---------------------------------------------


@torch.no_grad()
def _cosine_precision_at_k(
    feats: torch.Tensor,  # (B, D) float
    labels: torch.Tensor,
    Ks: tuple[int, ...],
    n_classes: int,
) -> dict[int, float]:
    """L2-normalize features, similarity = X @ X^T, balanced precision@K."""
    norms = feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Xn = feats / norms
    sim = Xn @ Xn.T
    return _precision_at_k_from_similarity(sim, labels, Ks, n_classes)


# ---------- shared k-NN precision logic ----------------------------------


def _precision_at_k_from_similarity(
    sim: torch.Tensor,  # (B, B) float
    labels: torch.Tensor,
    Ks: tuple[int, ...],
    n_classes: int,
) -> dict[int, float]:
    """For each row (query), top-K excluding self, fraction matching label.
    Reported as **balanced** precision: mean over classes of per-class avg.

    Balanced because class imbalance otherwise lets a method that "retrieves
    similar things" score high on the majority class and drag the average up,
    even if minority classes are unrecoverable.
    """
    B = sim.shape[0]
    self_mask = torch.eye(B, dtype=torch.bool)
    sim_masked = sim.masked_fill(self_mask, float("-inf"))

    out: dict[int, float] = {}
    for k in Ks:
        k_eff = min(k, B - 1)
        topk_idx = sim_masked.topk(k_eff, dim=1).indices  # (B, k)
        retrieved_labels = labels[topk_idx]  # (B, k)
        correct = (retrieved_labels == labels.unsqueeze(1)).to(torch.float64)
        per_query_precision = correct.mean(dim=1)  # (B,)
        # Balance across classes.
        per_class_precisions: list[float] = []
        for c in range(n_classes):
            mask = labels == c
            if mask.any():
                per_class_precisions.append(per_query_precision[mask].mean().item())
        out[k] = (
            float(sum(per_class_precisions) / len(per_class_precisions))
            if per_class_precisions
            else 0.0
        )
    return out
