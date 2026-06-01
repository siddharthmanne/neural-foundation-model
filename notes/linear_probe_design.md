# The §5.3 linear probe — what it tests, and why these choices

Companion to `neural_tokenizers/evaluation/probe.py` and `neural_tokenizers/CLAUDE.md §5.3`. If you forget why the probe is shaped the way it is, read this.

## What question the probe answers

> "If I freeze the tokenizer and try to read the stimulus off the tokens with the simplest possible classifier, can I?"

This is the question 4M's transformer implicitly asks of every modality it ingests. 4M's first layer is a learnable embedding lookup followed by a transformer block — its very first "read" of any token is a **linear** operation (embed + sum). If a linear probe on top of the tokens can't recover anything semantically meaningful, 4M's transformer is not going to magically rescue it from a single embed lookup.

So the probe is not asking "is the tokenizer good?" — it's asking "is the information **linearly decodable**?"

## Why linear, not MLP/transformer

The probe is one `nn.Linear` followed by softmax — no hidden layers, no nonlinearity. Two reasons:

1. **Stronger classifiers overstate quality.** An MLP could rescue tokens whose semantic information is tangled in ways a linear layer can't undo. 4M's first read can't undo that tangling either; an MLP probe would mask the actual problem.
2. **Linear separability is the relevant ceiling.** What 4M needs from its input embeddings is roughly "linearly separable representation of useful signal." If a linear probe can hit it, the tokens are 4M-ready in the only sense that matters. If it can't, no upstream transformer block ever sees the signal in usable form.

The corollary: a tokenizer whose tokens **require** a strong classifier to decode is **not** a tokenizer 4M can use. That's the whole point of the gate.

## The 3-way comparison: raw, tokens, random

Every probe run reports three numbers per top-k:

| Set | Feature | Role |
|---|---|---|
| `raw` | Flattened `(C·T,)` MEG signal | Upper bound — all info is there |
| `tokens` | Tokenizer output, pooled | The thing being tested |
| `random` | Uniform random tokens | Lower bound — no info |

These are bracketing anchors. The tokens score in isolation is uninterpretable:
- Is 5% top-1 "good"? Depends on the chance rate.
- Is "tokens beats random by 2%" a real signal? Depends on the noise floor.

With brackets: **tokens should sit closer to raw than to random.** If tokens ≈ random, the tokenizer threw the signal away. If tokens ≈ raw, it preserved everything useful. The gap to either side is the actual diagnostic.

## The label space — why 27, not 1854

THINGS-MEG has three nested label spaces:

| Space | Size | Origin |
|---|---|---|
| Image ID | ~22,449 | One per unique image shown |
| Concept ID | 1,854 | One per THINGS object category |
| Superordinate | **27** | High-level categories ("animal", "food", "vehicle", …) |

The original probe used 1,854 (densified to ~1,152 by dropping concepts absent from the eval sample). At ~3,000 eval trials this gave **~2.6 trials per class** — a linear head literally cannot fit per-class weights from two samples. Both raw and tokens sat at chance, so we couldn't distinguish "tokenizer is bad" from "probe is starved of data."

27-superordinate gives ~110 trials/class at the same sample size. That's enough for a linear head to actually converge, so the bracketed comparison becomes informative.

**The label space is matched to the statistical power of the trial count, not to the granularity of the underlying ontology.** If we ever scale to ~100k eval trials, 1,854 becomes feasible — but it's a function of n, not a fixed choice.

### What the 27 categories actually are

Canonical THINGS metadata (`ViCCo-Group/THINGS-data/THINGS/Metadata/Concept-specific/category_mat_manual.tsv`) — a 1,854 × 27 binary membership matrix. Single-category concepts get a clean label; multi-category concepts are excluded to keep the probe a K-way (not multi-label) classification problem.

## K-fold cross-validation — why 5

The original probe used a single (80/20) random split. Two problems:

1. **No error bars.** Was the 0.34% top-1 number actually 0.34, or is that a draw from a distribution with mean 0.5 ± 0.2?
2. **Confounded with which trials happen to land in the test set.** If the 20% test sample happened to overrepresent some categories, the number drifts.

5-fold CV reports `top1_tokens_mean` and `top1_tokens_std` across folds. Each trial is in the test set exactly once. ~2.5× the runtime of single-split, error bars become usable.

Why not 10-fold: at the current eval sample size (~3k trials), 10-fold gives 300 test trials per fold and 4× variance contribution per-fold; 5-fold's 600 trials/fold gives more stable per-fold estimates. Diminishing returns past 5 here.

Why not 3-fold: too few estimates to be meaningfully averaged; the std becomes itself high-variance.

`probe_n_folds=1` is preserved as a back-compat path (single split with `_std = 0`).

## Per-RVQ-layer probing (BrainOmni only)

BrainOmni's tokenizer uses a 4-layer **residual** VQ (RVQ):
- Layer 0 quantizes the original latent (coarse approximation)
- Layer 1 quantizes the residual after Layer 0 (refinement)
- Layers 2, 3 keep refining
- Sum of all 4 = full reconstruction

Reconstruction needs all 4 layers. But the **semantic** information is likely concentrated in Layer 0 — it captures the prototypical structure, while later layers add high-frequency detail that's task-irrelevant or even adversarial to a probe (more dimensions to fit, mostly noise from a classification standpoint).

The probe now runs separately on:
- `tokens_all` — sum of all 4 RVQ layers (the default reconstruction features)
- `tokens_rvq0` — Layer 0 only

If `rvq0` ≈ `all`, the coarse layer carries everything; the finer layers are just noise to the classifier. If `rvq0` < `all`, there's task-relevant signal in the residual layers too — interesting, but rarer in practice.

This is a one-line config knob (`EvalConfig.probe_rvq_layers`) — easy to extend to `(0, 1)`, `(2, 3)`, etc. for finer ablations.

## Common failure modes the probe catches

1. **High-MSE drift, low-signal collapse.** Tokenizer reconstructs slow baseline drift (dominates variance, drives MSE down) while losing the small evoked response. Reconstruction §5.1 says "fine"; probe says "tokens at random level." → tokenizer is broken.
2. **All-codes-used, no-signal collapse.** Codebook §5.2 says "full vocab utilization"; probe says "tokens at random level." → codes are used but as a random hash of the input, not as a representation of the task.
3. **Sequence-level redundancy.** Probe says "tokens look fine" but §5.4 says "bigram entropy gap is huge." → tokens carry stimulus info but the sequence is so predictable that 4M's masked-token objective has nothing to learn. The probe alone doesn't catch this; both the probe AND the sequence axis must pass.

## Common failure modes the probe does NOT catch

- **Within-class structure.** Probe asks "can I tell category A from B"; it does NOT ask "can I distinguish two animals of the same superordinate category." For that we'd need a within-category probe or a similarity metric (RSA). Out of scope here.
- **Generalization across subjects/sessions.** Probe trains and tests on pooled trials; a tokenizer overfit to one subject's noise distribution can still pass. Cross-subject CV would catch this — not implemented yet (would need a `kfold_by="subject"` knob).

## What "passing the gate" looks like in numbers

Order-of-magnitude rough thresholds for 27-way + 5-fold + n≈3k eval trials:
- Chance: 1/27 ≈ 3.7% top-1
- Random baseline: ~chance ± fold std
- Raw signal upper bound: depends heavily on preprocessing; published THINGS-MEG decoders hit ~10–15% top-1 for similar coarse categories, but they're not linear
- A passing tokenizer: meaningfully above random, ideally close to raw (within a fold-std)

The exact pass threshold is empirical — we don't know what "good" looks like for our specific preprocessing until we run the new probe. The mu_transform + BrainOmni 3b numbers under the old 1,854-way probe were chance; the new 27-way numbers will be the first interpretable measurement of these tokenizers.

## Pointers to code

- `neural_tokenizers/evaluation/probe.py` — the probe itself
- `neural_tokenizers/evaluation/protocol.py::EvalConfig` — knobs (`probe_n_folds`, `probe_rvq_layers`, `probe_top_k`, …)
- `neural_tokenizers/meg/data.py::SuperordinateMapping` — concept_id → superordinate index
- `neural_tokenizers/meg/modal/modal_download_things_superordinate.py` — one-time downloader
- `neural_tokenizers/meg/brainomni/adapter.py::_decode_rvq_indices` — the `layers=` switch
- `neural_tokenizers/meg/modal/modal_meg_eval.py` — the eval-time wiring
