# MEG tokenization — diagnostics and findings

Brief reference for what we've measured on the BrainOmni 3b tokenizer, how the eval was hardened against false positives/negatives, and the experimental grid that backs the verdict. See [neural_tokenizers/meg/CLAUDE.md §9](../neural_tokenizers/meg/CLAUDE.md) for the production leaderboard; [linear_probe_design.md](linear_probe_design.md) for probe rationale.

## Final conclusion (read this first)

### What the tokens decode — best probe per task, with the raw ceiling alongside

A tokenizer's job is to **preserve** raw signal in a discrete form. The question is "do tokens match raw, both above random?" — not "do tokens beat raw" (rare and not required).

All cells: balanced accuracy mean ± SEM (= fold_std / √5). σ-vs-random uses SEM. Best probe head per task.

| Task (chance) | Experiment | tokens (best probe) | raw (ceiling) | random (floor) | tokens vs raw | tokens σ-vs-random |
|---|---|---|---|---|---|---|
| **Subject** (25%) | 1 — 3b_nonavg | **97.4 ± 0.2** (CNN) | **98.0 ± 0.2** (CNN) | 25.5 ± 0.3 | -0.6pp (preserved) | **~+200σ** |
| **Cat27** (3.70%) | 1 — 3b_nonavg | 3.55 ± 0.24 (CNN) | 6.14 ± 0.20 (Linear unweighted) | 3.70 ± 0.00 | -2.6pp (partial loss) | -0.6σ (chance) |
| **Cat27** (3.70%) | 1.5 — avg eval | 4.09 ± 0.46 (CNN bal_acc) | 4.78 ± 0.66 (CNN bal_acc) | 4.40 ± 0.56 (CNN) | -0.7pp (≈parity) | -0.4σ (chance) |
| **Cat27** (3.70%) | 2 — 3b_avgcross | **4.65 ± 0.59** (CNN bal_acc) | **4.78 ± 0.66** (CNN bal_acc) | 3.65 ± 0.55 | -0.1pp (preserved) | **+1.2σ (borderline)** |
| **Animacy** (50%) | 1 — 3b_nonavg | 52.6 ± 0.72 (Linear rvq0) | 57.9 ± 0.94 (MLP) | 49.9 ± 1.34 | -5pp (partial loss) | +1.5σ (suggestive) |
| **Animacy** (50%) | 1.5 — avg eval | 50.6 ± 0.76 (CNN) | **58.9 ± 2.37** (CNN) | 49.8 ± 0.80 | -8pp (partial loss) | +0.7σ (chance) |
| **Animacy** (50%) | 2 — 3b_avgcross | 52.2 ± 0.76 (CNN) | **58.9 ± 2.37** (CNN) | 50.9 ± 1.12 | -7pp (partial loss) | +1.0σ (borderline) |

### Two clean reads of the table

**1. Raw is the ceiling. The ceiling is low for category-discriminative tasks.**

- Subject ID: raw CNN hits **98%** — head position + sensor profiles are massively decodable from MEG.
- Animacy: raw CNN hits **~58-59%** — Dixen 2024 reported 57-61% cross-subject on THINGS-EEG, so we're at the published ceiling.
- Category27: raw bal_acc tops out at **~5-6%** (linear) or **~4.8%** (CNN). Above 3.7% chance but only by ~1pp absolute. This is intrinsic to the data with linear/CNN probes — not a tokenizer artifact.

**2. Tokens preserve what raw has, modulo a partial loss on animacy.**

- **Subject (Exp 1):** tokens 97.4% vs raw 98.0% → essentially full preservation, both ~+200σ.
- **Cat27 (Exp 2):** tokens 4.65% vs raw 4.78% → full preservation of the small signal raw has; both ~+1.2σ above random.
- **Animacy:** tokens 50-53% vs raw 57-59% → **loses ~5-8pp of the animacy lift raw has**. Notable partial loss across all three experiments.

### Which to ship for 4M

**Experiment 1 (`3b_nonavg`)** remains the answer, but the reasoning now updates:

| Reason | Detail |
|---|---|
| Subject preservation | Only 3b_nonavg keeps subject decodable (97% CNN). Cross-subject averaging in 2 + 1.5 destroys it. |
| Animacy loss is small | 3b_nonavg loses ~5pp on animacy (52.6 vs 57.9); 1.5 / 2 lose ~7-8pp. **1 loses the least.** |
| Cat27 is at the noise floor regardless | All three experiments give tokens 3.5-4.7% bal_acc vs raw ceiling 4.8-6.1%. Choosing among them on this metric is essentially arbitrary; the tokenizer is not what's failing here. |
| Train-eval distribution match | 3b_nonavg sees single trials at both train and inference time. 4M will see single trials in production. |

### The cat27 caveat is structural, not about which experiment

Object category is barely linearly/CNN-decodable from raw THINGS-MEG (~5% bal_acc, ~+1σ above chance). Any reconstruction-only tokenizer is bounded by this ceiling. None of 1 / 1.5 / 2 changes the ceiling because the bottleneck is the **probe + data**, not the tokenizer. To push past it we need either a different probe (deeper architecture, attention) or a different objective (contrastive / supervised) — not a different averaging regime.

### Corrected stats note

Earlier in this doc I called the Exp 1.5 / Exp 2 CNN cat27 cells "<1σ noise floor." That used fold-std as the denominator, which is wrong for comparing means across k=5 folds. SEM-based σ is √5× tighter, putting the Exp 2 CNN cat27 cells at +1.2σ on bal_acc (borderline-significant) — not noise. The substantive ship-3b_nonavg recommendation doesn't change because Exp 1 wins on subject + animacy, but the Exp 2 cat27 cell is honestly "tokens preserve a borderline-significant signal" rather than "tokens at noise."

### Which checkpoint to ship into 4M

**Ship Experiment 1's `3b_nonavg` checkpoint.** Reasoning:

1. **Only 3b_nonavg preserves subject identity** — 97% CNN decode, 62% even with a linear head. Cross-subject averaging in 2 (and inference-time averaging in 1.5) eliminates this by construction. If 4M ever wants to condition on subject or generate subject-specific MEG, only 3b_nonavg's tokens can do it.
2. **None of 1, 1.5, 2 preserve category** — so there is no reason to take 2 over 1; switching costs us subject signal and gains nothing on category. 1.5 is a diagnostic, not a deployable checkpoint (it requires inference-time averaging that doesn't match how 4M will receive single trials in production).
3. **Train-eval distribution matches in 1** — 3b_nonavg is trained on single trials and 4M will see single trials at inference. 1.5 has a train-eval mismatch; 2's "averaged single image" distribution is unrealistic for production single-trial input.

**But ship it with the caveat documented.** 3b_nonavg's tokens are **not category-discriminative**. If 4M needs object-level semantic alignment from MEG (e.g., MEG → image generation or classification), this tokenizer alone won't deliver it.

### What none of the three solves

Object category does not survive any of the three reconstruction-only regimes. The objective is the bottleneck, not the data quality:

- **Averaging the input (1.5)**: improves raw decoding marginally; tokens unchanged.
- **Averaging the training data (2)**: same conclusion; cross-subject averaging additionally eliminates subject signal.
- **Bigger probe (CNN)**: surfaces ~1pp lifts that exist in raw too — not a tokenizer claim.

The next intervention has to change the **objective**, not the input distribution. Concretely:

- **Auxiliary supervised / contrastive loss during BrainOmni finetune** — give the tokenizer a gradient signal that's specifically about discriminability (e.g., InfoNCE pairing same-image trials, or supervised cross-entropy on superordinate labels). This is the most direct path.
- **Stage 2 (Cho2026 / EphysTokenizer)** — different architectural family; revisit if the auxiliary-loss path on BrainOmni fails. Won't fix the objective issue if it's just a different reconstruction-only stack — but it has a smaller compression ratio and may behave differently.

---

## TL;DR (one-paragraph version)

**BrainOmni 3b's tokens preserve SUBJECT identity richly but lose OBJECT CATEGORY structure entirely.** Same tokens, three probe families. CNN → tokens decode subject at **97%** (vs 25% chance, near-perfect) but category at **3.6%** (vs 3.7% chance, at floor). The story isn't "tokens are bad" — it's selective: bits got spent on the signal that dominates variance (subject-level / sensor-level features), the small-amplitude category signal got dropped.

Raw MEG behaves the same way (98% subject vs 4.2% category with CNN) — the asymmetry is intrinsic to the data, not the tokenizer. Tokens preserve roughly the subject-level information present in raw; they don't preserve category info, but raw barely has it linearly decodable either.

## What we are running and why

Three tasks × three probe architectures × four feature sets. We compare two BrainOmni checkpoints to test whether averaging trials at training time changes the picture.

| Axis | Levels |
|---|---|
| **Task** (`probe_label_space`) | `category27` (the gate) / `animacy` (easy 2-way) / `subject` (pipeline sanity) |
| **Probe head** (`probe_classifier`) | `linear` / `mlp` / `cnn` (new — temporal+latent-spatial CNN; see §"CNN probe head") |
| **Feature set** | `raw` (signal upper bound) / `random` (lower bound) / `tokens_all` / `tokens_rvq0` |
| **Checkpoint** | **`3b_nonavg`** (current — finetuned on single trials) / **`3b_avg`** (pending — finetuned on image-averaged trials) |

All evals: full ~10k test split, 5-fold CV, **class-weighted CE**, balanced accuracy as the headline metric. Chance is 1/n_classes for every task (the whole point of balanced acc).

Run pattern, fully parameterized:

```bash
modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni \
    --calibration <checkpoint config.json> \
    --n-test 0 --seed 0 \
    --probe-classifier {linear|mlp|cnn} \
    --probe-label-space {category27|animacy|subject}
```

Results land at `<checkpoint>/evals/eval_ntest=full_s0[_<classifier>][_<label_space>].json`.

---

## CNN probe head — design

The flat MLP probe has no inductive bias for the structure that THINGS-MEG decoding actually exploits (Cichy / Hebart papers: channel-topographic patterns over time). A 2-layer MLP on flattened tokens may simply lack the right shape; per Dixen 2024 even on EEG, EEGNet/Conformer-style temporal convolutions massively outperform flat heads at this SNR.

The new `cnn` head exploits the token tensor's natural structure rather than flattening it:

- **For BrainOmni tokens** of shape `(B, 16 latents, 8 time, D=512 embed)`: a small **2-D CNN** with conv over `(latent × time)` and `D` as channels. Two conv blocks with batch-norm + ReLU + global average pooling → linear classifier. ~150k params.
- **For raw MEG** of shape `(B, 271 channels, 281 time)`: a small **1-D CNN** over time, channels as input dims. Two strided conv blocks → global average pooling → linear. ~50k params.
- **For random tokens**: same shape transform as `tokens` (random codes → `tokens_to_embedding` → CNN). Tests "what does the CNN do with uninformative tokens of the right shape?".
- **For μ-transform-style position-preserving tokens** `(B, 1, channels × time)`: reshape to `(B, channels, time)` then use the 1-D CNN — equivalent to applying the raw-MEG CNN to the dequantized signal.

Rationale: small enough to not overfit on ~4k training trials per fold, while giving the probe the same inductive bias (temporal smoothness + latent-position locality) that real MEG decoders use.

---

## Experiment 1 — BrainOmni 3b_nonavg (current, single-trial finetune)

Chance: 27-way = 3.70%, animacy 2-way = 50%, subject 4-way = 25%.

### 1a — Category27 (the gate). Chance = 3.70% (top-1); 5/27 ≈ 18.5% (top-5 under uniform classes — the `random` column is the true imbalanced floor).

Cell format: `top-1 / top-5` (top-1 is bal-acc for "weighted" rows, raw top-1 for the unweighted row).

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, unweighted | 3.76% ± 0.07 / — | 3.71% ± 0.01 / — | **6.14% ± 0.44** / — | 3.70% ± 0.00 / — |
| Linear, weighted | 3.35% ± 0.91 / 16.85% ± 0.80 | 3.42% ± 0.38 / 16.53% ± 0.82 | **5.16% ± 0.21 / 32.11% ± 2.28** | 3.64% ± 0.73 / 18.96% ± 1.15 |
| MLP, weighted | 3.18% ± 0.97 / 15.44% ± 1.25 | 2.94% ± 0.78 / 15.22% ± 0.78 | **4.31% ± 0.49 / 31.38% ± 1.90** | 3.09% ± 0.82 / 20.37% ± 1.00 |
| **CNN, weighted** | 3.55% ± 0.54 / **28.84% ± 1.13** | 3.77% ± 0.53 / **29.73% ± 1.79** | **4.19% ± 0.87** / 32.27% ± 1.46 | 3.61% ± 0.52 / 27.64% ± 1.50 |

All four probe regimes agree on the headline: tokens at chance under top-1 / bal-acc, raw between 4.2% and 6.1% (small but real). CNN's inductive bias does not rescue tokens on the strict metric — the fine-grained category info isn't there.

**One footnote-worthy detail on top-5.** Under the CNN probe specifically, tokens_all top-5 (28.8%) and tokens_rvq0 top-5 (29.7%) sit ~1–2pp above the random top-5 floor (27.6%), while linear/MLP put tokens *below* random on top-5. This is the only knob in the Exp 1 grid where tokens beat random anywhere. Most likely interpretation: CNN's temporal-spatial bias surfaces a faint *coarse* category neighborhood in the tokens — enough to bump top-5 by a percentage point, but well within fold variance (CNN top-5 std ≈ 1.5pp on random) and nowhere near raw's lift (+5pp). Worth noting; not worth re-deciding the verdict on. Unweighted top-5 unavailable: the row's data is from an older run whose JSON is no longer extant; we kept its top-1 numbers in the table but cannot reconstruct its top-5.

> "—" means the underlying JSON is no longer available (older class-unweighted probe run that has since been overwritten — only its top-1 numbers survived in the table). Re-running would cost ~$0.50; we haven't because nothing hinges on it.

### 1b — Subject ID (pipeline sanity). Chance = 25%. (Top-5 omitted: mechanically 100% since n_classes=4 < k=5.)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted | 62.6% ± 1.0 | 45.1% ± 0.7 | 93.9% ± 0.9 | 25.2% ± 0.9 |
| MLP, weighted | 69.3% ± 1.0 | 52.2% ± 1.1 | 95.4% ± 0.4 | 25.4% ± 1.3 |
| **CNN, weighted** | **97.4% ± 0.4** | **92.1% ± 0.5** | **98.0% ± 0.5** | 25.5% ± 0.7 |

CNN nearly closes the raw→tokens gap. **97% subject decoding from tokens** says the tokens carry rich structural/spatial info; the §5.3 linear gate was simply the wrong probe for that signal. Each architecture step (linear → MLP → CNN) buys more — clean monotone improvement, exactly what you expect when more inductive bias surfaces existing signal.

### 1c — Animacy (easy 2-way, sensitivity check). Chance = 50%. (Top-5 omitted: mechanically 100% since n_classes=2 < k=5.)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted | 51.1% ± 1.8 | 52.6% ± 1.6 | **57.8% ± 2.5** | 50.0% ± 2.2 |
| MLP, weighted | 50.8% ± 2.0 | 51.4% ± 0.8 | **57.9% ± 2.1** | 49.9% ± 3.0 |
| **CNN, weighted** | 51.6% ± 2.0 | 51.2% ± 1.3 | 52.1% ± 1.3 | 49.5% ± 0.6 |

Tokens at chance across all classifiers. Raw at 52-58% (consistent with Dixen 2024 cross-subject ceiling of 57-61% for binary living/non-living). CNN slightly hurts raw on animacy — likely 2-class label-imbalance noise interacting with CNN regularization; not investigated further since tokens are at chance regardless.

### 1d — Model-free retrieval (§5.5), 3b_nonavg

Zero trainable parameters. Cosine on `tokens_to_embedding` features, Jaccard on token sets, both reported. Each row is one task; retrieval numbers are independent of the classifier head (same JSON across linear/MLP/CNN runs of the same task).

| Task (n) | tokens (cosine) prec@1 / @5 | tokens (Jaccard) prec@1 / @5 | raw prec@1 / @5 | random prec@1 / @5 |
|---|---|---|---|---|
| Category27 (~5032, chance 3.70%) | 3.71% / 3.59% | 3.38% / 3.81% | 3.76% / 3.94% | 4.11% / 3.54% |
| Animacy (~5032, chance 50%) | 50.5% / 50.3% | 51.1% / 50.5% | 50.2% / 50.6% | 49.7% / 49.3% |
| Subject (~10k, chance 25%) | **50.3% / 47.5%** | **37.7% / 36.7%** | **65.9% / 61.5%** | 24.7% / 24.9% |

Retrieval confirms the classifier-probe picture, with zero trainable parameters:

- **Subject** lifts massively above chance in token retrieval (cosine 50.3%, Jaccard 37.7%, both vs 25% chance) — the token-set geometry directly clusters trials by subject. Same conclusion as the §5.3 probe but achieved without any training.
- **Category and animacy** retrieval are at chance across all feature sets. Notably **raw retrieval is also at chance** for these tasks at n≈5k, even though the linear probe finds ~6% bal_acc on raw — telling us k-NN is too weak to surface the small linear signal that's still there, but the *relative* tokens-vs-raw comparison is preserved (both at floor).

The Subject-task gap between retrieval and the trained classifier (50% retrieval vs 97% CNN) is what we expect: retrieval reads off whatever's natively close in feature space; a trained probe additionally rotates/scales features to amplify the discriminative directions. CNN's 97% on subject is real signal *plus* learned exploitation; retrieval's 50% on subject is real signal *alone*. The fact that retrieval is above chance is the key — confirms the signal *is* in the geometry, the classifier isn't fabricating it.

---

## Experiment 1.5 — BrainOmni 3b_nonavg, image-averaged INPUT (diagnostic, no retrain)

The cheap diagnostic from `meg/CLAUDE.md` §9: take the **existing 3b_nonavg** checkpoint and feed it **image-averaged trials at eval time**. No retraining — just a different input distribution at inference. Answers: is per-trial noise the bottleneck for category, or did the reconstruction objective itself drop the signal?

Eval pipeline: full ~10k test split (10,036 trials) → average by `image_id` across subjects + within-subject reps → 2,245 unique averaged signals → tokenizer forward → 5-fold CV probe. Codebook stats unchanged (same 512-code vocab, perplexity ~493).

> **A note on reconstruction MSE under averaging.** Time-domain MSE drops 1.07 → 0.29, but this is **not** a 3.6× improvement in the tokenizer's reconstruction quality — it's mostly the **target's noise floor falling**. BrainOmni's preprocess per-trial-per-channel z-scores both single and averaged inputs to unit variance, but the *composition* of that variance differs: in a single z-scored trial ~90% of variance is per-trial noise; in an N-rep average that fraction shrinks by 1/N before z-scoring. The bottleneck drops noise either way; with less noise in the target there's less for the tokenizer to "miss." The honest cross-regime metric is **per-channel Pearson r**, which barely moves: 0.870 (Exp 1) → 0.859 (Exp 1.5). So the right reading is "averaging gives the *probe* a cleaner target," not "the tokenizer reconstructs better" — and the probe results below confirm that the *tokens* see none of that benefit. (General rule: MSE is comparable across checkpoints under fixed input regime, NOT across input regimes or across tokenizer families.)

Run pattern (already wired via `--averaging cross_subject`):

```bash
modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni \
    --calibration neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b/config.json \
    --n-test 0 --seed 0 \
    --probe-classifier {linear|mlp|cnn} \
    --probe-label-space {category27|animacy} \
    --averaging cross_subject
```

Subject ID is **omitted by design** — cross-subject averaging eliminates the subject variable; the script rejects `probe_label_space=subject` when averaging is active.

### 1.5a — Category27 (averaged input). Chance = 3.70% (top-1); 5/27 ≈ 18.5% (top-5 uniform). n=1,126 averaged trials with valid labels.

Cell format: `top-1 / top-5`. Unlike Exp 1, **both** rows here come from the same class-weighted training run — the "Linear, unweighted" row reports the raw top-1 metric, the "Linear, weighted" row reports balanced accuracy. They share the same top-5 column by construction.

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, unweighted | 3.55% ± 0.89 / 18.30% ± 3.75 | 3.11% ± 1.41 / 18.21% ± 4.85 | **10.30% ± 1.36 / 35.88% ± 1.77** | 4.44% ± 1.86 / 21.94% ± 3.94 |
| Linear, weighted | 3.35% ± 0.90 / 18.30% ± 3.75 | 3.88% ± 2.74 / 18.21% ± 4.85 | **6.01% ± 2.37 / 35.88% ± 1.77** | 4.02% ± 2.24 / 21.94% ± 3.94 |
| MLP, weighted | 4.44% ± 1.21 / 17.67% ± 1.95 | 4.01% ± 2.55 / 15.45% ± 2.32 | **4.51% ± 0.78** / 30.01% ± 3.17 | 4.17% ± 1.55 / **33.13% ± 1.51** |
| **CNN, weighted** | 4.09% ± 1.03 / **30.82% ± 3.77** | 4.05% ± 0.66 / **29.58% ± 3.43** | **4.78% ± 1.47** / 30.46% ± 2.29 | 4.40% ± 1.25 / 28.41% ± 3.63 |

Tokens still at chance across all four probe regimes under top-1 / bal-acc. Raw lifts more under averaging than it did in Experiment 1 (linear unweighted top-1: 6.14% → **10.30%**; raw top-5: 32.11% → 35.88%) — confirming the SNR boost reaches raw features, but **the tokens do not benefit at the strict metric**. tokens_all and tokens_rvq0 sit inside the random bracket on every weighted row.

Two top-5 oddities worth noting (neither changes the verdict):
- **CNN tokens beat random on top-5 again** (tokens_all 30.82% vs random 28.41%; tokens_rvq0 29.58% vs random 28.41%) — same direction as Exp 1.4 [[1a]], slightly larger gap, but fold std (~3.7pp) still covers it. Faint coarse-category signal, not a contradiction of the chance verdict.
- **MLP raw top-5 (30.01%) is *below* MLP random top-5 (33.13%)** on the averaged set. Most plausible read: with only 1,126 averaged samples and 5-fold CV (~225 per fold), the MLP overfits the random features into a confident top-5-most-common-classes prediction; on raw features it spreads probability mass too widely. A probe-architecture artifact, not a tokenizer claim.

### 1.5b — Subject ID. **N/A** — cross-subject averaging eliminates the subject variable by construction.

### 1.5c — Animacy (averaged input). Chance = 50%. (Top-5 omitted: mechanically 100% since n_classes=2 < k=5.)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, unweighted | 55.3% ± 2.9 | 51.1% ± 2.8 | **79.5% ± 2.2** | 64.5% ± 4.1 |
| Linear, weighted | 49.3% ± 1.4 | 45.0% ± 3.7 | **61.1% ± 3.4** | 48.2% ± 4.3 |
| MLP, weighted | 50.3% ± 1.8 | 46.8% ± 2.8 | **63.8% ± 2.1** | 50.8% ± 3.1 |
| **CNN, weighted** | 50.6% ± 1.7 | 51.4% ± 2.5 | **58.9% ± 5.3** | 49.8% ± 1.8 |

Same pattern: tokens sit at random under every weighted probe; raw lifts to 59–80%. Unweighted top-1 numbers inflate because the averaged set's animacy distribution is imbalanced (more inanimate images survive the superordinate filter) — class-weighted balanced accuracy is the honest column.

### 1.5d — Model-free retrieval (§5.5), 3b_nonavg with averaged input

Same retrieval probe as [[1d]] but on the averaged-input regime. Zero trainable parameters; cosine on `tokens_to_embedding` features, Jaccard on token sets. Subject task N/A under cross-subject averaging.

| Task (n) | tokens (cosine) prec@1 / @5 | tokens (Jaccard) prec@1 / @5 | raw prec@1 / @5 | random prec@1 / @5 |
|---|---|---|---|---|
| Category27 (~1126, chance 3.70%) | 3.25% / 3.28% | 4.47% / 3.91% | 4.24% / 4.37% | 3.72% / 3.84% |
| Animacy (~2245, chance 50%) | 48.4% / 49.3% | 50.5% / 50.4% | 50.7% / 51.4% | 51.2% / 49.9% |

**Tokens retrieval is at chance on both tasks under averaged input** — same regime as 3b_nonavg single-trial [[1d]] and as the averaged-trained 3b_avgcross [[2d]]. Bag-of-tokens geometry doesn't cluster averaged trials by category or animacy any better than randomly-drawn token bags do.

Raw retrieval is *also* at chance on both tasks at this n (1,126 / 2,245). Same caveat as 2d: k-NN with cosine similarity on the 76k-dim flattened raw signal is too weak to surface the small linear signal that the §5.3 classifier still finds at 6.0% (category) and 61.1% (animacy). So retrieval is a strong "tokens at floor" signal but not a strict ceiling proof for raw at this sample count.

**What retrieval adds beyond §5.3 here.** Per the gate from the user's framing: if retrieval at chance, no classifier will work. Tokens are at chance under retrieval on category and animacy in BOTH input regimes (single-trial 1d and averaged 1.5d) — independent confirmation that the §5.3 MLP/CNN failure on those tasks is "the signal isn't in the geometry," not "the probe was too weak." Subject is the control: there, tokens lift to 50.3% prec@1 (vs 25% chance) in 1d, confirming retrieval *can* find token signal when it exists.

### Headline reading of Experiment 1.5

**Averaging the input does not rescue category/animacy in 3b_nonavg's tokens.** Reconstruction quality improves (MSE −73%), the raw-feature probe improves correspondingly, but the tokens still carry no decodable category or animacy signal at any probe complexity. This rules out "single-trial noise was the bottleneck" as the explanation — the bottleneck is the tokenizer's **reconstruction objective**, which budgets bits to variance (subject-level / sensor-level) and not to discrimination. Per the decision tree in `meg/CLAUDE.md` §9, this argues against spending a finetune run on averaged trials (Experiment 2 below) and toward the architectural intervention: contrastive / supervised finetune, or moving to Stage 2 (Cho2026 / EphysTokenizer).

JSONs: `brainomni/runs/V512_rvq4_win512_sf256_3b/evals/eval_ntest=full_s0[_<clf>][_<label>]_avgcross_subject.json`.

---

## Experiment 2 — BrainOmni 3b_avgcross (finetuned on cross-subject-averaged trials)

**Checkpoint:** `V512_rvq4_win512_sf256_3b_avgcross` (`adapt` mode, lr 3e-5, batch 32, 10 epochs, patience-based early stopping with `patience=2 min_epochs=3`). Train: 17,959 averaged trials (was 88,340 single trials); val: 2,245 (was 9,816). Final epoch: 10 (no early-stop; train and val tracking closely, best val 1.4691 at epoch 9 vs 3b_nonavg's 1.401).

**Eval pipeline:** same `--averaging cross_subject` flag → ~10k test trials → 2,245 averaged signals → tokenizer (`3b_avgcross` weights) → 5-fold CV probe + §5.5 retrieval. n=1,126 with valid superordinate labels for category27; n=2,245 for animacy.

**Probe coverage so far:** linear (cheapest), CNN (right inductive bias for temporal-spatial structure). MLP intentionally skipped — the §5.5 retrieval probe is at chance for tokens on both tasks (see §2d below), and CNN was added explicitly to test whether the right inductive bias can find faint signal that retrieval misses. If retrieval fails AND CNN fails, MLP between them has no plausible mechanism to find anything new.

### 2a — Category27 (averaged input + averaged-trained tokenizer). Chance = 3.70%.

Cell format: `top-1 / top-5 / bal_acc` (mean ± std across 5 folds).

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted | 3.37±1.68 / 17.58±3.15 / 2.88±1.80 | 3.11±0.71 / 17.50±1.85 / 3.24±1.32 | **10.30±1.36 / 35.88±1.77 / 6.01±2.37** | 4.44±1.86 / 21.94±3.94 / 4.02±2.24 |
| MLP, weighted | not run — retrieval rules out | not run | not run | not run |
| **CNN, weighted** | 9.32±1.85 / 30.64±3.88 / **4.65±1.31** | 8.52±2.17 / 30.20±3.58 / 4.07±0.82 | 9.06±1.38 / 30.46±2.29 / 4.78±1.47 | 7.73±2.50 / 27.89±1.98 / 3.65±1.24 |

### 2b — Subject ID. **N/A** — cross-subject averaging eliminates the subject variable by construction.

### 2c — Animacy (averaged input + averaged-trained tokenizer). Chance = 50%. (Top-5 omitted: mechanically 100% for 2-class.)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted (top-1) | 54.1±4.4 | 54.2±4.2 | **79.5±2.2** | 64.5±4.1 |
| Linear, weighted (bal_acc) | 48.5±4.6 | 50.9±7.3 | **61.1±3.4** | 48.2±4.3 |
| MLP, weighted | not run — retrieval rules out | not run | not run | not run |
| **CNN, weighted (top-1)** | 81.4±2.1 | 78.5±1.2 | 76.7±3.8 | 83.0±1.4 |
| **CNN, weighted (bal_acc)** | **52.2±1.7** | 51.4±0.3 | **58.9±5.3** | 50.9±2.5 |

### 2d — Model-free retrieval (§5.5), 3b_avgcross

Bag-of-tokens / cosine on `tokens_to_embedding` features / Jaccard on token sets. Zero trainable parameters; tests **feature geometry directly**.

| Task | tokens (cosine) prec@1 / @5 | tokens (Jaccard) prec@1 / @5 | raw prec@1 / @5 | random prec@1 / @5 |
|---|---|---|---|---|
| Category27 (chance 3.70%) | 3.98% / 3.72% | 4.51% / — | **4.24% / 4.37%** | 3.72% / 3.84% |
| Animacy (chance 50%) | 48.6% / 50.7% | 48.6% / 49.4% | 50.7% / 51.4% | 51.2% / 49.9% |

**Tokens retrieval is at chance on both tasks.** Same regime as 3b_nonavg's retrieval, same regime as 1.5's retrieval. The token-set geometry separates trials at the rate of a random shuffle.

Raw retrieval is *also* at chance on both tasks under cross-subject averaging — which is informative: it tells us **the model-free probe is too weak on this n** (1,126 / 2,245 samples after dropping unmapped). Compare against the classifier numbers: linear raw lifts to 6.01% (category) and 61.1% (animacy) — so signal *is* in raw, the linear head with class-weighted CE picks it up where k-NN cannot. This caveats the retrieval test: it's a strong signal that tokens are at floor, but not a strict ceiling proof for raw at this sample count.

### Headline reading of Experiment 2

bal_acc means; chance = 3.70% for category27, 50% for animacy.

| Task | tokens (linear) | tokens (CNN) | tokens (retrieval) | raw (best probe) | Decodable from tokens? |
|---|---|---|---|---|---|
| Category27 | 2.9% (at chance) | 4.65 ± 1.31% (1pp lift, ~1σ) | 4.0% prec@1 (at chance) | 6.0% (linear, above chance) | ❌ marginal — within fold std |
| Animacy | 48.5% (at chance) | **52.2 ± 1.7%** (2pp lift) | 48.6% prec@1 (at chance) | 61.1% (linear, above chance) | ⚠️ small lift — see below |

**Finetuning on averaged trials does not meaningfully rescue category or animacy in the tokens.** Two small lifts under CNN do appear (cat 4.65% vs random 3.65%; animacy 52.2% vs random 50.9%) but both are within ~1σ of fold variance and don't change the verdict — they're consistent with the small top-5 footnote we already saw in Exp 1a. Compare against subject in Exp 1 where CNN tokens hit **97% vs 25% chance** when real signal exists; the contrast tells you what a *real* CNN-rescued signal looks like in this harness vs the noise-floor wiggle we see here.

Combined with Experiment 1.5 (same diagnosis from a no-retrain diagnostic), the evidence is consistent and convergent: the **reconstruction objective is the bottleneck**, not the per-trial noise level of the input. The averaged-input regime (1.5) and the averaged-training regime (2) both fail to make tokens carry category info — even when paired with the right-inductive-bias CNN probe.

Decision: do not iterate further on reconstruction-only finetuning. Move to one of:

1. **Auxiliary supervised / contrastive loss during BrainOmni finetune** — gives the tokenizer a gradient signal that's specifically about discrimination, not just variance.
2. **Stage 2 (Cho2026 / EphysTokenizer adapter)** — different architectural family; revisit if (1) doesn't work.

JSONs: `brainomni/runs/V512_rvq4_win512_sf256_3b_avgcross/evals/eval_ntest=full_s0[_<clf>][_<label>]_avgcross_subject.json`.

---

### Headline reading of Experiment 1

| Task | Best on tokens | Best on raw | Tokens vs raw | Decodable from tokens? |
|---|---|---|---|---|
| Subject (4-way) | **97.4% (CNN)** | 98.0% (CNN) | -0.6pp | ✅ **YES — near-perfect** |
| Animacy (2-way) | 52.6% (Linear rvq0) | 57.9% (MLP) | -5.3pp | ⚠️ marginal (within fold noise) |
| Category27 (27-way) | 3.8% (CNN rvq0) | 6.1% (Linear) | -2.3pp | ❌ **NO — at chance** |

**The selective failure is the key result.** The tokenizer preserves whatever signal dominates variance (subject identity is large; category-discriminative spatial-temporal microstructure is small) and drops what doesn't fit in the 1,152-bit-per-trial budget. CNN proves it isn't a probe-architecture limitation — when signal exists in the tokens, CNN finds it (97% subject); when it doesn't (category), no inductive bias rescues it.

This is the most informed verdict yet on whether 3b_nonavg's tokens are 4M-ready: they are if 4M needs subject/recording-level features; they are not if 4M needs to discriminate object category from this modality alone.

## The harness — five axes (eval contract)

| Axis | What it answers | Source |
|---|---|---|
| §5.1 reconstruction | Does encode→decode round-trip preserve the waveform? | `evaluation/reconstruction.py` |
| §5.2 codebook | Is the vocab used (no dead codes / collapse)? | `evaluation/codebook.py` |
| §5.3 linear probe | Are tokens **linearly decodable** to a class label? | `evaluation/probe.py` |
| §5.4 sequence | Is the token *sequence* learnable (entropy gap, runs)? | `evaluation/sequence.py` |
| §5.5 retrieval | Same as §5.3 but **model-free** — does feature geometry separate classes? | `evaluation/retrieval.py` |

## §5.3 probe hardening (timeline)

| Iteration | What we added | Why |
|---|---|---|
| v0 | Linear head, top-1 + top-5, single split | Baseline §5.3 contract |
| v1 | Label space → THINGS **27-way superordinate** (was 1854-way concept) | 2.6 trials/class is statistically dead; 27-way → ~185/class |
| v2 | **5-fold CV**, mean ± std | Single-split was un-error-barred |
| v3 | **Class-weighted CE** | Unweighted CE on imbalanced labels collapses to majority prediction |
| v4 | **Balanced accuracy** as the headline metric | Top-1 floor with imbalance is `max(class_freq)`, not 1/n_classes |
| v5 | **2-layer MLP** option (`probe_classifier="mlp"`) | Catches "info is there but nonlinear" |
| v6 | **Per-RVQ-layer probing** (`tokens_rvq0` vs `tokens_all`) | Asks whether coarse layer alone carries info |
| v7 | **Position-preserving `tokens_to_embedding`** for μ-transform | BoC was discarding the position μ-transform faithfully preserves |
| v8 | **3 label spaces** (`category27`/`animacy`/`subject`) | Subject is pipeline sanity check (Dixen 2024 trick); animacy is sensitivity check |
| v9 | **§5.5 retrieval axis** (cosine prec@K + Jaccard) | Zero-trainable probe: separates "tokens bad" from "classifier bad" |
| **v10** | **CNN probe head** (`probe_classifier="cnn"`) | Right inductive bias for temporal-spatial structure; Dixen 2024 shows architecture matters at this SNR |

## What broke when, and how we caught it

| Sanity flag | Cause | Diagnostic |
|---|---|---|
| `top1_random = 12.4%` on 27-way labels | Unweighted CE collapses to majority prediction | Class-weighted CE drops top1_random to 3.7% |
| `raw < tokens` under top-1 | 76k-dim raw features overfit on ~4k training samples | Balanced accuracy correctly puts raw above tokens |
| `bal_acc_tokens ≈ random` everywhere | "Tokens bad" or "probe broken"? | Subject-ID probe shows tokens at 62%, random at 25% → probe works, tokens lose category specifically |
| μ-transform tokens at chance despite near-lossless reconstruction | BoC featurization discards position | Position-preserving `tokens_to_embedding` |
| "Reconstruction-only" framing for BrainOmni | Misread the trainer | Actual loss has 5 terms: time + per-channel-correlation + commitment + FFT amplitude + 0.5 × FFT phase |

## Testing strategy

| Layer | Examples | File |
|---|---|---|
| Unit — label mappings | `test_super_mapping_encode_drops_multi_membership_concepts`, `test_animacy_encode_binary_output` | `meg/test_data.py` |
| Unit — probe internals | `test_class_weights_match_sklearn_formula`, `test_probe_class_weighted_prevents_majority_collapse`, `test_build_head_mlp_2layer_with_dropout`, `test_build_head_cnn_*` (pending) | `test_tokenizer.py` |
| Unit — featurization | `test_tokens_to_embedding_preserves_position_as_bin_centers` (μ-transform), `test_decode_rvq_indices_layer_selection` (BrainOmni) | `meg/mu_transform/test_mu_transform.py`, `meg/brainomni/test_brainomni.py` |
| Unit — retrieval | `test_retrieval_random_at_balanced_chance`, `test_retrieval_jaccard_skipped_for_saturated_codes` | `test_retrieval.py` |
| Integration | `test_evaluate_runs_all_axes` (full harness on stub) | `test_tokenizer.py` |
| Integration (Modal) | `modal_meg_eval.py::run` with `--probe-classifier` and `--probe-label-space` flags | `meg/modal/modal_meg_eval.py` |

## Cross-checks backing every verdict

1. **Pipeline sanity** — subject decodes at 62% (vs 25% chance) → pipeline works.
2. **Multiple probe families** — linear, MLP, model-free retrieval (and now CNN) all consulted.
3. **Multiple training regimes** — unweighted, class-weighted, both metric variants agreed before we moved on.
4. **Cross-tokenizer agreement** — both μ-transform and BrainOmni fail §5.3 on category; both pass §5.1 reconstruction. Common factor: large compression ratio + global reconstruction objective.
5. **Bracketing brackets stable** — raw and random brackets are stable across configurations.

## Open follow-ups

| Item | Why |
|---|---|
| Finetune BrainOmni on image-averaged trials → 3b_avg, repeat the §1 grid | Tests whether per-trial noise was the bottleneck for category. The CNN result above shows the tokens have *room* for subject-level signal but spent the bit budget on subject not category; averaging may shift the budget. (Experiment 2 in this doc.) |
| Re-run μ-transform with new position-preserving `tokens_to_embedding` × the §1 grid | μ-transform is near-lossless. Expected: tokens ≈ raw for all three tasks. Validates the position-preserving featurization on real data. |
| Document corrected 5-term loss (including FFT) in `meg/CLAUDE.md §9` | Old text said "reconstruction-only" — wrong. The loss already has FFT amplitude + phase. |
| Consider a category-aware finetune objective (contrastive / supervised) | Reconstruction loss budgets bits to variance, not to discrimination. If averaged trials don't fix category, this is the architectural intervention. |

## Pointers

- Production leaderboard: [`neural_tokenizers/meg/CLAUDE.md §9`](../neural_tokenizers/meg/CLAUDE.md)
- Probe design rationale: [`linear_probe_design.md`](linear_probe_design.md)
- 4M-modality plan: [`4m_neural_modality_design.md`](4m_neural_modality_design.md)
- Probe source: [`neural_tokenizers/evaluation/probe.py`](../neural_tokenizers/evaluation/probe.py), [`evaluation/retrieval.py`](../neural_tokenizers/evaluation/retrieval.py)
- Eval dispatcher: [`neural_tokenizers/meg/modal/modal_meg_eval.py`](../neural_tokenizers/meg/modal/modal_meg_eval.py)
