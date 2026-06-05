# MEG tokenization v2 — F2-corrected re-runs

This file has the F2-corrected numbers for every MEG eval config. For analysis and verdicts see [meg_tokenization.md](meg_tokenization.md). All JSONs live in `evals/v2/` subfolders; originals untouched.

## What changed (F2 fix)

`evaluation/probe.py` and `evaluation/retrieval.py` had a dimensionality mismatch in the random baseline: when a tokenizer has `tokens_to_embedding`, the random baseline was using a bag-of-codes histogram (V-dimensional) while the token baseline was using mean-pooled embeddings (D-dimensional). Fixed: random now uses `_embed_and_mean` (same D-dimensional path as tokens).

**Effect:** random column numbers shift; verdicts unchanged (subject at 97%, category/animacy conclusions are large-margin).

---

## Summary leaderboard (v2, bal_acc mean ± SEM across 5 folds)

| Task (chance) | Experiment | tokens best probe | raw ceiling | random floor | Δ vs random |
|---|---|---|---|---|---|
| **Subject** (25%) | 1 — 3b_nonavg | **97.4 ± 0.2%** (CNN all) | **98.0 ± 0.2%** (CNN) | 25.5 ± 0.3% | ~+200σ |
| **Cat27** (3.70%) | 1 — 3b_nonavg | 3.6 ± 0.2% (CNN rvq0) | 5.2 ± 0.1% (Linear) | 3.9 ± 0.2% | −0.8σ (chance) |
| **Cat27** (3.70%) | 1.5 — avg eval | 4.4 ± 0.5% (MLP all) | 6.0 ± 1.1% (Linear) | 3.1 ± 0.4% | +1.0σ (borderline) |
| **Cat27** (3.70%) | 2 — 3b_avgcross | **4.7 ± 0.6%** (CNN all) | **4.8 ± 0.7%** (CNN) | 3.7 ± 0.6% | **+1.2σ (borderline)** |
| **Animacy** (50%) | 1 — 3b_nonavg | 52.6 ± 0.7% (Linear rvq0) | 57.9 ± 0.9% (MLP) | 50.7 ± 1.0% | +1.6σ (suggestive) |
| **Animacy** (50%) | 1.5 — avg eval | 50.6 ± 0.8% (CNN all) | 58.9 ± 2.4% (CNN) | 49.8 ± 0.8% | +0.7σ (chance) |
| **Animacy** (50%) | 2 — 3b_avgcross | 52.2 ± 0.7% (CNN all) | 58.9 ± 2.4% (CNN) | 50.9 ± 1.1% | +1.0σ (borderline) |
| **Cat27** (3.70%) | mu_transform | 5.0 ± 0.3% | 5.2 ± 0.1% (raw) | 3.8 ± 0.6% | +1.6σ |

Verdicts from v1 hold. The random floor shifted (was inflated by BoC mismatch in v1); tokens numbers are unchanged; the Exp 2 cat27 result remains borderline.

---

## Experiment 1 — BrainOmni 3b, non-averaged single trials

All cells: bal_acc mean ± fold_std. Chance: cat27 = 3.70%, animacy = 50%, subject = 25%.

### Cat27

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 3.4 ± 0.91 | 3.4 ± 0.38 | **5.2 ± 0.21** | 3.9 ± 0.46 |
| MLP | 3.2 ± 0.97 | 2.9 ± 0.78 | **4.3 ± 0.49** | 3.6 ± 0.68 |
| CNN | 3.6 ± 0.54 | 3.8 ± 0.53 | **4.2 ± 0.87** | 3.6 ± 0.52 |

### Animacy

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 51.1 ± 1.78 | **52.6 ± 1.57** | **57.8 ± 2.46** | 50.7 ± 2.15 |
| MLP | 50.8 ± 1.97 | 51.4 ± 0.82 | **57.9 ± 2.06** | 51.3 ± 1.64 |
| CNN | 51.6 ± 1.97 | 51.2 ± 1.33 | 52.1 ± 1.33 | 49.5 ± 0.61 |

### Subject

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 62.6 ± 0.97 | 45.1 ± 0.72 | **93.8 ± 0.90** | 25.2 ± 0.60 |
| MLP | 69.3 ± 1.00 | 52.2 ± 1.06 | **95.4 ± 0.41** | 25.3 ± 1.22 |
| CNN | **97.4 ± 0.35** | 92.1 ± 0.51 | **98.0 ± 0.51** | 25.5 ± 0.73 |

---

## Experiment 1.5 — BrainOmni 3b, averaged input at eval time (no retrain)

Inference-time cross-subject averaging via `--averaging cross_subject`. Subject task N/A (eliminated by averaging). n≈1,126 (cat27) / 2,245 (animacy).

### Cat27

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 3.3 ± 0.90 | 3.9 ± 2.74 | **6.0 ± 2.37** | 2.4 ± 0.84 |
| MLP | **4.4 ± 1.21** | 4.0 ± 2.55 | **4.5 ± 0.78** | 3.1 ± 1.00 |
| CNN | 4.1 ± 1.03 | 4.0 ± 0.66 | **4.8 ± 1.47** | 4.4 ± 1.25 |

### Animacy

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 49.3 ± 1.38 | 45.0 ± 3.72 | **61.1 ± 3.37** | 48.6 ± 5.25 |
| MLP | 50.3 ± 1.76 | 46.8 ± 2.85 | **63.8 ± 2.14** | 47.6 ± 6.42 |
| CNN | **50.6 ± 1.74** | 51.4 ± 2.45 | **58.9 ± 5.34** | 49.8 ± 1.78 |

---

## Experiment 2 — BrainOmni 3b_avgcross (finetuned on averaged trials)

Checkpoint: `V512_rvq4_win512_sf256_3b_avgcross`. MLP intentionally not run (retrieval at chance for tokens on both tasks — no plausible mechanism for MLP to find signal).

### Cat27

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 2.9 ± 1.80 | 3.2 ± 1.32 | **6.0 ± 2.37** | 1.5 ± 0.58 |
| CNN | **4.7 ± 1.31** | 4.1 ± 0.82 | **4.8 ± 1.47** | 3.7 ± 1.24 |

### Animacy

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear | 48.5 ± 4.58 | 50.9 ± 7.31 | **61.1 ± 3.37** | 47.0 ± 2.69 |
| CNN | **52.2 ± 1.67** | 51.4 ± 0.33 | **58.9 ± 5.34** | 50.9 ± 2.50 |

---

## mu_transform — V256 per-channel calibration

Category27 only (single eval run). Probe fields use top-1 key names (mu_transform outputs flat integer tokens, no RVQ layers).

| Metric | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| top1_mean | 8.6 ± 0.85 | 8.6 ± 0.85 | 9.2 ± 0.42 | 3.4 ± 0.88 |
| bal_acc_mean | 5.0 ± 0.58 | 5.0 ± 0.58 | 5.2 ± 0.21 | 3.8 ± 1.30 |

Note: mu_transform is near-lossless (tokens ≈ raw on both top1 and bal_acc). The jump from ~3.6% (old full-dataset, non-v2) to 5.0% (v2) came primarily from the **position-preserving fix**: the old featurization used bag-of-codes (an unordered histogram of which codes appeared), which discards which channel and timepoint each code came from. The v2 fix uses position-preserving `tokens_to_embedding`, restoring that structure. Without the fix, full-dataset tokens sit at ~3.57% bal_acc (chance); with it, 4.96% ≈ raw's 5.16%. Underfitting at n=3000 is also real (76k features, 2,400 training samples) but secondary -- the position fix did the heavy lifting.

---

## Run commands (for reference)

### Experiment 1 — BrainOmni 3b, non-averaged

```bash
CALIB=neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b/config.json
EVALS_V2=neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b/evals/v2

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier linear --probe-label-space animacy \
    --output $EVALS_V2/eval_ntest=full_s0_animacy.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier linear --probe-label-space subject \
    --output $EVALS_V2/eval_ntest=full_s0_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier mlp --probe-label-space category27 \
    --output $EVALS_V2/eval_ntest=full_s0_mlp.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier mlp --probe-label-space animacy \
    --output $EVALS_V2/eval_ntest=full_s0_mlp_animacy.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier mlp --probe-label-space subject \
    --output $EVALS_V2/eval_ntest=full_s0_mlp_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space category27 \
    --output $EVALS_V2/eval_ntest=full_s0_cnn.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space animacy \
    --output $EVALS_V2/eval_ntest=full_s0_cnn_animacy.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space subject \
    --output $EVALS_V2/eval_ntest=full_s0_cnn_subject.json
```

### Experiment 1.5 — BrainOmni 3b, averaged input

```bash
CALIB=neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b/config.json
EVALS_V2=neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b/evals/v2

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier linear --probe-label-space category27 \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier linear --probe-label-space animacy \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_animacy_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier mlp --probe-label-space category27 \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_mlp_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier mlp --probe-label-space animacy \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_mlp_animacy_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space category27 \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_cnn_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space animacy \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_cnn_animacy_avgcross_subject.json
```

### Experiment 2 — BrainOmni 3b_avgcross

```bash
CALIB=neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b_avgcross/config.json
EVALS_V2=neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b_avgcross/evals/v2

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier linear --probe-label-space category27 \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier linear --probe-label-space animacy \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_animacy_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space category27 \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_cnn_avgcross_subject.json

modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
    --tokenizer brainomni --calibration $CALIB \
    --n-test 0 --seed 0 \
    --probe-classifier cnn --probe-label-space animacy \
    --averaging cross_subject \
    --output $EVALS_V2/eval_ntest=full_s0_cnn_animacy_avgcross_subject.json
```
