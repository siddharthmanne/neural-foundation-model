# MEG tokenization — lean

Compressed reference for BrainOmni 3b probe results. Full doc:
[meg_tokenization.md](meg_tokenization.md).

## Setup

- Checkpoint: BrainOmni `V512_rvq4_win512_sf256_3b` (RVQ, V=512, 4 layers, 512 tokens/trial)
- Tasks: **category27** (chance 3.70%), **animacy** (50%), **subject** (25%)
- Probes: linear / MLP / **CNN**, all 5-fold CV, class-weighted CE, balanced accuracy
- Features: `tokens_all` / `tokens_rvq0` / `raw` (ceiling) / `random` (floor)

---

## Experiment 1 — `3b_nonavg` (single-trial finetune)

Train: 88,340 single trials. Best val 1.401.

### 1a — Category27 (chance 3.70%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, unweighted | 3.76 ± 0.07 | 3.71 ± 0.01 | **6.14 ± 0.44** | 3.70 ± 0.00 |
| Linear, weighted | 3.35 ± 0.91 | 3.42 ± 0.38 | **5.16 ± 0.21** | 3.64 ± 0.73 |
| MLP, weighted | 3.18 ± 0.97 | 2.94 ± 0.78 | **4.31 ± 0.49** | 3.09 ± 0.82 |
| **CNN, weighted** | 3.55 ± 0.54 | 3.77 ± 0.53 | **4.19 ± 0.87** | 3.61 ± 0.52 |

**Tokens at chance across all probes.** Raw barely lifts (~5%).

### 1b — Subject (chance 25%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted | 62.6 ± 1.0 | 45.1 ± 0.7 | 93.9 ± 0.9 | 25.2 ± 0.9 |
| MLP, weighted | 69.3 ± 1.0 | 52.2 ± 1.1 | 95.4 ± 0.4 | 25.4 ± 1.3 |
| **CNN, weighted** | **97.4 ± 0.4** | **92.1 ± 0.5** | **98.0 ± 0.5** | 25.5 ± 0.7 |

**Tokens nearly match raw under CNN.** Monotone linear → MLP → CNN.

### 1c — Animacy (chance 50%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted | 51.1 ± 1.8 | 52.6 ± 1.6 | **57.8 ± 2.5** | 50.0 ± 2.2 |
| MLP, weighted | 50.8 ± 2.0 | 51.4 ± 0.8 | **57.9 ± 2.1** | 49.9 ± 3.0 |
| **CNN, weighted** | 51.6 ± 2.0 | 51.2 ± 1.3 | 52.1 ± 1.3 | 49.5 ± 0.6 |

**Tokens at chance; raw lifts ~5–8pp.**

### 1d — Retrieval (zero trainable params)

| Task | tokens cos @1/@5 | tokens Jacc @1/@5 | raw @1/@5 | random @1/@5 |
|---|---|---|---|---|
| Cat27 | 3.71 / 3.59 | 3.38 / 3.81 | 3.76 / 3.94 | 4.11 / 3.54 |
| Animacy | 50.5 / 50.3 | 51.1 / 50.5 | 50.2 / 50.6 | 49.7 / 49.3 |
| Subject | **50.3 / 47.5** | **37.7 / 36.7** | **65.9 / 61.5** | 24.7 / 24.9 |

**Subject geometry lifts without training; cat/animacy at chance.**

---

## Experiment 2 — `3b_avgcross` (cross-subject-averaged finetune)

Train: 17,959 averaged trials. Best val 1.469. Subject task N/A by construction.

### 2a — Category27 (chance 3.70%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted (bal_acc) | 2.88 ± 1.80 | 3.24 ± 1.32 | **6.01 ± 2.37** | 4.02 ± 2.24 |
| **CNN, weighted (bal_acc)** | **4.65 ± 1.31** | 4.07 ± 0.82 | **4.78 ± 1.47** | 3.65 ± 1.24 |

**CNN tokens 4.65 vs raw 4.78 — essentially matched, ~+1σ above random.**

### 2c — Animacy (chance 50%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted (bal_acc) | 48.5 ± 4.6 | 50.9 ± 7.3 | **61.1 ± 3.4** | 48.2 ± 4.3 |
| **CNN, weighted (bal_acc)** | **52.2 ± 1.7** | 51.4 ± 0.3 | **58.9 ± 5.3** | 50.9 ± 2.5 |

**Tokens ~2pp above random; raw ~7pp above. Partial loss persists.**

### 2d — Retrieval

| Task | tokens cos @1/@5 | tokens Jacc @1/@5 | raw @1/@5 | random @1/@5 |
|---|---|---|---|---|
| Cat27 | 3.98 / 3.72 | 4.51 / — | **4.24 / 4.37** | 3.72 / 3.84 |
| Animacy | 48.6 / 50.7 | 48.6 / 49.4 | 50.7 / 51.4 | 51.2 / 49.9 |

**All at chance** — raw retrieval also at chance at n=1,126/2,245 (too weak for the small linear signal).

---

## Experiment 3 — `3b_occ_avg` (occipital-only + cross-subject averaged finetune)

Train: 17,959 averaged trials. Best val **1.269** (early-stopped epoch 7).
Channels: **39 occipital** (CTF `MLO + MRO + MZO`). Channel restriction lives
at the data layer; BrainOmni's cross-attention from sensors → 16 fixed
latents is channel-count agnostic, so no model surgery was needed. Tokens
still shape `(B, 16, 8, 4)` — same as Exp 1/2 — so only the `raw` feature
changes dim (`(B, 39, 281)` vs `(B, 271, 281)`); `random` and `tokens_*` are
unaffected.

### 3a — Category27 (chance 3.70%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted (bal_acc) | 2.80 ± 0.86 | 3.32 ± 1.39 | 3.76 ± 1.47 | 4.23 ± 1.83 |
| **CNN, weighted (bal_acc)** | 4.30 ± 1.48 | 3.74 ± 1.29 | 4.30 ± 2.14 | 3.83 ± 1.03 |

**All four tied at chance under either probe.** Under linear, tokens_all
sits *below* random (2.80 < 4.23) — small-sample noise, not signal.
`raw_occipital` lost the Exp 2 lift in both probes (linear: 6.01 → 3.76;
CNN: 4.78 → 4.30).

### 3c — Animacy (chance 50%)

| Classifier | tokens_all | tokens_rvq0 | raw | random |
|---|---|---|---|---|
| Linear, weighted (bal_acc) | **54.75 ± 5.70** | 52.41 ± 2.45 | 55.54 ± 5.83 | 50.80 ± 3.48 |
| **CNN, weighted (bal_acc)** | 48.93 ± 3.00 | 49.82 ± 4.33 | 51.41 ± 4.19 | 50.09 ± 1.52 |

**Linear surfaces a token lift the CNN doesn't.** `tokens_all` 54.75 vs
random 50.80 (~+4pp) — and notably *up* from Exp 2's linear 48.5 (+6pp
gain from restricting to occipital). Raw lift collapsed under both probes
(Exp 2 → Exp 3: linear 61.1 → 55.5, CNN 58.9 → 51.4) — the full-head raw
animacy signal was distributed across the scalp.

### 3d — Retrieval

| Task | tokens cos @1/@5 | tokens Jacc @1/@5 | raw @1/@5 | random @1/@5 |
|---|---|---|---|---|
| Cat27 | 4.00 / 3.22 | 3.69 / 3.54 | 3.21 / 3.71 | 3.95 / 3.78 |
| Animacy | 50.97 / 50.20 | **56.18** / 51.87 | 54.03 / 52.13 | 49.46 / 49.66 |

**Cat27 all at chance.** **Animacy tokens Jaccard@1 = 56.2 lifts ~+6pp above
random** — the only above-chance occipital-token number. Suggests a weak
layer-mixed signal that cosine pooling misses but bag-of-codes overlap catches.

### Verdict

- **Cat27 unchanged: occipital restriction is not the missing ingredient.**
  Tokens at chance both with and without restriction; restriction also
  cratered the raw cat27 baseline (linear −2.3pp, CNN −0.5pp), arguing the
  linear cat27 signal in full-head MEG is **distributed across the head**,
  not concentrated in occipital sensors.
- **Animacy: a partial token lift surfaces under the linear probe.**
  `tokens_all` linear bal_acc 54.75 vs random 50.80 (~+4pp) — and *up* from
  Exp 2's 48.5 (+6pp gain from restricting to occipital). CNN doesn't see
  it. The animacy bag-of-codes retrieval also lifted ~+6pp (Jacc@1 56.18
  vs 49.46). Both signals point to **layer-mixed / bag-of-codes-friendly
  geometry** that the CNN's spatial convolutions over the (16, 8) latent
  grid don't exploit.
- **Raw animacy collapsed** (linear 61.1 → 55.5, CNN 58.9 → 51.4) — the
  full-head raw lift was distributed across the scalp; occipital-only loses
  most of it.
- **Reconstruction-only training is still the cat27 bottleneck.** Best val
  dropped 1.469 → 1.269 with fewer channels (less to reconstruct =
  naturally lower loss); probe c27 didn't follow — consistent with the
  cross-Exp-1/2 verdict below.

---

## What the two experiments together say

- **Reconstruction objective is the bottleneck**, not single-trial noise. Both averaged-input diagnostics and the averaged-trained Exp 2 fail to recover cat27.
- **Variance budget goes to subject/sensor features**, not to fine category structure.
- **Next intervention**: contrastive/supervised auxiliary loss, OR Stage 2 (Cho2026 / EphysTokenizer). Don't iterate further on reconstruction-only finetuning.
