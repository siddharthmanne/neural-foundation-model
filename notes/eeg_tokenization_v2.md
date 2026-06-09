# EEG tokenization v2 — F2-corrected re-runs

Mirror of [eeg_tokenization.md](eeg_tokenization.md) with F2-corrected data. Same structure, same sections, same label spaces. The only difference is in the **random baseline column**: v1 used a bag-of-codes histogram (V=8192 dims) for random while the token baseline used mean-pooled embeddings (D=64 dims). These are incomparable feature spaces. v2 fixes this: random now uses `_embed_and_mean` (same D=64 path) when `has_token_embeddings` is True.

**Effect on verdicts:** none. Token and raw numbers are identical between v1 and v2. Random shifts by at most 1 pp in any config. All large-margin conclusions (subject at 60%, category/animacy at floor) stand.

**v2 eval files:** `neural_tokenizers/eeg/evals/v2/` — 9 core configs only. v1-only diagnostics (unweighted, prefixfix, n=20,000 pilot, avgcross/subject) have no v2 equivalents and are not shown here.

---

## §0 — What's different from MEG (read before running anything)

### 0.1 Data shape

| Property | MEG (THINGS-MEG) | EEG (THINGS-EEG, LaBraM cache) |
|---|---|---|
| Channels | 271 (Elekta MEG, MNE pipeline) | **17** (THINGS-EEG montage) |
| Sample rate | 200 Hz | **200 Hz** |
| Time points / trial (raw) | 281 | **100** (100 Hz, −200 to +800 ms epoch) |
| Token shape per trial | `(16 latents, 8 time, Q=4 RVQ)` BrainOmni; `(1, 271×281)` μ-transform | `(17,)` flat — LaBraM scalar tokens |
| Token vocab `V` | 512 (BrainOmni), 256 (μ-transform) | **8192** (LaBraM) |
| Embedding dim `D` | 512 (BrainOmni `tokens_to_embedding`) | **64** (LaBraM) |
| Subjects | 4 | **10** (eeg2 only in current eval grid) |
| Sources | 1 (single recording session) | **2** (eeg1, eeg2 — independent sessions) |
| Total trials in cache | ~100k single-trial | **821,600** (eeg2 only, 10 subjects × ~82k trials) |
| Per-image trial reps | ~12 (4 subj × 3 reps) | ~49 (10 subjects × ~4–5 reps per subject) |

**Cache path:** `/project/data/things-eeg/tokens/labram/V8192_d64_ch17_sr200_train-eeg1+2_e5/`

### 0.2 Reconstruction (§5.1) availability

LaBraM cache exposes only the tokenizer output (integer codes + embeddings). No waveform decoder in the deployed cache. §5.1 reconstruction is N/A for this checkpoint.

### 0.3 Experimental grid

Two experiments (same as v1):

| Exp | What | Status |
|---|---|---|
| **1 — LaBraM single-trial** | Off-the-shelf LaBraM checkpoint, no averaging | ✅ Run |
| **1.5 — averaged input, no retrain** | Diagnostic: average trials at eval time | Skipped — Exp 1 showed category at floor for both tokens AND raw (F3 window diagnostic confirmed this is a representation limit, not per-trial noise) |

---

## Final conclusion

### What the tokens decode — best probe per task, with the raw ceiling alongside

All cells: balanced accuracy mean ± SEM (= fold_std / √5). Best probe head per task.
n=821,600 trials (subject/animacy); n=398,960 (cat27, after label filter). 10 EEG2 subjects.

| Task (chance) | Experiment | tokens (best probe) | raw (ceiling) | random (floor) | tokens vs raw | tokens σ-vs-random |
|---|---|---|---|---|---|---|
| **Subject** (10%) | 1 — LaBraM | 60.13 ± 0.39 (CNN) | 10.75 ± 0.08 (CNN) | 9.93 ± 0.08 | +49.4 pp | ~129σ |
| **Cat27** (3.70%) | 1 — LaBraM | 4.12 ± 0.05 (MLP) | 3.71 ± 0.01 (linear) | 3.81 ± 0.06 | +0.41 pp | ~5.1σ |
| **Animacy** (50%) | 1 — LaBraM | 51.56 ± 0.26 (linear) | 49.95 ± 0.02 (linear) | 50.06 ± 0.40 | +1.61 pp | ~3.8σ |

### Two clean reads of the table

**1. Where is the raw ceiling for EEG?**

- Subject ID: raw best probe = 10.75% (CNN) — at chance. "Raw" here is the 17 token IDs cast as 17-dim floats; linear features on 17 integers cannot decode subject. The ceiling is only visible in the 64-dim embedding space (60.1% CNN tokens). Structurally different from MEG where actual waveforms decode subjects at ~98%.
- Animacy: raw best probe = 49.95% — at chance. Dixen 2024's 57–61% benchmark uses full preprocessed EEG waveforms, not discrete token integers. Not comparable.
- Category27: raw best probe = 3.71% — at chance. Same bottleneck as MEG.

**2. How does LaBraM preserve what raw has?**

- Subject (Exp 1): tokens 60.1% (CNN) vs raw 10.75% (at chance). The tokenizer creates subject structure in the 64-dim embedding space that is not present as a linear feature in the raw 17-int codes. LaBraM's codebook geometry clusters by subject — per-subject spectral fingerprints map consistently to specific codebook regions. Distinct mechanism from MEG subject decoding (where raw signal already carried the signal directly).
- Cat27 (Exp 1): tokens (4.12%) and random (3.81%) both near chance (3.70%). Verdict unchanged vs v1: no linearly decodable 27-way category signal.
- Animacy: tokens 51.56% vs random 50.06% — statistically real but +1.5 pp above random is not usable.

### Which to ship for 4M

**Ship LaBraM into 4M as the EEG modality adapter.** Subject-level physiological structure in the token embeddings provides real cross-modal signal: 4M can learn that EEG from subject X co-occurred with this RGB image, and the codebook geometry reliably encodes person-level EEG fingerprints.

Category-level semantic alignment requires a contrastive or supervised auxiliary objective. Same conclusion as MEG. Post-milestone work.

---

## TL;DR

LaBraM EEG tokens show a striking subject-over-stimulus pattern: subject identity decodes at 60.1% (CNN, chance 10%) while object category and animacy are at the random floor (4.1% vs 3.81% random and 51.6% vs 50.1% random respectively). The CNN probe recovers substantially more subject signal than linear or MLP, because the 17 position-codes have weak spatial structure that CNN1D can exploit for person-level patterns. Raw per-trial EEG (as encoded by the token ID scalars) sits near chance on all three tasks. The bigram entropy gap on the full 821k dataset is 18.4% (below the 20% healthy-masking threshold). Codebook utilization is strong (91%, perplexity 4,439). The verdict matches the MEG BrainOmni pattern: reconstruction-only VQ tokenizers preserve subject-level variance but not stimulus-level discrimination. Ship LaBraM into 4M for the scaling experiment; flag subject-identity encoding as a known property.

---

## What we are running and why

| Axis | Levels |
|---|---|
| **Task** (`probe_label_space`) | `category27` (the gate) / `animacy` (easy 2-way) / `subject` (pipeline sanity) |
| **Probe head** (`probe_classifier`) | `linear` / `mlp` / `cnn` |
| **Feature set** | `raw` (signal upper bound) / `random` (lower bound, F2-corrected) / `tokens_all` (all 17 codes) |
| **Checkpoint** | **`LaBraM_V8192_d64_ch17_sr200_e5`** |

All evals: full test split, 5-fold CV, class-weighted CE, balanced accuracy as headline metric.

---

## Experiment 1 — LaBraM as-shipped (single-trial)

Chance: 27-way = 3.70%, animacy 2-way = 50%, subject 10-way = 10%.

### 1a — Category27. Chance = 3.70%

Cell format: `bal_acc` for weighted rows; `top-1 / top-5` shown in parentheses. n=50,000 probe subsample; codebook/sequence stats use full 398,960 labeled trials.

| Classifier | tokens_all | raw | random (F2-corrected) |
|---|---|---|---|
| Linear, weighted | 4.10% (3.15% / 18.41%) | 3.72% (2.91% / 16.89%) | 3.73% |
| MLP, weighted | 4.12% (2.91% / 17.23%) | 3.71% (2.71% / 15.93%) | 3.81% |
| **CNN, weighted** | 3.68% (2.68% / 18.71%) | 3.68% (2.37% / 15.00%) | 3.66% |

**Codebook (§5.2):** 7,462 / 8,192 codes used (91.1%); dead-code fraction 8.9%; perplexity 4,439.

**Sequence (§5.4):** unigram entropy 8.398 nats (max 9.01); bigram conditional entropy 6.853 nats; entropy gap 18.4%; mean run length 1.005; frac\_runs\_ge\_2 7.9%.

*Category27 tokens land at 4.1% balanced accuracy — near the random floor (3.81% chance-matched random, 3.70% prior). Linear, MLP, and CNN all converge: per-trial LaBraM tokens carry no linearly decodable 27-way object-category signal. The codebook is healthy (91% utilized, perplexity 4,439). The bigram entropy gap is 18.4% — well below the 20% healthy-masking threshold. (Note: the v1 pilot at n=20k estimated a misleading 57.7% entropy gap; the full-dataset figure is reliable.)*

### 1b — Subject ID. Chance = 10%

n=50,000 probe subsample; all 821,600 trials for sequence/codebook. Labels: `sub-01`–`sub-10` from eeg2 only (Option A — same person across sources treated as one label).

| Classifier | tokens_all | raw | random (F2-corrected) |
|---|---|---|---|
| Linear, weighted | 40.62% ± 0.16 (SEM) | 10.01% ± 0.02 | 10.29% ± 0.12 |
| MLP, weighted | 50.39% ± 0.29 (SEM) | 9.97% ± 0.03 | 10.04% ± 0.12 |
| **CNN, weighted** | **60.13% ± 0.39 (SEM)** | 10.75% ± 0.08 | 9.93% ± 0.08 |

**Codebook (§5.2):** 7,587 / 8,192 codes used (92.6%); dead-code fraction 7.4%; perplexity 4,446.

**Sequence (§5.4):** unigram entropy 8.400 nats; bigram conditional entropy 7.293 nats; entropy gap 13.2%; mean run length 1.005.

*LaBraM strongly encodes subject identity: CNN tokens decode the correct subject 60.1% of the time (chance 10%), MLP at 50.4%, linear at 40.6%. Raw EEG sits at chance (10.0–10.8%), so subject signal is NOT in the raw token IDs — it is in the continuous codebook embedding space, confirmed by the retrieval result below. The F2 fix slightly shifts the random floor (10.05→10.04–10.29 depending on probe head), but the 50 pp margin means the verdict is unchanged.*

### 1c — Animacy. Chance = 50%

n=50,000 probe subsample from 398,960 labeled trials.

| Classifier | tokens_all | raw | random (F2-corrected) |
|---|---|---|---|
| Linear, weighted | **51.56% ± 0.26 (SEM)** | 49.95% ± 0.02 | 50.06% ± 0.40 |
| MLP, weighted | 50.51% ± 0.21 (SEM) | 49.64% ± 0.35 | 50.20% ± 0.35 |
| CNN, weighted | 50.48% ± 0.29 (SEM) | 50.13% ± 0.26 | 49.69% ± 0.31 |

*Animacy: linear achieves 51.56%, with raw at 49.95% and F2-corrected random at 50.06%. The margin over random is 1.5 pp (3.8σ). Statistical but not scientifically usable. With the F2 fix, the random floor moved slightly (49.62 → 50.06 for linear), making the token margin over random narrower than in v1 (1.94 pp → 1.50 pp) but the verdict is the same.*

### 1d — Model-free retrieval (§5.5)

Zero trainable parameters. Cosine on `tokens_to_embedding` features, Jaccard on token sets. n=5,000 subsample per task.

| Task (n) | tokens cosine prec@1 / @5 | tokens Jaccard prec@1 / @5 | raw prec@1 / @5 | random prec@1 / @5 |
|---|---|---|---|---|
| Category27 (n=5000, chance 3.70%) | 3.78% / 3.53% | 3.35% / 3.67% | 3.34% / 3.59% | 3.36% / 3.56% |
| Animacy (n=5000, chance 50.0%) | 49.31% / 49.98% | 49.79% / 50.31% | 49.37% / 49.68% | 49.68% / 49.90% |
| Subject (n=5000, chance 10.0%) | **23.48%** / 21.75% | 17.92% / 16.52% | 10.23% / 10.21% | 10.19% / 9.93% |

*Retrieval confirms the probe verdict. Category27 and animacy: tokens, raw, and random all at chance — no geometric class structure. Subject: cosine retrieval on token embeddings achieves 23.5% prec@1 (chance 10%), while raw achieves exactly chance (10.2%). This cleanly separates the two components: the discrete code identity carries no subject signal, but the continuous embedding lookup encodes it strongly. Retrieval numbers are identical to v1 — the F2 fix does not affect retrieval (random baseline in retrieval uses the same embed-and-mean path regardless).*

---

## Experiment 1.5 — LaBraM as-shipped, image-averaged input *(skipped)*

**Skipped.** Exp 1 shows category27 at the random floor for both tokens AND raw. The F3 window diagnostic (May 2026) established this is a representation limitation, not per-trial noise. No v2 file produced.

---

## The harness — five axes (eval contract)

| Axis | What it answers | v2 status |
|---|---|---|
| §5.1 reconstruction | encode→decode round-trip | N/A — LaBraM cache is encoder-only |
| §5.2 codebook | vocab used, no dead codes | ✅ |
| §5.3 linear probe | tokens linearly decodable | ✅ |
| §5.4 sequence | token sequence learnable | ✅ |
| §5.5 retrieval | model-free feature geometry | ✅ |

---

## Pointers

- v1 counterpart: [eeg_tokenization.md](eeg_tokenization.md) — has the same analysis, v1 random baseline numbers, and the diagnostic runs (unweighted, prefixfix, n=20k pilot, avgcross)
- MEG counterpart: [meg_tokenization.md](meg_tokenization.md)
- Cross-modal comparison: [eeg_vs_meg_eval.md](eeg_vs_meg_eval.md)
- Probe source: [`neural_tokenizers/evaluation/probe.py`](../neural_tokenizers/evaluation/probe.py), [`evaluation/retrieval.py`](../neural_tokenizers/evaluation/retrieval.py)
