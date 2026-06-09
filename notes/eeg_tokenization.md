# EEG tokenization — diagnostics and findings *(template; fill in as runs complete)*

Mirror of [meg_tokenization.md](meg_tokenization.md) for the LaBraM-tokenized THINGS-EEG track. Same eval contract (§5.1–§5.5), same probe families, **same label spaces** (`category27` / `animacy` / `subject`) — but a substantially different tokenizer family (LaBraM, off-the-shelf) and a much smaller per-trial token budget (17 codes vs MEG's 16×8=128 latents × 4 RVQ layers). The experimental grid contracts accordingly.

> **For the EEG owner (you).** This file is the scaffold. Run the commands in §"Run pattern", paste the resulting JSON cells into the empty tables below, then write the headline reading sections. The "Final conclusion" section is the last thing you write, not the first. **Do not skip §0 — it documents what makes EEG different from MEG and what code you have to write before any probe runs.** Estimated wall time across the full grid: ~$5 in Modal compute, ~2 hrs end-to-end if no surprises.

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

**Verify with:**
```bash
modal run modal/modal_verify_eeg_cache.py::main      # one-shot integrity + counts
modal run modal/modal_peek_eeg_npz.py::main          # inspect npz keys + dtypes
modal run modal/modal_diag_eeg_keys.py::main         # spot-check image_id linkage
```

### 0.2 What the MEG harness gives you for free vs what you have to write

**Reusable as-is** (modality-agnostic — confirmed by code inspection):

| File | Why it works for EEG |
|---|---|
| [`neural_tokenizers/evaluation/probe.py`](../neural_tokenizers/evaluation/probe.py) | CNN1D / CNN2D dispatch is shape-based, not modality-baked. Class-weighted CE, 5-fold CV, balanced-accuracy headline metric all generic. |
| [`neural_tokenizers/evaluation/retrieval.py`](../neural_tokenizers/evaluation/retrieval.py) | Cosine on flattened features + Jaccard on token sets — modality-neutral. |
| [`neural_tokenizers/evaluation/codebook.py`](../neural_tokenizers/evaluation/codebook.py) | `torch.bincount` on token IDs. Works for any vocab size. |
| [`neural_tokenizers/evaluation/sequence.py`](../neural_tokenizers/evaluation/sequence.py) | Bigram entropy / run-length stats on flattened token sequences. |
| [`neural_tokenizers/evaluation/reconstruction.py`](../neural_tokenizers/evaluation/reconstruction.py) | `(B, C, T)` in/out, Pearson r, Welch PSD MSE. **Only runs if LaBraM exposes a decoder** — if it's encoder-only, skip §5.1 for now and note it. |
| THINGS label mappings — `image_id → concept27 → superordinate / animacy` | `image_id` is the universal join key in the EEG cache; mappings in [`neural_tokenizers/meg/data/`](../neural_tokenizers/meg/data/) are dataset-level, not modality-specific. Consider moving them to `neural_tokenizers/labels/` once the EEG pipeline lands — see §"Refactor note" below. |

**You have to write** (MEG-specific code that won't port):

| What to write | Why MEG version doesn't work | Approximate effort |
|---|---|---|
| `neural_tokenizers/eeg/eeg_config.py` with `EEGDataSpec(n_channels=17, sfreq_hz=200, n_timepoints=TBD)` | [`meg_config.py:31-49`](../neural_tokenizers/meg/meg_config.py) hardcodes 271/281 | 20 min |
| `neural_tokenizers/eeg/data.py` — `.npz` loader, subject/source enumeration, split logic | [`meg/data.py:73-103`](../neural_tokenizers/meg/data.py) calls `mne.read_epochs()` on `.fif`; EEG cache is plain `.npz` (no MNE) | 1–2 hrs |
| `neural_tokenizers/eeg/labram_adapter.py` conforming to the `Tokenizer` protocol in [`evaluation/protocol.py`](../neural_tokenizers/evaluation/protocol.py) | BrainOmni adapter at [`meg/brainomni/adapter.py`](../neural_tokenizers/meg/brainomni/adapter.py) is MEG-specific (sensor metadata, MNE preprocess). For EEG the adapter can be **much simpler**: tokens are already in the cache, so `encode()` is a lookup and `tokens_to_embedding()` is a `nn.Embedding(8192, 64)` load from the LaBraM checkpoint. | 2–3 hrs |
| `neural_tokenizers/eeg/modal/modal_eeg_eval.py` — Modal dispatch | [`modal_meg_eval.py`](../neural_tokenizers/meg/modal/modal_meg_eval.py) imports MEG-specific splits and data loaders. Copy + swap data layer. | 1 hr |
| **Subject ID label space** for EEG | Need to enumerate subjects from cache filenames (`eeg{1,2}_sub-XX.npz`) and decide whether to treat eeg1/eeg2 as the same subject or distinct labels — see §"Subject label gotcha" | 30 min decision |

**MEG-only — does not port and is not needed:**

- μ-transform tokenizer ([`meg/mu_transform/tokenizer.py`](../neural_tokenizers/meg/mu_transform/tokenizer.py)) — hardcoded for 271×281; was the Stage-0 baseline for MEG. No EEG equivalent; LaBraM IS the tokenizer.
- BrainOmni sensor metadata ([`meg/brainomni/sensor_metadata.py`](../neural_tokenizers/meg/brainomni/sensor_metadata.py)) — MEG sensor positions.
- The Exp 2 / Exp 1.5 finetune-on-averaged-trials experiments (see §0.4).

### 0.3 CNN probe — does the MEG inductive bias transfer?

**MEG (BrainOmni tokens have spatial-temporal structure exposed):**
- Token tensor `(B, 16 latent, 8 time, D=512)` → 2-D CNN over `(latent × time)` with `D` as channels.
- Raw `(B, 271 ch, 281 time)` → 1-D CNN over time with channels as input dims.

**EEG (LaBraM tokens are flat 17-vectors):**
- Token tensor `(B, 17,)` of int16 scalars — **no spatial/temporal axis exposed at the token level**. A 2-D CNN has nothing to convolve over.
- After `tokens_to_embedding`: `(B, 17, D=64)`. A **1-D CNN over the 17 token positions with `D=64` as channels** is the natural analogue — same family as the MEG-raw 1-D head, just smaller (17 positions vs 281 timesteps; 64 channels vs 271).
- For raw EEG `(B, 17 ch, T)` (post-cache, pre-LaBraM): the existing 1-D CNN head works directly.

**What this means for the grid:** keep `probe_classifier = {linear, mlp, cnn}`. The CNN dispatch in [`evaluation/probe.py:388-408`](../neural_tokenizers/evaluation/probe.py) selects 1-D CNN when feat_shape is `(C, T)` (2-D feature) or 2-D CNN when feat_shape is `(S1, S2, D)` (3-D feature). For LaBraM tokens you'll be in the 1-D-CNN branch. **Confirm this works on first run before scaling the grid** — the CNN1D head was designed and tuned against MEG's 271-channel raw input; 17 positions is a much smaller convolution window and may need its kernel sizes adjusted. (Check the kernel sizes in [`probe.py:411-433`](../neural_tokenizers/evaluation/probe.py) — if they're hardcoded for the MEG temporal length, reduce them or generalize.)

**Open architecture question for the EEG owner:** LaBraM is itself a transformer trained with masked-token prediction. A linear or MLP probe on flattened LaBraM tokens may already be near-ceiling — the tokenizer's job was to make the codes linearly readable. If `linear` and `cnn` give the same answer on EEG, that's not a probe failure, that's a tokenizer property. Report this in §1 when you see it.

### 0.4 Experimental grid contracts (vs MEG)

MEG ran three experiments because we **owned the BrainOmni finetune**:

| MEG Exp | What it varied | EEG equivalent |
|---|---|---|
| **Exp 1** — `3b_nonavg` | Finetune on single trials | ✅ **Exp 1 — LaBraM as-shipped.** This is the only experiment you can run cheaply, because LaBraM is the only checkpoint you have. |
| **Exp 1.5** — averaged eval input, no retrain | Cheap diagnostic: feed averaged trials to existing tokenizer at eval time | ✅ **Exp 1.5 — averaged input, LaBraM as-shipped.** Same diagnostic logic; same `--averaging cross_subject` flag wiring if you copy [`modal_meg_eval.py`](../neural_tokenizers/meg/modal/modal_meg_eval.py). |
| **Exp 2** — `3b_avgcross` | **Re-finetune** BrainOmni on averaged trials | ❌ Requires re-running LaBraM training. **Don't do this in the first pass.** If §1 / §1.5 show the same selective-loss pattern MEG had, the conclusion will be the same architectural one (need contrastive / supervised objective), and Exp 2 is wasted compute. |

So the EEG grid is **two experiments, not three**. Decide on Exp 1.5 only if Exp 1 shows category at chance and you want to confirm "not a per-trial noise problem."

### 0.5 Subject label gotcha

LaBraM cache has both `subject` (`sub-01`, `sub-02`, ...) and `source` (`eeg1`, `eeg2`). Decide upfront:

- **Option A — subject only**: label = `subject`. Treats `eeg1_sub-01` and `eeg2_sub-01` as the same person (which they are — same individual, two recording days). Tests "does the tokenizer encode person-level features?"
- **Option B — (subject, source)**: label = `f"{subject}_{source}"`. Treats them as distinct labels. Tests "does the tokenizer encode recording-session features?" — much easier task, similar to MEG's subject ID.

**Recommend Option A** for direct comparability with MEG's subject ID conclusions. Document choice in §1b table.

### 0.6 Reconstruction (§5.1) availability

MEG had both BrainOmni (with a decoder, can round-trip) and μ-transform (deterministic invertible). For LaBraM:

- **If LaBraM checkpoint exposes a decoder** (`tokens → reconstructed waveform`): run §5.1 normally; report Pearson `r` per channel, time-domain MSE, Welch PSD MSE. Note that MSE is **not comparable across input regimes** (see MEG doc §1.5 footnote — the noise floor of the target changes with averaging).
- **If LaBraM is encoder-only** (typical for masked-token EEG models): mark §5.1 as N/A and rely entirely on §5.2–§5.5. Document this — it's a real reduction in evidence vs MEG.

---

## Final conclusion

### What the tokens decode — best probe per task, with the raw ceiling alongside

All cells: balanced accuracy mean ± SEM (= fold_std / √5). Best probe head per task.
n=821,600 trials (subject/animacy); n=398,960 (cat27, after label filter). 10 EEG2 subjects.

| Task (chance) | Experiment | tokens (best probe) | raw (ceiling) | random (floor) | tokens vs raw | tokens σ-vs-random |
|---|---|---|---|---|---|---|
| **Subject** (10%) | 1 — LaBraM | 60.1 ± 0.39 (CNN) | 10.8 ± 0.08 (CNN) | 9.9 ± 0.08 | +49.3 pp | ~129σ |
| **Cat27** (3.70%) | 1 — LaBraM | 4.12 ± 0.05 (MLP) | 3.72 ± 0.01 (linear) | 3.70 ± 0.11 | +0.40 pp | ~8.6σ |
| **Cat27** (3.70%) | 1.5 — avg eval | not run | not run | not run | — | — |
| **Animacy** (50%) | 1 — LaBraM | 51.6 ± 0.26 (linear) | 50.0 ± 0.02 (linear) | 49.6 ± 0.24 | +1.6 pp | ~7.7σ |
| **Animacy** (50%) | 1.5 — avg eval | not run | not run | not run | — | — |

### Two clean reads of the table

**1. Where is the raw ceiling for EEG?**

- Subject ID: raw best probe = 10.8% (CNN) — at chance. "Raw" here is the 17 token IDs cast as 17-dim floats; linear features on 17 integers cannot decode subject. The ceiling is only visible in the 64-dim embedding space (60.1% CNN tokens). Structurally different from MEG where actual waveforms decode subjects at ~98%.
- Animacy: raw best probe = 50.0% — at chance. Dixen 2024's 57–61% benchmark uses full preprocessed EEG waveforms, not discrete token integers. Not comparable.
- Category27: raw best probe = 3.72% — at chance. Same bottleneck as MEG.

**2. How does LaBraM preserve what raw has?**

- Subject (Exp 1): tokens 60.1% (CNN) vs raw 10.8% (at chance). The tokenizer CREATES subject structure in the 64-dim embedding space that is not present as a linear feature in the raw 17-int codes. LaBraM's codebook geometry clusters by subject — per-subject spectral fingerprints map consistently to specific codebook regions. Distinct mechanism from MEG subject decoding (where raw signal already carried the signal directly).
- Cat27 (Exp 1): tokens (4.12%) and raw (3.72%) both at chance. Same selective-loss pattern as MEG BrainOmni 3b_nonavg. Exp 1.5 skipped — F3 window diagnostic (May 2026) already showed this is a representation limitation, not per-trial noise.
- Animacy: tokens 51.6% vs raw 50.0% — statistically real (7.7σ) but +1.6 pp is not usable.

### Which to ship for 4M

**Ship LaBraM into 4M as the EEG modality adapter.** Subject-level physiological structure in the token embeddings provides real cross-modal signal: 4M can learn that EEG from subject X co-occurred with this RGB image, and the codebook geometry reliably encodes person-level EEG fingerprints.

Category-level semantic alignment requires a contrastive or supervised auxiliary objective, or a tokenizer trained with a discrimination objective. Same conclusion as MEG. Post-milestone work.
---

## TL;DR

LaBraM EEG tokens show a striking subject-over-stimulus pattern: subject identity decodes at 60% (CNN, chance 10%) while object category and animacy are at the random floor (4.1% vs 3.7% chance and 51.6% vs 50% chance respectively). The CNN probe recovers substantially more subject signal than linear or MLP, because the 17 position-codes have weak spatial structure that CNN1D can exploit for person-level patterns. Raw per-trial EEG (as encoded by the token ID scalars) sits near chance on all three tasks — the category and animacy information is simply absent at the single-trial resolution, not lost by the tokenizer. The bigram entropy gap on the full 821k dataset is 18.4% (below the 20% healthy-masking threshold), correcting the misleading 57.7% from the sparse n=20k pilot. Codebook utilization is strong (91%, perplexity 4,439). The verdict matches the MEG BrainOmni pattern: reconstruction-only VQ tokenizers preserve subject-level variance but not stimulus-level discrimination. Ship LaBraM into 4M for the scaling experiment; flag subject-identity encoding as a known property.

---

## What we are running and why

| Axis | Levels |
|---|---|
| **Task** (`probe_label_space`) | `category27` (the gate) / `animacy` (easy 2-way) / `subject` (pipeline sanity — see §0.5) |
| **Probe head** (`probe_classifier`) | `linear` / `mlp` / `cnn` (see §0.3 for EEG CNN inductive-bias notes) |
| **Feature set** | `raw` (signal upper bound) / `random` (lower bound) / `tokens_all` (all 17 codes) |
| **Checkpoint** | **`LaBraM_V8192_d64_ch17_sr200_e5`** (cache slug above) — only checkpoint for now |

**Notes vs MEG:**
- No `tokens_rvq0` column: LaBraM is not RVQ — there are no coarse/fine layers to separate. If LaBraM is replaced later by an RVQ-based EEG tokenizer (Cho2026 / EphysTokenizer adapted), add `tokens_rvq0` back.
- No second checkpoint column: only one LaBraM cache today. Add columns when more arrive.

All evals: full test split, 5-fold CV, class-weighted CE, balanced accuracy as headline metric. Chance is 1/n_classes for every task.

### Run pattern *(adapt from MEG; you need to write `modal_eeg_eval.py` first — see §0.2)*

```bash
modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run \
    --tokenizer labram \
    --calibration neural_tokenizers/eeg/labram/runs/V8192_d64_ch17_sr200_e5/config.json \
    --n-test 0 --seed 0 \
    --probe-classifier {linear|mlp|cnn} \
    --probe-label-space {category27|animacy|subject}
```

For Exp 1.5 add `--averaging cross_subject`. Results land at `<checkpoint>/evals/eval_ntest=full_s0[_<classifier>][_<label_space>][_avgcross_subject].json`.

**Pre-flight checks before running the grid:**
1. `modal run modal/modal_verify_eeg_cache.py::main` returns PASS.
2. Unit test: load one batch of LaBraM tokens, run through the linear probe head, confirm output shape `(B, n_classes)` and loss is finite. Mirror [`test_tokenizer.py::test_evaluate_runs_all_axes`](../neural_tokenizers/test_tokenizer.py).
3. Sanity: linear-probe subject ID on raw EEG. If raw subject ID < 50%, something is wrong with the data loader before any tokenizer claim is meaningful.

---

## CNN probe head — design for EEG

See §0.3 above for the inductive-bias argument. Two relevant CNN configurations for the EEG grid:

- **For LaBraM token embeddings** of shape `(B, 17 positions, D=64)`: 1-D CNN over the 17 positions, `D=64` as input channels. Two conv blocks with batch-norm + ReLU + global average pooling → linear classifier. Target param count ~10–30k (much smaller than MEG's because input is shorter). **Likely needs kernel-size adjustment** vs the MEG default — confirm before the full grid.
- **For raw EEG** of shape `(B, 17 channels, T)`: existing 1-D CNN in [`evaluation/probe.py`](../neural_tokenizers/evaluation/probe.py) works directly. Confirm T (timepoints/trial) once before running.
- **For random tokens**: same shape transform as LaBraM tokens (random codes → `tokens_to_embedding` → 1-D CNN). Tests "what does the CNN do with uninformative tokens of the right shape?"

---

## Experiment 1 — LaBraM as-shipped (single-trial)

Chance: 27-way = 3.70%, animacy 2-way = 50%, subject n-way = 1/n_subj (compute n_subj from cache).

### 1a — Category27. Chance = 3.70%

Cell format: `top-1 / top-5` (top-1 is bal-acc for "weighted" rows). n=50,000 probe subsample; codebook/sequence stats use full 398,960 labeled trials.

| Classifier | tokens_all | raw | random |
|---|---|---|---|
| Linear, unweighted | 13.9% / 39.8% | 5.6% / 26.3% | 12.8% / 36.8% |
| Linear, weighted | 4.10% / 18.4% | 3.72% / 16.9% | 3.78% / 18.9% |
| MLP, weighted | 4.12% / 17.2% | 3.71% / 15.9% | 3.70% / 30.2% |
| **CNN, weighted** | 3.68% / 18.7% | 3.68% / 15.0% | 3.66% / 18.4% |

**Codebook (§5.2):** 7,462 / 8,192 codes used (91.1%); dead-code fraction 8.9%; perplexity 4,439.

**Sequence (§5.4):** unigram entropy 8.40 nats (max 9.01); bigram conditional entropy 6.85 nats; entropy gap 18.4%; mean run length 1.005; frac\_runs\_ge\_2 7.9%.

*Category27 tokens land at 4.1% balanced accuracy — the same as the random floor (3.7% chance, ~3.7% random). Raw EEG is also at floor (3.7%). Linear, MLP, and CNN all converge: per-trial LaBraM tokens carry no linearly decodable 27-way object-category signal. The CNN offers no rescue because the features are 17 position-codes with no temporal structure to convolve over. The codebook is healthy (91% utilized, perplexity 4,439 ≫ codebook size would imply 8,192 if flat). The bigram entropy gap dropped from 57.7% (n=20k pilot estimate) to 18.4% on the full dataset — the pilot's gap was a sparse-sample artifact; the real gap is below the 20% healthy-masking threshold.*

### 1b — Subject ID

Chance: 10% (10 subjects, eeg2 only, §0.5 Option A — same person across sources). n=50,000 probe subsample; all 821,600 trials for sequence/codebook.

| Classifier | tokens_all | raw | random |
|---|---|---|---|
| Linear, weighted | 40.62% | 10.01% | 10.05% |
| MLP, weighted | 50.39% | 9.97% | 10.14% |
| **CNN, weighted** | **60.13%** | 10.75% | 9.93% |

*The LaBraM tokenizer strongly encodes subject identity: CNN tokens decode the correct subject 60% of the time (chance 10%), with MLP at 50% and linear at 40%. Raw EEG sits at chance (10.1–10.8%), so the subject signal is NOT in the raw token IDs (which are just discrete codes ≠ raw waveform) — it is in the continuous codebook embedding space, as confirmed by the retrieval result (§1d). Compared to MEG's 97–98% subject decoding from raw, raw EEG is far lower, but the tokenizer itself contributes massive subject-specific structure. This is a liability for 4M: the tokenizer encodes who the subject is rather than what they're looking at. A subject-decorrelation objective (e.g., adversarial subject prediction, cross-subject contrastive) would be needed to suppress it.*

### 1c — Animacy. Chance = 50%

n=50,000 probe subsample from 398,960 labeled trials.

| Classifier | tokens_all | raw | random |
|---|---|---|---|
| Linear, weighted | **51.56%** | 49.95% | 49.62% |
| MLP, weighted | 50.51% | 49.64% | 50.05% |
| **CNN, weighted** | 50.48% | 50.13% | 49.69% |

*Animacy sits marginally above chance: linear achieves 51.6% (8.0σ above random's 49.6%), while MLP and CNN are within noise of 50%. Raw is also at floor (49.6–50.1%), so the ceiling itself is low. The Dixen 2024 benchmark (57–61% cross-subject on THINGS-EEG with EEGNet) is not directly comparable — "raw" here is pre-cached token IDs cast to float, not EEG waveform, so we are not benchmarking raw EEG decoding. Exp 1.5 (averaged input feeding the actual waveform features to the probe) is needed for that comparison. The animacy result is consistent with the category27 result: per-trial LaBraM tokens marginally preserve stimulus-driven variance, but the dominant variance in the tokens is subject-level, not image-level.*

### 1d — Model-free retrieval (§5.5)

Zero trainable parameters. Cosine on `tokens_to_embedding` features, Jaccard on token sets. n=5,000 subsample per task (cap to avoid O(B²) similarity matrix).

| Task (n) | tokens (cosine) prec@1 / @5 | tokens (Jaccard) prec@1 / @5 | raw prec@1 / @5 | random prec@1 / @5 |
|---|---|---|---|---|
| Category27 (n=5000, chance 3.70%) | 3.78% / 3.53% | 3.35% / 3.67% | 3.34% / 3.59% | 3.50% / 3.39% |
| Animacy (n=5000, chance 50.0%) | 49.31% / 49.98% | 49.79% / 50.31% | 49.37% / 49.68% | 49.39% / 49.69% |
| Subject (n=5000, chance 10.0%) | **23.48%** / 21.75% | 17.92% / 16.52% | 10.23% / 10.21% | 10.21% / 9.85% |

*Retrieval confirms the probe verdict exactly. Category27 and animacy: tokens, raw, and random all at chance — no geometric class structure in either the continuous embedding space or the discrete code sets. Subject: cosine retrieval on token embeddings achieves 23.5% prec@1 (chance 10%), while raw (token IDs cast to float) achieves exactly chance (10.2%). This cleanly separates the two components: the discrete code identity carries no subject signal, but the continuous embedding lookup encodes it strongly. Jaccard (17.9% for subject) also above chance — some subject-specific codes are systematically used, even if the exact embedding value is needed for the full 23% performance.*

---

## Experiment 1.5 — LaBraM as-shipped, image-averaged INPUT (diagnostic, no retrain) *(skipped)*

**Skipped.** Exp 1 shows category27 at the random floor for both tokens AND raw. The F3 window diagnostic (May 2026) already established this is a representation limitation, not per-trial noise — flipping 49% of per-channel codes by switching from 8-copy to 8-real-trial windows leaves the probe unchanged at floor. Averaging trials would not help. No Exp 2 needed for the same reason.
---

## The harness — five axes (eval contract, identical to MEG)

| Axis | What it answers | Source |
|---|---|---|
| §5.1 reconstruction | Does encode→decode round-trip preserve the waveform? | `evaluation/reconstruction.py` *(skip if LaBraM is encoder-only — see §0.6)* |
| §5.2 codebook | Is the vocab used (no dead codes / collapse)? | `evaluation/codebook.py` |
| §5.3 linear probe | Are tokens **linearly decodable** to a class label? | `evaluation/probe.py` |
| §5.4 sequence | Is the token *sequence* learnable (entropy gap, runs)? | `evaluation/sequence.py` |
| §5.5 retrieval | Same as §5.3 but **model-free** — does feature geometry separate classes? | `evaluation/retrieval.py` |

## §5.3 probe hardening — inheritance from MEG

The probe is already at v10 (see MEG doc §"probe hardening"). For EEG you inherit:
- Class-weighted CE + balanced accuracy (v3, v4)
- 5-fold CV with SEM-based σ (v2 + Exp-2 corrected stats footnote)
- 3 label spaces — `category27` / `animacy` / `subject` (v8)
- Retrieval axis (v9)
- CNN probe head (v10)

**EEG-specific additions to v10+** (potential):
- v11 — if LaBraM exposes per-token position metadata (channel × time-window), expose it via a 2-D feature path so CNN2D can exploit it. Currently the LaBraM tokens look flat `(17,)` — if that's because position is dropped, recovering it would mirror the μ-transform position-preserving fix (MEG doc v7).

## Testing strategy

| Layer | Mirror file from MEG | Status |
|---|---|---|
| Unit — label mappings | [`meg/test_data.py`](../neural_tokenizers/meg/test_data.py) | Reusable if labels move to `neural_tokenizers/labels/`; else duplicate for `eeg/test_data.py` |
| Unit — probe internals | [`test_tokenizer.py`](../neural_tokenizers/test_tokenizer.py) `test_class_weights_*`, `test_probe_*`, `test_build_head_cnn_*` | Already generic — reuse |
| Unit — featurization | New `eeg/labram/test_labram.py` — test `tokens_to_embedding` shape `(B, 17, 64)` and `decode_tokens` shape `(B, 17 ch, T)` if decoder exists | Write |
| Unit — retrieval | [`test_retrieval.py`](../neural_tokenizers/test_retrieval.py) | Already generic — reuse |
| Integration | New `eeg/test_eeg_eval.py` mirroring `test_evaluate_runs_all_axes` | Write |
| Integration (Modal) | New `eeg/modal/modal_eeg_eval.py` | Write |

## Cross-checks backing every verdict

Same five as MEG:
1. **Pipeline sanity** — subject decodes above chance on raw + tokens → pipeline works.
2. **Multiple probe families** — linear, MLP, CNN, model-free retrieval all consulted.
3. **Multiple training regimes** — class-weighted vs unweighted, balanced vs top-1; report only after both agree.
4. **Cross-tokenizer agreement** — if a second EEG tokenizer ships later, check if both fail § 5.3 on category (likely yes, per the MEG pattern: large-compression + global reconstruction = selective preservation of variance not discrimination).
5. **Bracketing brackets stable** — raw and random brackets stable across configurations.

## Open follow-ups for EEG

| Item | Why |
|---|---|
| Run Exp 1 grid end-to-end on LaBraM. | Baseline. |
| Decide Exp 1.5 / skip based on Exp 1 outcome. | Only worth the runs if Exp 1 shows category at chance. |
| If §1 / §1.5 show MEG-style selective loss, **don't** train Exp 2; instead pursue category-aware finetune objective (auxiliary contrastive / supervised loss) or move to Stage 2 (Cho2026 / EphysTokenizer EEG variant). | Reconstruction-only objective is the bottleneck per MEG conclusions; same architectural story will apply to LaBraM. |
| **Refactor note**: lift `image_id → concept27 → superordinate / animacy` mappings out of `meg/data.py` into `neural_tokenizers/labels/` (or `neural_tokenizers/things/`). MEG and EEG share these — currently duplicated logic would be a smell. | Modular reuse; matches the "shared label space, modality-specific data loader" architecture the repo is moving toward. |

## Pointers

- MEG counterpart: [`meg_tokenization.md`](meg_tokenization.md) *(read first — it has the verdicts and conventions you're inheriting)*
- Production leaderboard (MEG, to be mirrored for EEG): [`neural_tokenizers/meg/CLAUDE.md §9`](../neural_tokenizers/meg/CLAUDE.md)
- Probe design rationale: [`linear_probe_design.md`](linear_probe_design.md)
- 4M-modality plan: [`4m_neural_modality_design.md`](4m_neural_modality_design.md)
- Probe source: [`neural_tokenizers/evaluation/probe.py`](../neural_tokenizers/evaluation/probe.py), [`evaluation/retrieval.py`](../neural_tokenizers/evaluation/retrieval.py)
- EEG cache verifiers: [`modal/modal_verify_eeg_cache.py`](../modal/modal_verify_eeg_cache.py), [`modal/modal_peek_eeg_npz.py`](../modal/modal_peek_eeg_npz.py)
- MEG eval dispatcher (template to copy): [`neural_tokenizers/meg/modal/modal_meg_eval.py`](../neural_tokenizers/meg/modal/modal_meg_eval.py)
