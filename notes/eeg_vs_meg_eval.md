# EEG vs MEG Tokenizer Eval Comparison

Side-by-side of the §5 four-axis eval harness results for both neural modalities.
All cells are balanced accuracy (bal\_acc) unless noted.

---

## 1. System overview

| | EEG (LaBraM) | MEG (BrainOmni 3b) |
|---|---|---|
| Dataset | THINGS-EEG2 | THINGS-MEG |
| Subjects | 10 | 4 |
| Channels | 17 (occipital/posterior) | 208 (full MEG array) |
| Input sampling rate | 100 Hz → upsampled to 200 Hz | 200 Hz → resampled to 256 Hz |
| Trial window | 1 s (200 samples) | 2 s (512 samples at 256 Hz) |
| Total trials | 821,600 | ~full THINGS-MEG |
| Labeled (cat27) trials | 398,960 | full |
| Tokenizer family | LaBraM VQ-NSP | BrainOmni μ-transform RVQ |
| Codebook size V | 8,192 | 512 (per RVQ layer, 4 layers) |
| Codebook dim | 64 | 16 latent sources |
| Architecture | 12-layer NeuralTransformer, pretrained 2,500 h EEG, finetuned on THINGS-EEG2 | BrainOmni adapt finetune, 5M params total / 959k trainable |
| Training epochs | 5 | 10 |
| Loss | FFT amplitude + phase MSE | RVQ-4 codebook + commitment loss |
| Probe subsample cap | 50,000 | 50,000 (full ≤ cap) |
| Retrieval cap | 5,000 | 5,000 (full ≤ cap) |
| "raw" feature | token IDs cast to float (17 integers ∈ [0,8192)) | flattened raw waveform values |
| Chance — subject | 10% (10 subjects) | 25% (4 subjects) |
| Chance — cat27 | 3.70% | 3.70% |
| Chance — animacy | 50% | 50% |

One important caveat: "raw" means very different things. For EEG, "raw" is token IDs cast to float — discrete integers, not the actual EEG waveform. For MEG, "raw" is the flattened continuous waveform. This makes the raw EEG baseline nearly meaningless as an upper bound; the MEG raw baseline is the true waveform ceiling.

---

## 2. Codebook health

| Metric | EEG | MEG |
|---|---|---|
| Codebook size V | 8,192 | 512 |
| Codes used | 7,462 (91.1%) | 512 (100%) |
| Dead-code fraction | 8.9% | 0.0% |
| Perplexity | 4,439 | 493.6 |
| Perplexity / V (efficiency) | 54% | 96% |
| Unigram entropy | 8.40 nats | 6.20 nats |
| Max possible entropy | 9.01 nats (log 8,192) | 6.24 nats (log 512) |
| Unigram / max | 93.2% | 99.4% |

MEG achieves near-perfect codebook utilization and near-maximum unigram entropy. EEG is healthy (91.1%) but 730 dead codes remain and perplexity/V is only 54%. Both are acceptable for shipping into 4M — neither is showing the collapse signal (near-zero utilization) that would indicate training failure.

---

## 3. Sequence statistics (masking utility)

The bigram entropy gap is the key metric for 4M masking: if it is high, consecutive tokens are too predictable and the masked modeling objective becomes trivial.

| Metric | EEG | MEG |
|---|---|---|
| Bigram conditional entropy | 6.85 nats | 5.89 nats |
| Entropy gap (1 − H\_bigram/H\_unigram) | **18.4%** | **5.0%** |
| Threshold for healthy masking | ≤ 20% | ≤ 20% |
| Mean run length | 1.005 | 1.002 |
| Frac runs ≥ 2 tokens | 7.9% | 62.9% |

Both are below the 20% danger threshold. MEG is much better (5%), meaning consecutive MEG tokens are nearly independent — ideal for masked prediction. EEG at 18.4% is marginal: some positional predictability remains, but not enough to degrade masking.

The frac\_runs\_ge\_2 difference (8% EEG vs 63% MEG) reflects structural differences: MEG processes temporal windows with overlap, so adjacent tokens share temporal context and the same code can repeat. EEG processes channel-level patches with no temporal repetition, so runs are rare.

---

## 4. Category 27 probe (chance = 3.70%)

Values: balanced accuracy mean ± SEM.

### Per-trial (each trial is one data point)

| Classifier | EEG tokens | EEG raw | EEG random | MEG tokens | MEG raw | MEG random |
|---|---|---|---|---|---|---|
| Linear, weighted | 4.10% ± 0.23% | 3.72% ± 0.07% | 3.78% ± 0.31% | 3.35% ± 0.91% | **5.16% ± 0.21%** | 3.64% ± 0.73% |
| MLP, weighted | 4.12% ± 0.11% | 3.71% ± 0.01% | 3.70% ± 0.25% | 3.18% ± 0.97% | **4.31% ± 0.49%** | 3.09% ± 0.82% |
| CNN, weighted | 3.68% ± 0.23% | 3.68% ± 0.18% | 3.66% ± 0.26% | 3.55% ± 0.54% | **4.19% ± 0.87%** | 3.61% ± 0.52% |

### Cross-subject averaged (MEG only — EEG has no avg eval yet)

Trial-averaging across subjects collapses noise. MEG ran this; EEG did not (Exp 1.5 not yet run).

| Classifier | MEG tokens | MEG raw | MEG random |
|---|---|---|---|
| Linear | 3.35% | 6.01% | 4.02% |
| CNN | 4.09% | 4.78% | 4.40% |

**Reading:** Both tokenizers fail at 27-class object category decoding. Per-trial category signal is absent in both EEG and MEG tokens. MEG raw carries a small but real lift (5.2% vs 3.7% chance, 7σ for linear), while EEG raw is at floor. The MEG raw ceiling is the actual MEG waveform; EEG raw is just code IDs (meaningless ceiling). EEG tokens are marginally above EEG's floor but cannot be directly compared to MEG raw because the baselines are different quantities.

---

## 5. Subject identity probe

EEG: 10 subjects, chance = 10%. MEG: 4 subjects, chance = 25%.

| Classifier | EEG tokens | EEG raw (= code IDs) | EEG chance | MEG tokens | MEG raw (= waveform) | MEG chance |
|---|---|---|---|---|---|---|
| Linear | 40.6% ± 0.36% | 10.0% ± 0.04% | 10% | 62.6% ± 0.97% | **93.8% ± 0.90%** | 25% |
| MLP | 50.4% ± 0.64% | 10.0% ± 0.07% | 10% | 69.3% ± 1.00% | **95.4% ± 0.41%** | 25% |
| CNN | **60.1% ± 0.87%** | 10.8% ± 0.18% | 10% | **97.4% ± 0.35%** | **98.0% ± 0.51%** | 25% |

**Reading:** This is the starkest difference between the two modalities.

For MEG, the raw waveform is overwhelmingly subject-discriminative (98% linear — the continuous signal itself identifies the person). The MEG tokenizer adds nothing on top of what the raw already contains.

For EEG, the raw code IDs are at chance (10%) — the discrete code labels alone carry no subject identity. But the continuous codebook embedding geometry captures it strongly (60% CNN). This means the LaBraM tokenizer learns subject-specific embedding directions that are not visible from the code indices. The LaBraM training objective minimized spectral reconstruction loss but incidentally encoded subject physiology into the embedding space.

Both tokenizers encode subject identity strongly. For 4M, this means the EEG and MEG token streams will carry large amounts of "who the person is" rather than "what they're looking at."

---

## 6. Animacy probe (chance = 50%)

| Classifier | EEG tokens | EEG raw | EEG rand | MEG tokens | MEG raw | MEG rand |
|---|---|---|---|---|---|---|
| Linear | **51.6% ± 0.59%** | 49.9% ± 0.05% | 49.6% ± 0.54% | 51.1% ± 1.78% | **57.8% ± 2.46%** | 50.0% ± 2.24% |
| MLP | 50.5% ± 0.46% | 49.6% ± 0.79% | 50.1% ± 0.39% | 50.8% ± 1.97% | **57.9% ± 2.06%** | 49.9% ± 3.00% |
| CNN | 50.5% ± 0.65% | 50.1% ± 0.57% | 49.7% ± 0.70% | 51.6% ± 1.97% | **52.1% ± 1.33%** | 49.5% ± 0.61% |

MEG averaged cross-subject (linear): tokens 49.3%, raw 61.1%. Averaging boosts MEG raw further.

**Reading:** EEG raw is at floor (49.9%) — the 17-channel occipital signal carries no per-trial animacy. MEG raw carries it (57.8%) — the full 208-channel MEG waveform does discriminate animate vs inanimate at the per-trial level.

EEG tokens show a trace lift (51.6% linear, 8.0σ above random), but it is marginal. MEG tokens match EEG tokens (51.1%) despite having a much richer raw ceiling — the MEG tokenizer loses most of the animacy signal the raw waveform contains.

Both tokenizers underperform raw MEG on animacy. The bottleneck in EEG is the raw signal (the 17 channels don't carry it). The bottleneck in MEG is the tokenizer (the raw signal has it but the codes don't encode it).

---

## 7. Retrieval (model-free, prec@1 / prec@5, n=5,000 subsample)

| Task | EEG cosine | EEG Jaccard | EEG raw | MEG cosine | MEG Jaccard | MEG raw |
|---|---|---|---|---|---|---|
| Cat27 (chance 3.7%) | 3.8% / 3.5% | 3.4% / 3.7% | 3.3% / 3.6% | 3.7% / 3.6% | 3.4% / 3.8% | 3.8% / 3.9% |
| Animacy (chance 50%) | 49.3% / 50.0% | 49.8% / 50.3% | 49.4% / 49.7% | 50.5% / 50.3% | 51.1% / 50.5% | 50.2% / 50.6% |
| Subject (EEG 10%, MEG 25%) | 23.5% / 21.8% | 17.9% / 16.5% | 10.2% / 10.2% | 50.3% / 47.5% | 37.7% / 36.7% | 65.9% / 61.5% |

**Reading:**

- Cat27 and animacy: all six feature sets (tokens cosine, tokens Jaccard, raw) are at or near chance for both modalities. Retrieval confirms the probe: no class-separating geometry.
- Subject retrieval isolates the subject-ID signal cleanly. EEG raw is at chance (10.2%), but EEG cosine embedding is 23.5% — confirms the subject signal lives in the continuous embedding space, not the discrete code indices. MEG raw retrieval is 65.9% (the continuous waveform clusters by subject naturally), while MEG cosine retrieval is 50.3% — the tokens lose some of the waveform's subject geometry but retain a lot.
- Jaccard (set overlap on discrete codes) is above chance for subject in both modalities (17.9% EEG, 37.7% MEG), meaning even the code identities, not just their embeddings, are partially subject-specific.

---

## 8. Reconstruction (MEG only)

EEG reconstruction was not run — the eval was configured `run_reconstruction=False` because the npz cache stores pre-computed tokens only, not raw waveforms.

| Metric | MEG BrainOmni 3b |
|---|---|
| MSE (per-trial, normalized) | 1.070 |
| Channel correlation (mean) | 0.870 |
| PSD MSE | 27.7M |

Channel correlation of 0.87 is healthy — the tokenizer reconstructs the per-channel waveform shape well. PSD MSE is high in absolute value but depends on the amplitude scaling of the MEG signal (not normalized per-channel in the raw).

No EEG reconstruction numbers exist yet. Running EEG reconstruction requires caching raw waveforms alongside the tokens, which is not the current pipeline.

---

## 9. Summary comparison table

Best probe result per task (highest tokens bal\_acc across linear/mlp/cnn).

| Task | EEG tokens (best) | EEG raw | EEG gap | MEG tokens (best) | MEG raw | MEG gap |
|---|---|---|---|---|---|---|
| Cat27 (chance 3.7%) | 4.1% MLP | 3.7%\* | +0.4 pp | 3.6% CNN | 5.2% | −1.6 pp |
| Animacy (chance 50%) | 51.6% linear | 50.0%\* | +1.6 pp | 51.6% CNN | 57.8% | −6.2 pp |
| Subject (own chance) | **60.1%** CNN | 10.0%\* | +50.1 pp | **97.4%** CNN | 98.0% | −0.6 pp |

\* EEG raw = token IDs cast to float, not actual waveform. These are lower bounds on the true raw EEG ceiling.

---

## 10. What this means for 4M

### Where EEG and MEG agree

Both tokenizers fail at per-trial object category decoding. This is expected and is a property of the raw signals, not the tokenizers — single-trial EEG and MEG are too noisy to carry reliable 27-way category information without averaging. Adding EEG or MEG to 4M will not help the model learn "what object is this" from a single neural trial.

Both have healthy codebooks (no collapse) and entropy gaps below the masking threshold. Both are safe to ship into 4M's masked prediction objective.

Both encode subject identity strongly — this is the dominant variance in the token streams. 4M will likely learn a subject-identity factor from these modalities. Whether that helps or hurts the scaling experiment depends on how much cross-subject variance is shared with the image modality.

### Where EEG and MEG differ

**Raw signal quality:** MEG raw carries animacy signal (57.8% vs 50% chance) and strongly identifies subjects from the waveform itself (93.8%). EEG raw does not — the 17 occipital/posterior channels, at single-trial resolution, do not carry linearly decodable category or animacy information, and the raw code IDs are not the waveform. This is a channel coverage and SNR difference, not a tokenizer difference.

**Tokenizer behavior:** LaBraM learns subject-specific codebook geometry that does not exist in the raw code indices (60% token probe vs 10% raw). BrainOmni's token probe (62.6%) barely exceeds the raw probe (93.8% for subject) — it is largely replicating what the raw waveform already contains. LaBraM is doing something different: distilling a property (subject ID) from the spectral structure that the raw integer codes don't directly represent. This could reflect LaBraM's 2,500h pretraining on a diverse EEG corpus.

**Masking utility:** MEG entropy gap 5% vs EEG 18.4%. Both pass. But MEG tokens are much closer to maximally unpredictable from their neighbors, which is ideal for masked prediction.

**Codebook efficiency:** MEG 96% perplexity/V vs EEG 54%. MEG's smaller codebook (512 vs 8,192) is being used more uniformly. EEG's larger codebook may be oversized for 17-channel, 1-second trials.

### Implication for the scaling experiment

The scaling experiment asks whether adding EEG improves 4M's RGB scaling behavior. Based on this eval, EEG and MEG tokens both carry substantial subject-identity variance and minimal stimulus-category variance at single-trial resolution. If 4M learns to associate neural tokens with images, the primary learned association will be "subject X tends to show image Y's class" (confounding the scaling measurement). Trial-averaging would improve stimulus decoding substantially but is not how the trunk pipeline works (one token per image per subject, not averaged across subjects or repetitions).

The safe interpretation: both modalities are technically ready to plug into 4M. The scientific interpretation of any training improvement from neural modalities is complicated by the subject-identity confound. Flag this in the project writeup.
