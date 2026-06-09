# Neural Tokenizers — CLAUDE.md

Scope: this file applies only to work in `neural_tokenizers/`. Read it before
making changes in this subdir. Repo-wide engineering norms live in the user's
global `~/.claude/CLAUDE.md` and are not restated here.

**Per-modality docs live in subfolders.** MEG-specific work goes in
[`meg/CLAUDE.md`](meg/CLAUDE.md); EEG-specific work goes in an `eeg/`
subfolder (to be created by the EEG owner). This file is the umbrella
contract — the `Tokenizer` protocol, the four §5 evaluation axes, the
file-layout norms — and it applies to every modality. Modality-specific
plans (data spec, recalibration recipes, vendored model choices) live in
the subfolder.

## 1. Goal

Build per-modality tokenizers for a mini-4M model. Each tokenizer maps raw
neural signals into discrete token IDs that the 4M transformer can consume.

Current target modalities, in priority order:
- **EEG** (THINGS-EEG2)
- **MEG** (THINGS-MEG)
- **Intracortical** (TVSD) — future, same interface

The output of this subdir is (a) trained tokenizer checkpoints and (b) a
modality-agnostic evaluation harness (`test_tokenizer.py`) that any tokenizer
must pass before it is wired into mini-4M.

## 2. System-level framing — why VQ-VAE per modality

4M's transformer only sees integer token sequences. To add a modality you must
provide three things:

- **Encoder** `x -> z` : continuous signal → continuous latents `(L, D)`
- **Quantizer** `z -> t` : continuous latents → discrete token IDs in `[0, V)`
- **Decoder** `t -> x_hat` : tokens → reconstructed signal (used for eval and
  detokenization at inference; not used during 4M training itself)

Why this is the right abstraction:
- A single masked-token objective works across every modality if every modality
  speaks the same `(seq_len, vocab_size)` interface.
- Tokens are pre-computed once and cached → 4M training does not pay the
  tokenizer's forward cost per step.
- The tokenizer can be swapped without touching 4M, so we can A/B different
  EEG tokenizers (scratch vs LABRAM, etc.) cleanly.

The 4M repo lives at [`../external/ml-4m/`](../external/ml-4m/). The relevant
entry points are:

- [`external/ml-4m/fourm/vq/vqvae.py`](../external/ml-4m/fourm/vq/vqvae.py) — `VQVAE` class
- [`external/ml-4m/fourm/vq/models/`](../external/ml-4m/fourm/vq/models/) — encoder/decoder architectures
- [`external/ml-4m/fourm/vq/quantizers/`](../external/ml-4m/fourm/vq/quantizers/) — quantizer implementations (vanilla, `memcodes`, FSQ, …)
- [`external/ml-4m/cfgs/default/tokenization/vqvae/human_poses/BMLP1024-BMLP1024_1k_224.yaml`](../external/ml-4m/cfgs/default/tokenization/vqvae/human_poses/BMLP1024-BMLP1024_1k_224.yaml) —
  closest template to ours: structured non-image signal, MLP backbone,
  `memcodes` quantizer, `codebook_size=1024`, `num_codebooks=8`,
  `smooth_l1` loss.
- [`external/ml-4m/run_training_vqvae.py`](../external/ml-4m/run_training_vqvae.py) — training entry point

When in doubt, mirror the human_poses config rather than the image configs.

## 3. Data — working assumptions

- **THINGS-EEG2** (Gifford et al. 2022): **verified** on the team `project`
  Modal Volume. Published preprocessed version selects **17 channels**
  (occipital + posterior only via regex `^O *|^P *` in
  `gifale95/eeg_encoding_model/02_eeg_preprocessing/preprocessing_utils.py`),
  100 Hz, MVNN-whitened, trial tensor shape `(17, 100)`. 10 subjects, 16,540
  training images × 4 reps + 200 test images × 80 reps. Raw recording has 63
  channels but Gifford drops the others before publishing.
- **THINGS-MEG** (Hebart et al. 2023): **verified** on the team `project`
  Modal Volume — see [`meg/CLAUDE.md`](meg/CLAUDE.md) §3 for exact shapes
  (271 mag channels, 200 Hz, 281 timepoints, 27,048 trials/subject × 4
  subjects). Preprocessed `.fif` files live at
  `/project/data/things-meg/preprocessed/`.
- Both: 2D `(channels, time)` per trial. Channels are *not* a spatial grid —
  treat as an unordered set or a sensor-topography graph, not as image rows.
- Stimulus labels: THINGS has ~1,854 object concepts (use for downstream probes);
  THINGS-MEG trigger codes are **image-level** (~22k), not concept-level.

Open data questions (EEG / TVSD):
- exact preprocessed shape of THINGS-EEG2 on our volume
- which preprocessing pipeline (raw vs. ICA-cleaned vs. baseline-corrected)
- where EEG data lives at training time (Modal Volume path)

## 4. Architecture — baseline vs target

This subdir trains tokenizers in **two explicit stages**. Stage 1 is a
deliberately simple baseline that we expect to be beaten; Stage 2 is the
actual production tokenizer that ships into 4M. Do not skip Stage 1 — the §5
harness only produces interpretable numbers if there is a baseline to
compare against.

### Stage 1 — Baseline (start here, all modalities)

- Backbone: `BottleneckMLP/B_6-Wi_1024` (encoder and decoder), inputs flattened
  to `(channels * time,)`. Simplest faithful backbone for a structured signal,
  mirrors 4M's existing non-image tokenizer (human poses).
- Loss: `smooth_l1` reconstruction in the time domain only. Single-term, easy
  to debug.
- Quantizer: `memcodes`, `codebook_size=1024`, `num_codebooks=8`,
  `latent_dim=1024`.

This baseline is **expected to fail the spectral-MSE metric** in §5.1 — that
failure is the motivation for Stage 2.  Get clean numbers across all four §5
axes for this baseline before touching anything.

### Stage 2 — Target (what actually ships into 4M)

Whatever the final tokenizer is, two upgrades over the baseline are
non-negotiable:

- **A stronger temporal-structure backbone than a flat MLP.** Neural responses
  are locally autocorrelated; an MLP on flattened `(C·T,)` ignores that prior.
  Either a 1-D CNN over the time axis or a temporal-patch transformer is the
  right shape, and either generalizes naturally to MEG / TVSD where `T` varies.
- **A spectral / band-power loss term**, added to `smooth_l1`. EEG/MEG task
  signal lives in frequency bands (θ/α/β/γ). Pure time-domain L1 can drive
  MSE down while still scrambling band power. Likely forms:
  - PSD MSE: `||PSD(x) - PSD(x_hat)||²` (Welch estimator)
  - Band-power MSE: integrate PSD over named bands and match per band
  - Multi-resolution STFT loss (borrowed from audio neural codecs)
  - Composite: `L = L1_time + λ_spec · L_spec + λ_band · L_band`, with `λ`
    chosen so each term contributes comparable gradient magnitude at init.

These two requirements are about *what the tokenizer must learn*, not about
which codebase produces it. Two independent decisions then determine the
actual Stage 2 we ship — architecture (whose design) and weights (whose data):

|                            | **Our own scratch design**       | **LABRAM's architecture**            |
|----------------------------|----------------------------------|--------------------------------------|
| **Trained from scratch**   | (2a) Generic, modality-agnostic  | (2b) Borrow design, not data         |
| **Pretrained + finetuned** | —                                | (2c) Borrow design AND data          |

- **(2a) Scratch design, scratch weights.** We design the upgraded backbone
  (e.g. 1-D ResNet/U-Net over time + spectral loss) ourselves. Same recipe
  works for EEG, MEG, and TVSD. Full control; no external dependency. Risk:
  we may rediscover what LABRAM already figured out about EEG.
- **(2b) LABRAM architecture, scratch weights.** Adopt LABRAM's
  temporal-patch transformer + channel embeddings + multi-objective loss
  *as a design template*, but train weights from scratch on THINGS data.
  We get the structural priors (patch-wise tokenization, channel-aware
  attention, frequency-domain loss already in the recipe) without inheriting
  any potential distribution mismatch from LABRAM's pretraining corpus
  (mostly resting-state EEG; THINGS is event-related). Cost: more complex
  than 2a, and we still have only ~160k trials — the data-efficiency argument
  for LABRAM disappears here.
- **(2c) LABRAM architecture, LABRAM weights, finetuned.** Full transfer:
  load LABRAM's pretrained tokenizer and finetune on THINGS-EEG. This is
  the only path that exploits the 2,500+ hrs of EEG LABRAM saw in
  pretraining. EEG-only — MEG and TVSD still need 2a or 2b.

Stage 2 selection is the open decision in §6. The §5 harness numbers — not
authorial preference — decide which one wins.

- Per modality: train one tokenizer at a time. Do *not* try to share encoders
  across EEG and MEG until both single-modality baselines are working.

## 5. Evaluation contract — what `test_tokenizer.py` must enforce

A tokenizer is a lossy compressor. `test_tokenizer.py` is the gate that any
tokenizer (scratch VQ-VAE, finetuned LABRAM, future TVSD tokenizer) must pass.

The harness consumes any object implementing this interface:

```python
class Tokenizer(Protocol):
    codebook_size: int

    def tokenize(self, x: Tensor) -> LongTensor: ...       # (B, C, T) -> (B, ...)
    def decode_tokens(self, tokens: LongTensor) -> Tensor: ...  # (B, ...) -> (B, C, T)
    # optional: tokens_to_embedding(tokens) -> (B, L, D) for probe features
```

Tokens may have any shape `(B, ...)`; the harness flattens to `(B, L)` where
needed. See [`evaluation/protocol.py`](evaluation/protocol.py).

Four required evaluation axes — implement one module per axis so they can be
run independently:

1. **Reconstruction fidelity** — encode → decode round trip.
   - Per-sample MSE and per-channel Pearson r between `x` and `x_hat`.
   - **Spectral MSE**: PSD of `x` vs `x_hat` (Welch). Neural data lives in
     bands; if alpha/beta power is wrong the tokenizer is broken even when
     time-domain MSE looks fine. This is the most diagnostic single metric.
2. **Codebook utilization**.
   - Perplexity `exp(H(p))` where `p` is the empirical token distribution.
     For `codebook_size=V`, perplexity ≪ V means dead codes.
   - Dead-code fraction (codes never selected on the eval set).
3. **Downstream linear probe** — does the token sequence still carry the
   information we actually care about?
   - *The failure mode this catches:* a tokenizer can have low reconstruction
     MSE (passes §5.1) and good codebook utilization (passes §5.2) while
     having silently thrown away the task-relevant signal. Example: the
     tokenizer faithfully reconstructs slow baseline drift (high-variance,
     drives MSE down) but loses the small evoked response that actually
     distinguishes one stimulus from another. Reconstruction looks great;
     the tokens are useless to 4M.
   - *The test:* freeze the trained tokenizer. Encode every THINGS trial to
     tokens. Train a **logistic regression** (linear probe) on those tokens
     to predict the trial's stimulus concept (THINGS has ~1,854 labels — use
     a coarser grouping or top-K if 1,854-way is too noisy). Report
     top-1 / top-5 accuracy on a held-out split.
   - *Why a linear probe and not a strong classifier:* if a linear model can
     read the stimulus off the tokens, the information is **linearly
     decodable** — which is what 4M's transformer needs from its input
     embeddings. A strong nonlinear classifier could rescue tokens that are
     entangled in ways 4M can't undo cheaply, so it would overstate
     tokenizer quality.
   - *Token representation for the probe:* either one-hot pool-then-flatten
     (count of each code per trial; bag-of-tokens), or embed each token via
     the VQ-VAE's codebook lookup and mean-pool over the sequence. Try
     both — they measure subtly different things (presence vs. average
     content).
   - *Reference points to bracket the score:* run the same probe on
     **(upper)** the raw flattened signal `x` itself (ceiling — all info is
     there) and on **(lower)** uniformly random tokens of the same shape
     (floor — no info). A good tokenizer should land closer to the upper
     bound than to the lower; if it sits near the floor, the tokens are not
     informative no matter how pretty the reconstruction looks.

4. **Token-sequence statistics** — is the token *sequence* learnable by a
   transformer, or did the tokenizer collapse to a degenerate code?
   - *The failure mode this catches:* a tokenizer can pass §5.1, §5.2, and
     §5.3 while emitting sequences so predictable that 4M's masked-token
     objective learns nothing. Example: suppose the tokenizer outputs an
     8-token sequence per trial, but for any given trial it emits *the same
     code 8 times* (e.g. `[42, 42, 42, ..., 42]`), with the specific code
     varying by stimulus class. The decoder can still reconstruct (mapping
     code → trial), the codebook is fully used (§5.2 sees 1,024 distinct
     trial-level codes), and the probe works perfectly (one code = one
     stimulus). But 4M is trained to predict masked tokens given context —
     if `t₂` is always equal to `t₁`, the prediction task is trivial and the
     model learns nothing transferable.
   - *What to compute:*
     - **Unigram entropy** `H(t) = -Σ p(t) log p(t)` — the entropy of the
       marginal token distribution. Already implied by perplexity in §5.2,
       but report it explicitly per-position too (some positions may collapse
       while others don't).
     - **Bigram conditional entropy** `H(tᵢ | tᵢ₋₁)`. This is the key number.
       If a transformer's next-token prediction has nothing left to learn
       after seeing one neighbor, this collapses toward 0. **Healthy
       tokenizers have `H(tᵢ | tᵢ₋₁) ≈ H(t)` — i.e. transitions are nearly
       independent of the previous token.** A large gap `H(t) - H(tᵢ | tᵢ₋₁)`
       means the sequence is highly redundant.
     - **Run-length distribution** — how long are runs of the same code?
       Long runs are the most visually obvious form of collapse. Report mean
       run length and the fraction of trials with run length ≥ 2.
   - *Pass criterion:* bigram-vs-unigram entropy gap small (say <20%), no
     position with degenerate marginal, mean run length near 1. If any of
     these fail, the tokenizer will technically train 4M to convergence but
     the resulting model will not generalize — fix the tokenizer first.

The harness should report all four for each tokenizer in a single table so we
can compare tokenizers side by side. Treat this as the source of truth for
"is tokenizer X better than tokenizer Y" — not training-loss curves.

## 6. Open architectural decisions

These are intentionally not resolved here. Record the rationale in the PR
when you pick one.

- **Stage 2 selection — which of 2a / 2b / 2c (see §4) wins for EEG?**
  - 2a (scratch design + scratch weights): uniform recipe across EEG / MEG /
    TVSD, drops into 4M with zero glue code, but THINGS-EEG has only ~16k
    trials per subject — small for a deep encoder we're designing from
    scratch.
  - 2b (LABRAM architecture + scratch weights): inherits LABRAM's structural
    priors for EEG (temporal-patch tokenization, channel embeddings,
    multi-objective loss) without inheriting any pretraining-distribution
    mismatch. Same architecture is reusable for MEG/TVSD if the channel
    embedding is generalized.
  - 2c (LABRAM architecture + pretrained weights, finetuned): only path that
    exploits LABRAM's 2,500+ hrs of EEG pretraining. EEG-only — MEG/TVSD
    still need 2a or 2b, so picking 2c means maintaining two codepaths
    long-term.
  - Decision blocker: we need Stage 1 baseline numbers on the §5 harness
    before we can rank the three Stage 2 options. Likely we run 2a and 2c
    against each other first (cleanest "design-only" vs "design + data"
    contrast) and only fall back to 2b if 2c shows distribution-mismatch
    symptoms (e.g. high reconstruction MSE despite the pretrained weights).
- **Codebook size / num_codebooks**: defaults from human_poses; sweep once the
  baseline runs.
- **Per-subject vs subject-pooled training**: pooled is simpler; per-subject
  may help if subject variance dominates.

## 7. Engineering norms specific to this subdir

- Three files per tokenizer: `<modality>_encoder.py`, `<modality>_quantizer.py`
  (usually a thin wrapper around a 4M quantizer), `<modality>_decoder.py`.
  Compose them in a `<modality>_tokenizer.py` that satisfies the `Tokenizer`
  protocol in §5. Mirrors 4M's own module layout.
- `test_tokenizer.py` is **tokenizer-agnostic** — it takes any object
  satisfying the protocol. Do not special-case EEG vs MEG inside it; put
  modality-specific quirks (PSD frequency ranges, label spaces) behind a
  small `EvalConfig` dataclass passed in.
- Write failing tests in `test_tokenizer.py` first, on a tiny random-weights
  tokenizer, before training anything. The harness needs to be trustworthy
  before its numbers mean anything.
- All training runs go through the Modal scaffold in [`../modal/`](../modal/),
  not local laptop. Checkpoints land on the shared Volume.
- No magic constants — codebook size, channel counts, sampling rates all
  named and centralized.

## 8. MEG progress (2026-05)

MEG-specific detail lives in [`meg/CLAUDE.md`](meg/CLAUDE.md). Current state:

| Phase | Tokenizer | Status |
|-------|-----------|--------|
| 1 | μ-transform (V=256, per-channel calibration) | **Done** — §5 harness on THINGS-MEG |
| 2 | Cho2026 learnable AE (EphysTokenizer) | Not started |
| 3 | BrainOmni BrainTokenizer (RVQ, finetuned) | **Done** — 3a zero-shot + 3b full finetune |
| 4 | Production token cache + `tok_meg/` shards aligned with RGB split | **Done (2026-05-23)** — 98,592 trials, 27 shards, 28/28 audit pass; see [`meg/CLAUDE.md`](meg/CLAUDE.md) §13 |

**Headline from §5 eval (n=3000 test, seed=0, identical split):** neither μ-transform
nor BrainOmni passes the linear-probe gate (~chance top-1 on object labels).
BrainOmni 3b wins on meaningful compression (512 tokens/trial vs ~76k) and
post-finetune reconstruction; μ-transform wins trivial round-trip fidelity.
See `meg/CLAUDE.md` §9 for the full metric table.

**Pending for 4M:** wire `meg_tokens_brainomni` in ml-4m `modality_info.py`
(draft in `meg/modality_registration.py`). Token cache + shard export is
shipped — see [`meg/CLAUDE.md`](meg/CLAUDE.md) §13.

## 9. Shared THINGS data layout on the `project` Volume (2026-05-23)

This section is cross-modality — every modality owner needs to read it. The
shared infrastructure is implemented under top-level [`../modal/`](../modal/),
NOT under `neural_tokenizers/<modality>/modal/`.

### 9.1 Layout

```
/project/data/
  things_catalog.json                # 26,107 image_id ↔ filename, split-invariant
  things_split.json                  # canonical train/val membership (v2 policy)
  things-meg/labels/meg_coverage.json
  things-eeg/labels/eeg_coverage.json   # optional alt path
  eeg_coverage.json                  # EEG1 ∩ EEG2 IDs (Volume root today)
  train/
    things_manifest.json             # legacy 85/15 shard layout (until repack)
    things_meg_manifest.json         # per-shard MEG entry counts (see meg/CLAUDE.md §13)
    things/
      rgb/shard_NNN.tar              # 23 shards (~85% of images)
      tok_meg/shard_NNN.tar          # 23 shards (BrainOmni 3b tokens)
      tok_eeg/                       # EEG owner's tokens (separate convention)
  val/
    things_manifest.json
    things_meg_manifest.json
    things/
      rgb/shard_NNN.tar              # 4 shards (~15%)
      tok_meg/shard_NNN.tar          # 4 shards
      tok_eeg/
```

### 9.2 Split JSON artifacts (v2, 2026-05)

**Canonical membership:** [`things_split.json`](../modal/data/things_split.json) on the
Volume (git mirror under `modal/data/`). Built by
[`modal_build_things_split.py`](../modal/modal_build_things_split.py) — **does not**
overwrite legacy `train/things_manifest.json` or shard tars.

| File | Role |
|------|------|
| `things_split.json` | `train_image_ids`, `val_image_ids`, `intersection_image_ids` (~16,718); val = 20% sample of intersection pool |
| `things-meg/labels/meg_coverage.json` | Unique catalog IDs with MEG trials (~22,448) |
| `eeg_coverage.json` | EEG1 / EEG2 separate counts; val pool uses EEG1 ∩ EEG2 |

Legacy manifests under `train/` and `val/` still reflect the original 85/15 repack.
Use `things_split.json` when reshuffling — derive shard maps from the ID lists
when repacking.

### 9.3 Conventions every modality must respect

- **`image_id`** = 9-digit zero-padded *alphabetical rank* of the THINGS image
  filename. `things_catalog.json` is the only source of truth — the same
  image_id refers to the same image forever, regardless of split.
- **Split (v2)** = image-level, seed=0, val_frac=0.20 of
  `catalog ∩ meg_coverage ∩ eeg_coverage`. Full ID lists in
  `things_split.json`. Legacy 85/15 manifests remain until an explicit repack
  consumes `things_split.json`.
- **Per-modality shard subfolder**: `things/<modality>/shard_NNN.tar`,
  using the **4M `tok_<modality>/` prefix** for tokenized modalities
  (`tok_meg`, `tok_eeg`, …) and the bare modality name for raw image data
  (`rgb`). 23 train + 4 val shards per modality, indexed identically to RGB.
- **WebDataset key convention** inside each tar:
  - one-sample-per-image modality → `<image_id>.<ext>` (RGB: `.jpg` + `.txt`)
  - multi-trial modality (MEG) → `<image_id>_<provenance>.<ext>`
    (e.g. `<image_id>_<subject>_t<trial_idx>.meg.npy`).
  - 4M loaders pair across modalities by stripping everything after the
    first `_` to recover image_id.

### 9.4 Source files (top-level `modal/`)

| File | Purpose |
|---|---|
| [`../modal/things_manifest.py`](../modal/things_manifest.py) | Pure-logic THINGS catalog + split + RGB shard repack |
| [`../modal/modal_build_things_split.py`](../modal/modal_build_things_split.py) | Build `things_split.json` + `meg_coverage.json` (JSON only, non-destructive) |
| [`../modal/modal_things_repack.py`](../modal/modal_things_repack.py) | Modal entrypoints: `plan` / `repack` / `verify` for the RGB train/val split |
| [`../modal/meg_token_shard.py`](../modal/meg_token_shard.py) | Pure-logic MEG shard packer (filename convention, planning, tar I/O) |
| [`../modal/modal_meg_pack_shards.py`](../modal/modal_meg_pack_shards.py) | Modal entrypoints: `plan` / `pack` / `verify` for MEG token shards |
| [`../modal/modal_meg_pipeline_audit.py`](../modal/modal_meg_pipeline_audit.py) | End-to-end 28-check audit (structural / consistency / integrity / protocol) |

### 9.5 Adding a new modality (EEG, fMRI, intracortical, …)

Mirror this layout. Write your tokens into `things/tok_<modality>/shard_NNN.tar`
for each split, using the **same shard indexing as the catalog manifests** —
read `things_split.json` (or legacy `train/things_manifest.json` until repack)
to know which image_ids belong in each shard. Don't invent your own split.

## 10. Notes on automation / `.claude` layering

This subdir is a candidate for a scoped `.claude/` later:
- A `train-tokenizer` skill or slash command that wraps the Modal launch + log
  tailing, once the recipe stabilizes.
- A pre-commit hook that runs the §5 harness on whatever checkpoint a PR
  touches, so we never merge a tokenizer regression.
- Neither is worth building until the scratch-baseline numbers exist —
  premature automation locks in the wrong recipe.
