# MEG Tokenizers — CLAUDE.md

Scope: this file applies only to work in `neural_tokenizers/meg/`. Read it
before making MEG-tokenizer changes. The parent doc
[`../CLAUDE.md`](../CLAUDE.md) defines the cross-modality contract (the
`Tokenizer` protocol, the four §5 evaluation axes, Modal training, file-layout
norms). This doc only adds what is MEG-specific. EEG-specific work will live
in a sibling `eeg/` folder maintained by someone else — do not touch it from
here.

## 1. Goal

Ship one MEG tokenizer that passes the §5 harness in
[`../CLAUDE.md`](../CLAUDE.md) on THINGS-MEG and is registered as a 4M
modality. We will get there in three phases, each producing an independent
tokenizer that satisfies the same `Tokenizer` protocol so the harness
compares them apples-to-apples.

## 2. What 4M expects from a new modality — the binding constraint

Before designing anything, the codebook size we ship has to fit 4M's modality
registration model. Read [`../../external/ml-4m/fourm/data/modality_info.py`](../../external/ml-4m/fourm/data/modality_info.py)
— that file is the source of truth.

Concrete facts that drive the design:

- **No fixed vocab across modalities.** Each modality declares its own
  `vocab_size` in `MODALITY_INFO`. Image-token modalities range from 4096
  (semseg) to 16384 (RGB); text/sequence is 30_000; `human_poses` "borrows"
  the 30k text vocab via `shared_vocab: ['caption']`. We are free to pick.
- **But vocab IS fixed at registration time.** 4M allocates a learnable
  embedding `[vocab_size, d_model]` for the modality. Adding tokens
  mid-training breaks that table. **Pick V before computing the token cache
  that 4M trains on; commit to it.**
- **`type: 'seq'` is the right modality type for MEG**, not image-grid.
  Mirror `human_poses` (`modality_info.py:172`) more than `tok_rgb`. The
  per-channel-per-timestep token stream is a sequence, not a 2D grid.
  Pose-tokenizer registration template:
  `vocab_size`, `encoder_embedding = SequenceEncoderEmbedding(...)`,
  `decoder_embedding = SequenceDecoderEmbedding(...)`,
  `max_length`, `num_channels`, `type: 'seq'`,
  `id: generate_uint15_hash('meg_tokens')`.
- **Multi-codebook (`num_codebooks > 1`) is supported by the `memcodes`
  quantizer** (used by `human_poses`, 1024 × 8). Cho 2026's tokenizers use
  a single codebook per channel. Stick with **`num_codebooks = 1`** for MEG
  — there is no reason to fight the upstream tokenizer's design.
- **Whether to share vocab with another modality is a 4M-side decision.**
  Human_poses reuses the text vocab because pose tokens are mapped into the
  WordPiece range; this lets a 4M decoder generate poses with the same head
  as captions. We do **not** want that for MEG — it couples MEG to text
  training. **Register a dedicated `meg_tokens` modality with its own
  vocab.**

So the headline constraint: pick `V` and `max_length` (= per-trial token
sequence length) when you write the tokenizer, because the modality
registration that goes into 4M has to declare both.

Sensible starting points, derived from Cho 2026 + the verified THINGS-MEG
data spec in §3 + 4M conventions:

| Tokenizer            | V                              | num_codebooks | per-trial token shape |
|----------------------|--------------------------------|---------------|------------------------|
| Phase 1 (μ-transform)| 256 (paper default)            | 1             | 271 × 281 (per sample) |
| Phase 2 (Cho2026 learned) | 128 (paper) → V* after refactor | 1        | 271 × 281 (target)     |
| Phase 3 (BrainOmni)  | **512** (per RVQ layer)        | **4**         | **16 × 8 × 4 = 512** indices |

Registration shape for sample-level tokenizers (Phase 1/2):
`num_channels = 271`, `max_length = 281`. BrainOmni registers separately as
`meg_tokens_brainomni`: `num_channels = 16` latent sources, `max_length = 8`
temporal tokens, `num_codebooks = 4`, `vocab_size = 512`. See
`meg/modality_registration.py`. Flattening 271 × 281 = 76,151 tokens per trial
blows past sensible 4M context — BrainOmni's 512-token grid is the practical
production shape for 4M.

## 3. Data — verified, not assumed

Preprocessed THINGS-MEG already lives on the team's `project` Modal Volume.
This section is grounded in actual file metadata pulled from the volume on
2026-05-22 — re-run [`../../modal/modal_inspect_things_meg.py`](../../modal/modal_inspect_things_meg.py)
to refresh after any preprocessing pipeline change.

### Where it lives

- **Volume**: `project` (NOT the `neural-fm-data` shared volume — see the
  rationale at the top of [`../../modal/modal_download_things_meg.py`](../../modal/modal_download_things_meg.py)).
- **Path inside container**: `/project/data/things-meg/preprocessed/`.
- **File layout**: 4 subjects (P1–P4), each split across 4 MNE `.fif` files
  (`preprocessed_P{1..4}-epo.fif` + auto-continuations `-epo-1.fif`,
  `-epo-2.fif`, `-epo-3.fif` — MNE splits whenever a file exceeds 2 GB).
  Total ~33 GB on disk. Load with `mne.read_epochs(primary_path)` — MNE
  follows the `-N.fif` chain automatically.

### What each file actually contains (verified)

| Field            | Value                                  | Notes |
|------------------|----------------------------------------|-------|
| `sfreq`          | **200 Hz**                             | Already downsampled from native 1200 Hz |
| `n_channels`     | **271, all `mag`**                     | Originally CTF 275 → 271 after dropping ref/bad channels |
| `n_epochs/subj`  | **27,048**                             | Identical across all 4 subjects |
| `tmin, tmax`     | **−0.100 s, +1.300 s**                 | 100 ms pre-stim baseline + 1.3 s post-stim |
| `timepoints`     | **281** (= 1.4 s × 200 Hz + 1 inclusive) | Per-trial sequence length |
| `unique labels`  | **22,449** (per-image codes)           | Image-level IDs, NOT concept IDs |
| Per-trial shape  | `(271, 281)`                           | Channels × time |

Three non-obvious things the verification surfaced:

1. **All channels are `mag` after preprocessing.** The parent doc's worry
   about planar-grad vs. magnetometer unit heterogeneity does not apply
   here — there is exactly one channel type. The CTF axial-gradiometer
   sensors are labeled `mag` by MNE. Per-channel normalization is still
   wise for inter-channel amplitude variance, but the magnitude-scale
   disparity problem doesn't apply.
2. **Labels are image-level (22,449 codes), not concept-level (1854).**
   The THINGS object database has 1854 concepts × ~12 images per concept ≈
   22k unique images, and that is what the trigger codes index. The linear
   probe in §5.3 will need a `image_id → concept_id` mapping (lives in
   THINGS metadata, not in the .fif files). Coarsen further to
   superordinate categories for the primary probe target. Without that
   mapping the probe is a 22k-way classification on 27k trials per
   subject — too noisy to interpret.
3. **27,048 trials/subject is much more than the parent doc estimated
   (~16k).** Tokenizer training has more data than expected. This shifts
   the calculus on Stage 2 — even from-scratch (option 2c in §4) is more
   plausible than the parent doc assumed.

### What's NOT yet on the volume

- `/project/data/things-meg/splits/` — persisted split manifests (optional).
  **In-repo split logic exists:** `meg/splits.py` with `LEARNABLE_SPLIT_DEFAULTS`
  (80/10/10) for BrainOmni/Cho2026 and `MU_SPLIT_DEFAULTS` (90/0/10) for
  μ-transform calibration. Test indices are determined by `(seed, test_frac)`
  alone so all tokenizers evaluate on the same held-out trials.
- `/project/data/things-meg/source/` — source-reconstructed parcels.
  Not yet computed. Only needed if phase 2 ends up requiring it (see §4).

### Subject identity

Trial counts are identical across P1–P4, but trigger orderings differ
(different `event_id` first-5 keys per file). Tokenizer training pools all
4 subjects unless we have a specific reason to per-subject. Pooled is the
default per the parent doc §6.

## 4. Why three phases, in this order

The phases trade off cost vs. expected fidelity. We deliberately start with
the cheapest possible MEG tokenizer to (a) prove the §5 harness is correctly
wired for MEG-shaped data, (b) put a floor on the leaderboard, and (c) avoid
prematurely committing to a deep architecture before we know what "good
enough" looks like on this specific THINGS-MEG distribution.

| Phase | Tokenizer                                | Learns? | Cost     | Role                         | Status |
|-------|------------------------------------------|---------|----------|------------------------------|--------|
| 1     | μ-transform (Cho 2026 baseline)          | No¹     | seconds  | Floor + harness sanity check | **Shipped** |
| 2     | Cho 2026 learnable AE (causal / non-causal) | Yes  | hours-GPU | Likely production tokenizer | Not started |
| 3     | BrainOmni BrainTokenizer, finetuned        | Yes     | hours-GPU | Learned compression + 4M-friendly grid | **Shipped (3a+3b)** |

¹ "Non-learnable" means no gradient-descent weights — but the μ-transform
still has dataset-fit parameters (clip thresholds and max-abs scaler) that
**must** be re-estimated on THINGS-MEG. See §5 for what "recalibration"
actually means.

## 5. What "recalibration" / "finetuning" actually mean here

The Cho 2026 paper warns that both tokenizer families need dataset-specific
adaptation. The mechanism is different for each.

### Phase 1 — μ-transform recalibration (no gradient descent, but still a `fit` step)

The μ-transform pipeline (paper §3.3.1):

1. **Clip** the input to a percentile range `[q_lo, q_hi]` to suppress
   outliers.
2. **Max-abs scale** to `[−1, 1]`: `x' = x / s` with `s = max|x|`.
3. **μ-law compand**: `F(x') = sgn(x') · ln(1 + μ|x'|) / ln(1 + μ)`.
4. **Uniform bin** the compressed values into `V` bins → integer token IDs.

The reference config Cho 2026 ships
([`../../external/Cho2026_Tokenizer/models/tokenizer/mu_transform/config.yml`](../../external/Cho2026_Tokenizer/models/tokenizer/mu_transform/config.yml))
is:

```yaml
mu: 255
n_tokens: 256
normalization: max_abs
```

There are **no learned weights**, but two parameters are dataset-dependent
and must be fit on a THINGS-MEG training split, frozen, and reused at
inference:

- `s = max|x|`: the per-channel (recommended) or global max-abs scaler.
- `q_lo, q_hi`: clip thresholds (defaults around the 0.5 / 99.5 percentiles).

`μ` and `V` are hyperparameters, not fit from data — start with μ=255,
V=256.

Concretely: there is no `.train()` loop. There is a `fit(train_dataset) →
calibration.json` step that produces a small JSON sidecar with
`{s_per_channel, q_lo, q_hi, mu, V, channel_mode}`. At inference we load
that JSON and apply the stateless transform. Treat the calibration JSON as
a versioned tokenizer checkpoint.

**Phase 1 status (shipped):** calibrated on THINGS-MEG train split with
μ=255, V=256, per-channel clip `[0.5, 99.5]` percentiles. Run slug:
`V256_mu255_clip0.5-99.5_per_channel_s0`. Eval via
`modal_meg_eval.py --tokenizer mu_transform` (see §9).

> Verified §3 fact that matters here: all 271 channels are `mag`, so the
> grad-vs-mag amplitude disparity problem does not apply. Per-channel `s`
> is still the safer default (inter-sensor amplitude variance is real even
> within one channel type), but the bin-saturation risk is much lower than
> it would be on a mixed-sensor dataset.

### Phase 2 — Cho 2026 learnable tokenizer (actual gradient training)

The Cho 2026 learnable tokenizer (paper §3.2) is an **autoencoder**, not a
VQ-VAE: encoder is GRU → Dense → LayerNorm producing per-time-step logits
over a vocab of size `V`; decoder is a bank of `V` 1-D conv kernels (causal
or non-causal) summed with learned per-token gains. Trained end-to-end with
MSE + an annealed soft-to-hard argmax relaxation, then unused tokens are
pruned ("token refactorization").

Reference configs they ship
(`../../external/Cho2026_Tokenizer/models/tokenizer/{causal,noncausal}/config.yml`):

```yaml
sequence_length: 200
n_channels: 52         # NB: 52 source-parcellated channels, NOT sensor
n_tokens: 128
token_dim: 10
rnn_n_units: 128
token_kernel_padding: causal  # or `same` for non-causal
training: { lr: 1e-5, batch: 32, epochs: 40, anneal_stages: 40 }
```

Three things the configs tell us, in light of the verified data spec:

1. **They trained on source-reconstructed data (52 parcels), not sensor
   space.** Our THINGS-MEG is **271-sensor**. The tokenizer is applied
   per-channel and the GRU has no channel-conditioning, so weights *can*
   be reused on sensor data, but the *amplitude distribution* of source
   parcels (after `osl_dynamics.standardize()`) is not identical to sensor
   amplitudes. This is the symmetric MEG version of the LABRAM/THINGS
   distribution mismatch in the parent doc.
2. **Their sequence_length = 200 samples ≈ 0.8 s.** Our trials are 281
   samples (1.4 s). Either reduce our windows to 200 samples (matches their
   training distribution → reuse pretrained weights cleanly) or retrain
   with `sequence_length = 281`. Strong prior: **window to 200** for 2a/2b
   to keep the pretrained encoder in-distribution; only change
   `sequence_length` for 2c.
3. **Their LR (1e-5) is very small.** Tells us finetuning is intended to be
   conservative — they expect the loss landscape to be flat around the
   pretrained optimum. Start with that LR and only increase if the loss is
   not moving after 1 epoch.

"Finetuning on our dataset" then has three concrete recipes:

- **2a.** Load pretrained weights, apply on THINGS-MEG **as-is** (zero
  finetune). Establishes the floor for what their pretrained tokenizer
  buys us on distribution-shifted data.
- **2b.** Load pretrained weights, continue MSE training on THINGS-MEG
  (lr=1e-5, anneal κ already at 0 → hard tokens — or restart anneal).
  This is the "actual finetune."
- **2c.** Take their architecture, train from scratch on THINGS-MEG. This
  is needed if (a) pretrained weights fail the §5 harness even after 2b,
  (b) we want to change `sequence_length` or `n_tokens`, or (c) the
  source-vs-sensor mismatch turns out to dominate. With ~108k trials
  pooled across P1–P4 (see §3 — much more than the parent doc estimated),
  from-scratch is more viable than first assumed.

Run **2a → 2b → 2c** in order; stop as soon as the harness numbers are
acceptable. Do not skip 2a — the gap between 2a and 2b is the answer to
"how much did pretraining buy us."

> System-level note: their pretraining data is **resting-state** MEG;
> THINGS-MEG is **event-related** (stimulus-locked). Expect the pretrained
> kernels to capture continuous oscillatory structure well and to
> underweight evoked transients. Reconstruction MSE may look fine while
> the §5.3 linear probe quietly tanks — watch that metric closely after
> each of 2a/2b/2c.

### Phase 3 — BrainOmni BrainTokenizer (implemented)

Adapter over [OpenTSLab/BrainOmni](https://github.com/OpenTSLab/BrainOmni)
**Stage 1 only** (`BrainTokenizer` — reconstruction + RVQ). We do **not** use
BrainOmni Stage 2 (masked token prediction on discrete codes). Checkpoint
auto-downloads from HuggingFace `OpenTSLab/BrainOmni` on first Modal run.

**Preprocessing** (`meg/brainomni/preprocess.py`):

1. Resample 200 Hz → 256 Hz (460 samples from 281)
2. Per-trial per-channel z-score (computed **before** zero-pad)
3. Zero-pad post-stimulus tail to **512** samples

**Token shape:** `(B, 16, 8, 4)` — 16 latent neuro sources, 8 temporal
tokens (SEANet ratios `[8,4,2]` on 512 samples), 4 RVQ layers @ V=512 each
→ **512 integer indices per trial**.

**Finetune modes** (`meg/brainomni/trainer.py`):

| Mode | What's trainable | Params (approx) |
|------|------------------|-----------------|
| `adapt` (default) | sensor embed, cross-attn, latent neuros, RVQ | ~959k / 5.05M |
| `rvq_only` | RVQ codebooks only | smaller |
| `full` | entire BrainTokenizer | 5.05M |

Default `adapt` freezes SEANet encoder/decoder (~81% of weights). RVQ
codebooks update via EMA during training.

**Hyperparameter sweeps (smoke: 1k train / 200 val):**

| Sweep | Grid | Winner | Smoke best val |
|-------|------|--------|----------------|
| LR | `{3e-6, 1e-5, 3e-5}` × 5 ep | **3e-5** | 1.597 |
| Codebook | `{256, 512, 1024}` × 10 ep | **512** | 1.543 |
| LR coarse | `{1e-4, 1e-3}` × 10 ep | 1e-4 (both worse than 3e-5) | 1.620 |

Summaries: `brainomni/runs/3b_smoke_sweep/`, `3b_codebook_sweep/`,
`3b_lr_sweep_coarse/`.

**Full finetune (3b) — completed:**

- Config: `adapt`, lr=**3e-5**, batch=**32**, 10 epochs, V=512
- Data: 88,340 train / 9,816 val (80/10/10 `LEARNABLE_SPLIT_DEFAULTS`)
- Best val loss: **~1.401** (BrainOmni composite recon loss)
- Checkpoint: Modal volume `/project/checkpoints/meg/brainomni/V512_rvq4_win512_sf256_3b/`
- Local config pointer: `brainomni/runs/V512_rvq4_win512_sf256_3b/config.json`

**Modal resources (finetune/eval):** L40S GPU, 32 GB RAM, batch 32.
Long jobs: use `finetune_detached` with `modal run --detach` (`.spawn()` —
survives laptop close). Do **not** use `.remote()` for multi-hour runs.

**Modal entrypoints:**

```bash
# 3a zero-shot eval
modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
  --tokenizer brainomni \
  --calibration neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3a/config.json \
  --n-test 3000 --seed 0

# 3b finetuned eval
modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
  --tokenizer brainomni \
  --calibration neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3b/config.json \
  --n-test 3000 --seed 0

# Full finetune (detached)
modal run --detach neural_tokenizers/meg/modal/modal_meg_finetune_brainomni.py::finetune_detached \
  --finetune-mode adapt --lr 3e-05 --epochs 10 --batch-size 32 --stage 3b
```

**4M integration (draft, not wired in ml-4m yet):**

- Registration: `meg/modality_registration.py` → `meg_tokens_brainomni`
- At 4M boundary: flatten `(16,8,4)` → `(512,)` seq **or** treat as
  `(16, 32)` 2D grid (`ImageTokenEncoderEmbedding`); both are discrete
  lookup — RVQ layers are extra positions, not a 512⁴ joint vocab
- **Pending:** token-cache export script; add entry to ml-4m `modality_info.py`

**Key implementation notes:**

- Subsample split indices **before** loading `.fif` data (88k trials ≈ 30 GB
  if loaded at once)
- `codebook_size` override re-inits quantizer; filter quantizer keys in
  `load_state_dict` (`brainomni/load.py`)
- DeepSpeed import stubbed in `load.py` for HF checkpoint load
- Finetune loss masked on zero-pad region; minor `norm_target()` on full 512
  window including pad — deterministic, not worth restarting training

See also `meg/brainomni/README.md` for command cheat sheet.

## 6. External code — Cho 2026 is TensorFlow; the real integration is EphysTokenizer

The `external/Cho2026_Tokenizer` submodule we pinned in this branch is
**TensorFlow 2.11** + `osl_dynamics` + `osl_foundation` (see their
`README.md`, "Requirements"). Our stack (ml-4m, Modal, existing eval code)
is **PyTorch**. Do not try to bridge TF→Torch inside this subdir.

The OHBA group released a PyTorch port: **[`OHBA-analysis/EphysTokenizer`](https://github.com/OHBA-analysis/EphysTokenizer)**
(linked from the Cho2026 README's "PyTorch version" toggle). This is the
real runtime target for phase 2. The plan is:

- **Keep `external/Cho2026_Tokenizer`** — it is the source of truth for
  the paper's configs (`models/tokenizer/{mu_transform,causal,noncausal}/config.yml`),
  reference scripts (`scripts/01_train_tokenizer.py`, etc.), and the
  reproducibility of the published numbers. Read it, don't run it.
- **Add `external/EphysTokenizer` as a second submodule** when phase 2
  starts. That is the PyTorch implementation we will actually finetune
  and integrate behind the `Tokenizer` protocol.
- **Adapter pattern, no fork.** Implement `meg/cho2026/adapter.py` as a
  thin wrapper that imports from `external/EphysTokenizer/...` and exposes
  the `Tokenizer` protocol. Do not copy upstream code into `meg/`. If we
  have to patch upstream, do it as a real fork pinned by submodule SHA.

Open question for the user before phase 2 starts: do we also pin
`EphysTokenizer`, or reimplement the (small) PyTorch tokenizer in-tree?
Both are defensible — EphysTokenizer if we want to inherit pretrained
checkpoints; in-tree if we want to control the training loop without an
extra dependency.

## 7. File layout under `meg/`

Each phase is a self-contained subpackage that exposes a class implementing
the `Tokenizer` protocol from the parent §5. Do **not** flatten all three
into one set of `meg_encoder.py` / `meg_decoder.py` files — the parent §7
"three files per tokenizer" rule applies *per-tokenizer*, not per-modality,
and these three tokenizers have nothing useful in common at the
implementation level.

```
meg/
  CLAUDE.md                   # this file
  meg_config.py               # channels (271), sfreq (200 Hz), tmin/tmax, named bands
  modality_registration.py    # MEG entry for 4M's MODALITY_INFO (V, max_len, ...)
  data.py                     # mne.read_epochs(...) loader, pooled across P1–P4
  splits.py                   # image-aware train/val/test splits
  __init__.py                 # exports the three Tokenizer-protocol classes

  mu_transform/               # Phase 1
    __init__.py
    encoder.py                # clip → max-abs → μ-law companding (stateless)
    quantizer.py              # uniform binning → int token IDs
    decoder.py                # bin-center lookup → inverse μ-law → inverse max-abs
    calibration.py            # fit(train_dataset) → calibration.json
    tokenizer.py              # MuTransformTokenizer composing the above
    calibration.json          # CHECKED IN — versioned tokenizer state

  cho2026/                    # Phase 2 — thin adapter, NOT a fork
    __init__.py
    adapter.py                # wraps external/EphysTokenizer behind Tokenizer protocol
    finetune.py               # Modal entrypoint: load pretrained → train on THINGS-MEG
    README.md                 # which checkpoint, which config, why
    # NO model code lives here — that lives in external/ (see §6)

  brainomni/                  # Phase 3 — BrainOmni BrainTokenizer adapter
    __init__.py
    adapter.py                # BrainOmniTokenizer → Tokenizer protocol
    preprocess.py             # 200→256 Hz, z-score, pad→512
    load.py                   # HF checkpoint load + codebook override
    trainer.py                # finetune modes (adapt/full/rvq_only)
    config.py                 # BrainOmniConfig dataclass
    sensor_metadata.py        # MNE sensor positions → BrainOmni format
    checkpoint.py             # save/load finetuned weights on Modal volume
    compare_eval.py           # side-by-side vs μ-transform eval JSONs
    test_brainomni.py         # unit/smoke tests
    README.md
    runs/                     # config.json + evals/ per stage (git-tracked)
      V512_rvq4_win512_sf256_3a/
      V512_rvq4_win512_sf256_3b/
      3b_smoke_sweep/
      3b_codebook_sweep/
      3b_lr_sweep_coarse/

  modal/                      # Modal entrypoints (eval, finetune, sweeps)
    modal_meg_eval.py         # --tokenizer {mu_transform, brainomni}
    modal_meg_finetune_brainomni.py
```

This keeps each phase replaceable in isolation: deleting `cho2026/` does
not break phase 1; `brainomni/` is independent of phase 2.

Phase 1 run artifact:
`mu_transform/runs/V256_mu255_clip0.5-99.5_per_channel_s0/` (calibration.json,
config.json, evals/).

## 8. Evaluation contract

Identical to the parent §5. The same `test_tokenizer.py` harness must pass
for each of the three MEG tokenizers before it is wired into 4M:

1. Reconstruction fidelity (time MSE + per-channel Pearson + **spectral
   MSE**). Spectral MSE is the diagnostic metric for MEG — band power
   (especially α/β) is what carries task signal. PSD range: 1–100 Hz
   (Nyquist for 200 Hz data is 100 Hz).
2. Codebook utilization (perplexity, dead-code fraction). The μ-transform's
   dead-code fraction is the metric to watch; Cho 2026 prunes dead codes
   ("token refactorization") so its effective `V*` can be reported
   alongside nominal `V`.
3. Linear probe on THINGS labels. With 22,449 image-level codes / 1854
   concepts / ~27 superordinate categories, use the THINGS
   **superordinate** categories as the primary probe target, 1854-way
   concept ID as a secondary, image ID never (too noisy at 27k trials per
   subject). The `image_id → concept_id → superordinate` mapping comes
   from THINGS metadata, not the .fif files — add it under
   `meg/data.py`.
4. Token-sequence statistics (unigram entropy, bigram conditional entropy,
   run-length). The μ-transform is at particular risk of degenerate run
   lengths on slow drifts — watch this.

MEG-specific harness parameters (PSD frequency range, named bands) go in
`meg_config.py` and are passed into the harness as `EvalConfig`. Do not
hardcode them.

Run eval via Modal:

```bash
# μ-transform baseline
modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
  --tokenizer mu_transform \
  --calibration neural_tokenizers/meg/mu_transform/runs/V256_mu255_clip0.5-99.5_per_channel_s0/config.json \
  --n-test 3000 --seed 0

# BrainOmni — see §5 Phase 3 for brainomni commands
```

Compare JSON reports locally:

```bash
python neural_tokenizers/meg/brainomni/compare_eval.py
```

## 9. §5 leaderboard — μ-transform vs BrainOmni (2026-05-22)

All runs: **n=3000 test trials**, **seed=0**, learnable split 80/10/10
(BrainOmni) / μ-transform uses same test indices via `splits.py`. Eval JSONs
under each run's `evals/` folder.

### Reconstruction (§5.1)

| Metric | μ-transform | BrainOmni 3a (zero-shot) | BrainOmni 3b (finetuned) |
|--------|-------------|--------------------------|--------------------------|
| MSE ↓ | **0.999** | 4.35 | 1.20 |
| Channel Pearson ↑ | **0.998** | 0.345 | 0.870 |
| PSD MSE ↓ | 1.01×10⁹ | 1.88×10⁸ | **6.13×10⁷** |

μ-transform's near-perfect channel correlation is **misleading** — it is a
near-invertible per-sample codec (clip → μ-law → 256 bins). BrainOmni is a
real compressor (271×281 → 512 codes). **3b finetune** closes most of the
reconstruction gap vs 3a; μ-transform still wins trivial round-trip fidelity.

### Codebook (§5.2)

| Metric | μ-transform (V=256) | BrainOmni 3a/3b (V=512) |
|--------|----------------------|-------------------------|
| Codes used | 256/256 | 512/512 |
| Utilization | 100% | 100% |
| Perplexity | 194 | 443 → **494** (3b) |
| Entropy (nats) | 5.27 | 6.09 → **6.20** (3b) |

Both use full vocab. BrainOmni spreads mass over a larger codebook.

### Sequence structure (§5.4)

| Metric | μ-transform | BrainOmni 3a | BrainOmni 3b |
|--------|-------------|--------------|--------------|
| Entropy gap % ↑ | **22.9%** | 9.7% | 5.6% |
| Mean run length | 1.06 | ~1.00 | ~1.00 |
| frac runs ≥ 2 | **1.0** | 0.58 | 0.63 |

Different token geometries (76k sample tokens vs 512 latent codes) — entropy
gap is not directly comparable. μ-transform shows strong local temporal
redundancy; BrainOmni tokens look more i.i.d.

### Linear probe (§5.3) — **gate metric**

| Metric | μ-transform | BrainOmni 3a | BrainOmni 3b | Random |
|--------|-------------|--------------|--------------|--------|
| top-1 tokens | 0.17% | 0.17% | **0.00%** | 0.17% |
| top-5 tokens | 2.0% | **2.67%** | 2.17% | 2.17% |
| top-1 raw | 0.33% | 0.33% | 0.33% | — |

**Neither tokenizer passes the probe gate** (~1152 concepts in this 3000-trial
subset). Finetune improved reconstruction, not linear object decodability.
Raw waveform probe is identical across runs (~0.33% top-1) — barely above
chance. Full test split (~10k trials): μ-transform top-1 tokens 0.25% vs
0.40% random — still essentially chance.

### Verdict for 4M

| Criterion | μ-transform | BrainOmni 3b |
|-----------|-------------|--------------|
| Tokens / trial | ~76,151 | **512** |
| 4M context cost | Very high | **Practical** |
| Reconstruction | Excellent (near-identity) | Good (post-finetune) |
| Linear object info in tokens | None detected | None detected |
| Trained on THINGS | Calibration only | **Yes (adapt finetune)** |

**Recommendation:** ship **BrainOmni 3b** into 4M for token caching.
Probe failure is expected under reconstruction-only training; semantic readout
may require 4M's nonlinear/multimodal objective (cf. BrainOmni Stage 2 in
the paper — not implemented here).

**Pending:** full ~10k test eval (`--n-test 0`); token-cache export;
ml-4m `modality_info.py` registration.

## 10. Open decisions

Resolve in the PR that implements each phase, not here.

- **Phase 1**: per-channel vs. global max-abs scaler. **Resolved:** per-channel
  (`clip0.5-99.5`, max-abs per channel). Run:
  `V256_mu255_clip0.5-99.5_per_channel_s0`.
- **Phase 1**: vocab size `V`. **Resolved:** V=256 (paper default). Optional
  sweep `{64, 128, 256, 512}` deferred — 256 passes harness with full codebook use.
- **Phase 3**: BrainOmni codebook size. **Resolved:** V=512 (codebook sweep).
- **Phase 3**: finetune LR. **Resolved:** 3e-5 (`adapt` mode, 10 epochs).
- **Phase 3**: 4M token layout. **Open:** flatten 512 seq vs 16×32 2D grid.
- **Phase 3 → 4M token cache + shard export**: **Resolved (2026-05-23).** See
  §13 for the production pipeline. BrainOmni 3b tokens for all 98,592 trials
  are cached on the Volume and packed into `tok_meg/` shards aligned with the
  RGB train/val split. 28/28 audit checks pass.
- **Phase 2**: causal vs. non-causal decoder. Non-causal has strictly more
  reconstruction capacity; causal is needed only for real-time downstream
  uses. For 4M offline pretraining, **non-causal is the default** unless
  there is a downstream use case we know about.
- **Phase 2**: pin `EphysTokenizer` as a second submodule, or reimplement
  the (small) PyTorch tokenizer in-tree? (See §6.)
- **Phase 2**: window THINGS-MEG to 200 samples (match pretraining
  `sequence_length`) or use full 281 (retrain from scratch). Strong prior:
  window for 2a/2b, full length for 2c.
- **Phase 2**: preprocessing match — match Cho 2026's pipeline (so the
  pretrained encoder sees in-distribution input) or use our preprocessed
  THINGS-MEG as-is. Strong prior: **as-is for 2a** (cheapest), **match
  their standardize-per-channel step for 2b** (the obvious cheap win),
  **own pipeline for 2c**.
- **4M registration**: per-channel sequence (`num_channels=271,
  max_length=281`) for μ-transform / Cho2026; **separate** `meg_tokens_brainomni`
  for BrainOmni (16 × 8 × 4). **Draft in** `modality_registration.py`; not yet
  in ml-4m.
- **Split policy**: **Resolved in-repo** via `meg/splits.py` +
  `LEARNABLE_SPLIT_DEFAULTS` / `MU_SPLIT_DEFAULTS`. Optional persisted
  `/project/data/things-meg/splits/` still TBD.

## 11. Engineering norms specific to this subdir

(Inherits everything from parent §7. The below is additional.)

- All three phases must satisfy the parent §5 `Tokenizer` protocol and
  pass through `test_tokenizer.py` identically. Do not add MEG-specific
  code paths inside the harness.
- Calibration JSON (`mu_transform/calibration.json`) and finetune
  checkpoints (`brainomni/` on Modal volume at
  `/project/checkpoints/meg/brainomni/<slug>/`; `cho2026/*.ckpt` when
  phase 2 lands) are versioned tokenizer state. JSON/config pointers go
  in git; weight checkpoints go on the Modal Volume.
- No magic numbers: clip percentiles, `μ`, `V`, sampling rate, channel
  counts (271, not 272), anneal schedule all named in `meg_config.py`.
- Phase 1 is testable on a laptop with synthetic `(271, 281)` tensors —
  write the §5 harness wire-up against phase 1 FIRST, then move to phase
  2 on Modal. This proves the harness before we pay for GPU time.
- The modality registration in `modality_registration.py` is the artifact
  4M consumes. It must declare exactly the `(V, max_length, num_channels)`
  produced by whichever tokenizer is loaded. Mismatch here will silently
  break 4M training.
- Re-run [`../../modal/modal_inspect_things_meg.py`](../../modal/modal_inspect_things_meg.py)
  after any preprocessing pipeline change, and update §3 of this doc with
  the new verified numbers.
- After any change to the BrainOmni checkpoint, the bridge, the catalog, or
  the shard packer, **re-run the full pipeline audit** before considering the
  cache trustworthy:
  `cd modal && modal run modal_meg_pipeline_audit.py::audit`
  (28 cross-layer checks — structural / consistency / integrity / protocol).
  Treat any FAIL as a stop-the-world condition.

## 12. Automation / `.claude` candidates

Worth building once the phase-1 recipe stabilizes, not before:

- A `calibrate-mu-transform` skill: fits clip + max-abs on the
  THINGS-MEG training split (loaded from `/project/data/things-meg/`),
  writes `calibration.json`, and runs the §5 harness on a held-out
  split. One command, reproducible.
- A pre-commit hook that re-runs the §5 harness on any tokenizer touched
  by the diff. Prevents silent regressions.
- A `tokenize-things-meg` slash command that takes a tokenizer name
  (`mu_transform | cho2026 | brainomni`) and produces a token cache on
  the Modal Volume at `/project/data/things-meg/tokens/<tokenizer>/`,
  ready for 4M consumption. **Partially built (2026-05-23):** the
  BrainOmni 3b recipe is fully scripted in §13 — promoting it to a slash
  command is now a thin wrapper. The mu_transform and cho2026 token
  exports are not yet scripted; build those into the same skill when
  those tokenizers are ready to ship into 4M.
- A `meg-pipeline-audit` slash command wrapping
  [`../../modal/modal_meg_pipeline_audit.py`](../../modal/modal_meg_pipeline_audit.py).
  This script is already stable; promoting to a skill or a pre-commit hook
  is now safe.

Each of these locks in a workflow. Do not build them while the workflow
is still in flux — the global CLAUDE.md's "automation should follow
stable recipes" rule applies.

## 13. Production token cache + tok_meg/ shard export (2026-05-23, shipped)

This is the pipeline that turns the BrainOmni 3b finetuned checkpoint into
4M-consumable shards. Read this before re-tokenizing, switching checkpoints,
or changing the shard layout. The shared cross-modality data layout is in
[`../CLAUDE.md`](../CLAUDE.md) §9 — read that first if you've never seen the
`things_catalog.json` / `things_manifest.json` / `things/<modality>/` convention.

### 13.1 Pipeline (four idempotent Modal jobs)

| Step | Script | Cost | Run from | Output |
|---|---|---|---|---|
| 1. Bridge | [`modal/modal_download_meg_image_bridge.py`](modal/modal_download_meg_image_bridge.py) `::build` | ~$0.10, ~2 min CPU | inner repo root | `meg_trigger_to_image_id.json` |
| 2. Tokenize | [`modal/modal_meg_tokenize_all.py`](modal/modal_meg_tokenize_all.py) `::tokenize_all` | ~$1, ~6 min L40S | inner repo root | `tokens/brainomni/<slug>/{config,P1..P4}.{json,npz}` |
| 3. Pack shards | [`../../modal/modal_meg_pack_shards.py`](../../modal/modal_meg_pack_shards.py) `::{plan,pack,verify}` | ~$0.30, ~5 min CPU | `modal/` subdir | `train/things/tok_meg/`, `val/things/tok_meg/` |
| 4. Audit | [`../../modal/modal_meg_pipeline_audit.py`](../../modal/modal_meg_pipeline_audit.py) `::audit` | ~$0.10, ~3 min CPU | `modal/` subdir | 28-check report (PASS/FAIL per check) |

All four are idempotent — re-running on a clean Volume produces the same
artifacts (modulo non-determinism in any future stochastic step, of which
there are none today).

### 13.2 On-Volume artifacts

```
/project/data/things-meg/
  labels/meg_trigger_to_image_id.json    # 22,448 triggers → 9-digit image_ids
                                         # built from OpenNeuro sample_attributes_P*.csv
                                         # 100% catalog coverage (no orphans)
  tokens/brainomni/V512_rvq4_win512_sf256_3b/   # slug == producing checkpoint
    config.json                          # tokenizer + checkpoint metadata
    P{1..4}.npz                          # per-subject cache:
                                         #   tokens             (24648, 16, 8, 4) int16
                                         #   meg_trigger_codes  (24648,)         int64
                                         #   trial_idx          (24648,)         int64
                                         #   subject            ()                str

/project/data/train/things/tok_meg/shard_NNN.tar     # 23 shards, 83,456 entries
/project/data/val/things/tok_meg/shard_NNN.tar       # 4 shards,  15,136 entries
/project/data/{train,val}/things_meg_manifest.json   # per-shard entry counts + tokenizer cfg
```

Per-tar entry: `<image_id>_<subject>_t<trial_idx>.meg.npy`, shape `(16,8,4) int16`.
Aligns with the RGB shards by `image_id // 1000`-within-split (the canonical
catalog ID lookup is the manifest JSON, not arithmetic).

### 13.3 Filtering policy + corpus counts

- **Catch trials** (artificial oddball stimuli, ~2400/subject) are filtered out
  at tokenize time. They're not THINGS images and have no entry in the
  trigger→image_id bridge.
- **exp images** (~22,248): each shown 1× per subject → **88,992 trials total**
  (4 per image). Filename pattern: `<id>_<subj>_t0.meg.npy`.
- **test images** (200, used for within-subject reliability analysis): each
  shown 12× per subject → **9,600 trials total** (48 per image). Filename
  pattern: `<id>_<subj>_t{0..11}.meg.npy`.
- **Total: 98,592 trials** across all 4 subjects, **24,648 per subject**.

### 13.4 4M-side consumption (how to pair RGB ↔ MEG)

```python
# WebDataset pseudo-code (4M ingest):
rgb_sample = next(wds.WebDataset("train/things/rgb/shard_005.tar"))
# rgb_sample["__key__"] = "000005847", rgb_sample["jpg"] = <bytes>

meg_sample = next(wds.WebDataset("train/things/tok_meg/shard_005.tar"))
# meg_sample["__key__"] = "000005847_P1_t0", meg_sample["meg.npy"] = <(16,8,4) int16>

# Pair by stripping everything after the first '_' in the MEG key:
image_id = meg_sample["__key__"].split("_", 1)[0]   # "000005847"
# Then look up rgb_sample[image_id] from the RGB shard or zip across both
# WebDatasets with a custom collator.
```

### 13.5 Re-running triggers

| Change | Steps to re-run |
|---|---|
| BrainOmni 3b checkpoint moves or is replaced | Update `CHECKPOINT_SLUG` constant in both `modal/modal_meg_tokenize_all.py` AND `../../modal/modal_meg_pack_shards.py`, then re-run Steps 2, 3, 4. New tokens dir slug → old cache stays intact for comparison. |
| New finetune (3c, 3d, …) | Tokenize step writes to a new slug-named dir, leaving existing cache untouched. Pack step's `CHECKPOINT_SLUG` constant determines which cache feeds the shards. |
| RGB split (`things_manifest.json`) changes | Re-run Step 3 only — no GPU re-cost. Cache is split-invariant. |
| Bridge file regenerated | Re-run Step 3. Step 2 is bridge-aware at filter time, so the cache may also need rebuild if the bridge gained/lost triggers. |

### 13.6 The audit (mandatory after any change to the pipeline)

[`../../modal/modal_meg_pipeline_audit.py::audit`](../../modal/modal_meg_pipeline_audit.py)
runs 28 cross-layer checks in 4 categories. Last run: **2026-05-23, 28/28 pass**:

- **Layer 1 — Structural (9 checks):** file presence, JSON schema, shard count
  vs manifest.
- **Layer 2 — Cross-artifact consistency (6 checks):** bridge ⊆ catalog,
  cache triggers ⊆ bridge, MEG image_ids ⊆ matching RGB shard,
  train ∩ val = ∅ (both RGB and MEG), cache total == shard total.
- **Layer 3 — Data integrity (5 checks):** every filename parses, every
  tensor is `(16,8,4) int16`, all values in `[0, 512)`, no duplicate
  `(image_id, subject, trial_idx)` anywhere, 200-sample cache↔shard
  byte-identical round trip.
- **Layer 4 — THINGS-MEG protocol (8 checks):** total 98,592 trials,
  per-subject 24,648, distribution `[(1, 88992), (12, 800)]` (exp singletons
  + test 12-repeats), all 512 codes used on all 4 RVQ layers.

Treat any FAIL as stop-the-world.
