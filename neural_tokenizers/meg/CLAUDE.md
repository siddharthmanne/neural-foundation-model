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

| Tokenizer            | V                              | num_codebooks | per-channel max_length |
|----------------------|--------------------------------|---------------|------------------------|
| Phase 1 (μ-transform)| 256 (paper default)            | 1             | 281 (= trial timepoints) |
| Phase 2 (Cho2026 learned) | 128 (paper) → V* after refactor | 1        | 281                    |
| Phase 3 (BrainOmni)  | TBD                            | TBD           | TBD                    |

Registration shape (per-channel sequence, **not** flattened):
`num_channels = 271`, `max_length = 281`. The alternative — flatten to one
giant sequence of 271 × 281 = 76,151 tokens per trial — blows past every
sensible context length and discards the channel/time factorization 4M can
exploit. Mirror `human_poses`, which keeps `num_channels = 207` separate
from `max_length = 263`. Document the final choice in
`meg/modality_registration.py` when phase 1 lands.

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

- `/project/data/things-meg/splits/` — train/val/test split manifests.
  None exist yet. Splits will be by held-out image ID (cf.
  `modal_download_things_meg.py` docstring), not by trial index.
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

| Phase | Tokenizer                                | Learns? | Cost     | Role                         |
|-------|------------------------------------------|---------|----------|------------------------------|
| 1     | μ-transform (Cho 2026 baseline)          | No¹     | seconds  | Floor + harness sanity check |
| 2     | Cho 2026 learnable AE (causal / non-causal) | Yes  | hours-GPU | Likely production tokenizer |
| 3     | BrainOmni tokenizer, finetuned           | Yes     | hours-GPU | Stretch / comparison         |

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

### Phase 3 — BrainOmni (deferred)

Multi-modal tokenizer. Not the path forward unless phase 2 has clear
distribution-mismatch symptoms AND BrainOmni's pretraining corpus includes
event-related MEG. Revisit after phase 2 numbers land.

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

  brainomni/                  # Phase 3 — empty until phase 2 ships
```

This keeps each phase replaceable in isolation: deleting `cho2026/` does
not break phase 1; adding `brainomni/` does not touch phase 2.

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

## 9. Open decisions

Resolve in the PR that implements each phase, not here.

- **Phase 1**: per-channel vs. global max-abs scaler. (Strong prior:
  per-channel — see §5.)
- **Phase 1**: vocab size `V`. Defaults to 256 (paper). Sweep `{64, 128,
  256, 512}` once the §5 harness is producing stable numbers.
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
  max_length=281`) vs. flattened sequence (`max_length=271 × 281`).
  Strong prior: per-channel (see §2).
- **Split policy**: implement `/project/data/things-meg/splits/by_image.json`
  before training any tokenizer; trial-index splits leak THINGS image
  identity across train/val. Owner: whoever ships phase 1.

## 10. Engineering norms specific to this subdir

(Inherits everything from parent §7. The below is additional.)

- All three phases must satisfy the parent §5 `Tokenizer` protocol and
  pass through `test_tokenizer.py` identically. Do not add MEG-specific
  code paths inside the harness.
- Calibration JSON (`mu_transform/calibration.json`) and finetune
  checkpoints (`cho2026/*.ckpt`) are versioned tokenizer state. JSON goes
  in git; checkpoints go on the Modal Volume with a pointer file in git.
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

## 11. Automation / `.claude` candidates

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
  ready for 4M consumption. Wraps `modal run` so we don't retype the
  same incantation.

Each of these locks in a workflow. Do not build them while the workflow
is still in flux — the global CLAUDE.md's "automation should follow
stable recipes" rule applies.
