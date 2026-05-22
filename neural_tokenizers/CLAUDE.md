# Neural Tokenizers — CLAUDE.md

Scope: this file applies only to work in `neural_tokenizers/`. Read it before
making changes in this subdir. Repo-wide engineering norms live in the user's
global `~/.claude/CLAUDE.md` and are not restated here.

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

Data is not yet downloaded locally. Numbers below are from the canonical
public releases and **must be verified** on first contact.

- **THINGS-EEG2** (Gifford et al. 2022): 63 channels, 100 Hz, ~1.1s epochs →
  trial tensor shape roughly `(63, 100)`. ~10 subjects, ~16k training trials each.
- **THINGS-MEG** (Hebart et al. 2023): 272 channels, sampling rate depends on
  preprocessing pipeline (200 Hz or 1200 Hz), epoch length comparable.
- Both: 2D `(channels, time)` per trial. Channels are *not* a spatial grid —
  treat as an unordered set or a sensor-topography graph, not as image rows.
- Stimulus labels: THINGS has ~1,854 object concepts (use for downstream probes).

Open data questions to resolve before training:
- exact preprocessed shape of each dataset
- which preprocessing pipeline (raw vs. ICA-cleaned vs. baseline-corrected)
- where the data lives at training time (Modal Volume path)

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
    def encode(self, x: Tensor) -> LongTensor: ...   # (B, C, T) -> (B, L)
    def decode(self, t: LongTensor) -> Tensor:  ...  # (B, L)    -> (B, C, T)
    @property
    def codebook_size(self) -> int: ...
```

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

## 8. Notes on automation / `.claude` layering

This subdir is a candidate for a scoped `.claude/` later:
- A `train-tokenizer` skill or slash command that wraps the Modal launch + log
  tailing, once the recipe stabilizes.
- A pre-commit hook that runs the §5 harness on whatever checkpoint a PR
  touches, so we never merge a tokenizer regression.
- Neither is worth building until the scratch-baseline numbers exist —
  premature automation locks in the wrong recipe.
