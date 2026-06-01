# 4M neural modalities: types, embeddings, and staying loyal to token structure

A reference for *how* MEG/EEG tokens enter 4M and *why* we register them the way
we do. Read this before changing `fourm_neural_modalities.py`,
`neural_trial_transform.py`, or the neural encoder/decoder embeddings.

---

## 1. Two knobs people conflate: `type` vs `embedding`

When you "register a modality" in 4M (`MODALITY_INFO[name] = {...}`), two fields
do very different jobs:

| Field | Controls | Does **not** control |
|---|---|---|
| `type` (`img`, `seq`, `seq_token`, `seq_emb`, `meta`, our `neural_grid`) | which **masking** branch runs; which **decoder** concat path runs (parallel vs autoregressive); a few special cases (`max_tokens` auto-compute for `img`, `mod_is_img` budget math) | positional structure |
| `encoder_embedding` / `decoder_embedding` (an `nn.Module` factory) | how token-ids → vectors **and which positional embedding is added** | masking/decoder mechanics |

**Spatial/temporal structure lives in the embedding, not in `type`.** Stock options:

- `SequenceEncoderEmbedding` → **1D** sincos positions `0…L-1` (flat order).
- `ImageTokenEncoderEmbedding` → **2D** sincos positions over an `H×W` grid (fixed per cell).
- a **custom module** → any scheme (axial, 3D, RVQ-aware).

4M natively models **1D** (`seq*`) and **2D** (`img`) structure. There is **no native 3D.**

## 2. Symmetric modalities: input *and* target from one registration

A modality with **both** an `encoder_embedding` and a `decoder_embedding` can be listed in
`in_domains`, `out_domains`, or **both**. When it's in both, 4M masks it **once**:
`image_mask(tensor, max_tokens, input_budget, target_budget)` partitions its tokens into an
input set (encoder) and a target set (decoder), **disjoint by construction**. So a token is
either an encoder input *or* a decoder target on a step — never both. (This is exactly how
stock `tok_rgb` works as both input and output.)

**MEG/EEG are symmetric.** Predicting them is a reconstruction regularizer that shapes the
shared encoder/decoder; feeding them is brain-signal context. One registration does both.
The leak-free guarantee above is *why* we can do both at once without trivial copying — see
§6 for the subtlety that forced this exact shape.

## 3. Anatomy of the tokens

### MEG — BrainOmni `V512_rvq4_win512_sf256`, on-disk `(n_trials, 16, 8, 4)` int16

| Axis | Size | Meaning | Structural? |
|---|---|---|---|
| 1 | **16** | latent **source** variables `C'` (channel compression → fixed virtual brain sources; unordered) | spatial — **keep** |
| 2 | **8** | temporal latent steps `W` (SEANet strided downsampling; ordered) | temporal — **keep** |
| 3 | **4** | **RVQ** layers `N_q` (residual depth; 4 codes describe the *same* `(source,time)` cell) | not an axis — **one head per layer** |

So one MEG trial = a **`16×8` spatiotemporal grid (128 cells)**, each cell carrying
**4 residual codes**.

### EEG — LaBraM `V8192`, on-disk `(n_trials, 17)` int16

Raw output is a flat **1D** sequence of 17 tokens, single codebook. No RVQ axis,
no further structure to recover → a 1D sequence embedding is faithful.

## 4. The "pick 2 of 3" triangle (why MEG is 4 single-code modalities)

Three properties you'd like of the MEG design, of which only two are simultaneously
achievable:

| Want | Why it's nice |
|---|---|
| **A. Summed-RVQ input** | RVQ reconstructs a vector as the sum of its residual codes, so summing the 4 codebook lookups per cell is the faithful, compact (1 token/cell) input. |
| **B. One modality for in *and* out** | one mask → disjoint input/target cells → **leak-free** symmetric use (§2). |
| **C. No surgery into 4M's loss** | 4M is rigid: exactly **one cross-entropy head per modality** (`fm.forward_mod_loss`). |

- **A + C** → input (summed) and output (4 codes) are *different representations* → must be
  *different* modalities → two independent masks → **leaks** if both are in the config.
  (This was the earlier "input-only `tok_meg` + separate `tok_meg_rvq*` output heads" design.)
- **B + C** *(what we do)* → drop summation; register **4 symmetric single-code modalities**
  `tok_meg_rvq0..3` (vocab 512, 128-cell grid), each with an encoder **and** decoder
  embedding, each in both domains, each masked once → leak-free, one CE head each, no loss
  surgery.
- **A + B** → one modality that sums on input but emits 4 codes on output → needs a custom
  4-way loss head inside `external/ml-4m/` → avoid.

**Dropping summation is not a downgrade.** Once MEG is a *target* you are forced into
separate discrete codes anyway — there is no single "summed" token to cross-entropy against
(a sum is a continuous vector, not a vocabulary index). Summation was only ever an
*input-only* nicety; making the input also per-layer just makes MEG symmetric, exactly like
vision tokens (which the transformer combines via attention, not by pre-summing).

EEG sidesteps the triangle entirely: its input and output are the *same* representation
(raw 8192-vocab codes), so it is naturally **one** symmetric modality.

## 5. Keep `16×8` as a grid with **axial** positions

Two physically different axes deserve different position encodings, on **both** the encoder
and decoder side (they share the `_AxialPositions` mixin so they can't drift):

- **source (16): learned** positional embedding — latent sources have **no inherent order**, so imposing sincos order would be a lie.
- **time (8): sincos** positional embedding — time **is** ordered.

Per-cell position = `source_pos[s] + time_pos[t]`. More loyal than a single 2D sincos that
treats both axes as ordered metric grids. EEG uses 1D sincos (`_SincosPositions`).

## 6. How it maps onto 4M

| Piece | MEG (`tok_meg_rvq0..3`) | EEG (`tok_eeg`) |
|---|---|---|
| on-disk folder | `tok_meg` `(n_trials, 16, 8, 4)` | `tok_eeg` `(n_trials, 17)` |
| after trial pick + split | `(128,)` per layer, cell `p` ↔ `source=p//8, time=p%8` | `(17,)` |
| `max_tokens` (positions) | **128** | 17 |
| `type` | `neural_grid` | `neural_grid` |
| encoder embedding | `MegRVQEncoderEmbedding` (1 codebook + axial pos) | `EegEncoderEmbedding` (sincos) |
| decoder embedding | `MegRVQDecoderEmbedding` (1 head + axial pos) | `EegDecoderEmbedding` (sincos) |
| in / out domains | **both** | **both** |
| count | 4 modalities (one per RVQ layer) | 1 modality |

### Why a custom `type: neural_grid`

| Force | Resolution |
|---|---|
| `seq_token` decoding hits 4M's **autoregressive** path (`fm.cat_decoder_tensors`'s left-shift `logical_or` drops scattered `image_mask` targets — the "MEG loss = 0" bug) | `neural_grid` falls through to the **parallel** decoder branch (any non-`seq` type) |
| `type: img` routes correctly **but** the trainer overwrites img `max_tokens` with `(image_size/patch_size)²` (`run_training_4m.setup_modality_info`) — always a square; our grids are **128** and **17** | `neural_grid` is **not** `img`, so the square rule skips it and our `max_tokens` survive |
| targets must come from real on-disk data without a repack | each modality reads an existing folder via `path` (`tok_meg` / `tok_eeg`); the rename seam fans one folder out to the 4 RVQ modalities |

### Trial sampling ↔ coherence (the subtle part)

A tokenized neural example has many trials on disk; we sample **one** per step in the data
loader (`NeuralTargetSplitter`, invoked once per sample in the `rename_modalities` seam). The
splitter picks **one** trial for the shared `tok_meg` array and slices all four RVQ layers
from it, so the 4 MEG modalities always describe the *same* MEG token. The same sliced array
feeds both the encoder and decoder side of a modality (4M splits its cells), so there is no
second sampling between forward and backprop and no leakage.

### Leak-free, by construction

Because each neural modality is masked **once** (§2), the cells given to the encoder are
exactly the cells **withheld** from the decoder. There is no trivial-copy path even though
neural is both input and target. (Contrast the old A+C design, where input `tok_meg` and
output `tok_meg_rvq*` were *different* modalities with *independent* masks → overlapping
cells → partial leak. That is the bug the symmetric design eliminates.)

### Training gotcha: `find_unused_params`

With many heads (4 MEG + 1 EEG + vision) and a stochastic Dirichlet target budget, some heads
get **0 targets** on a given step, so their parameters produce no gradient. Stock 4M wraps the
model in DDP, which then raises *"Expected to have finished reduction… parameters that were
not used in producing loss"*. Set `find_unused_params: true` in the main config (already set in
`configs/4m_things_main.yaml`) for any neural run.

### Where it's proven

- `tests/test_overfit_smoke.py` — one fixed batch, every modality a target: all 4 MEG heads
  and the EEG head must drive their loss down (local CPU proof that the heads learn).
- `tests/test_neural_masking.py::...::test_input_and_target_cells_are_disjoint_no_leak` — the
  leak-free guarantee.
- `tests/test_neural_output_modalities.py::TestNeuralOutputGradientFlow` — finite loss +
  gradient reaches each neural head.
- `modal/modal_smoke_train.py` — GPU proof that the neural losses descend over steps.

---

Related: [[neural_fm_modal_main_collision]] (Modal entrypoint gotcha).
