# BrainOmni MEG tokenizer (Phase 3)

Adapter over [OpenTSLab/BrainOmni](https://github.com/OpenTSLab/BrainOmni)
`BrainTokenizer` for THINGS-MEG.

## Checkpoint

Weights auto-download from [HuggingFace OpenTSLab/BrainOmni](https://huggingface.co/OpenTSLab/BrainOmni)
on first Modal run if missing locally.

## Preprocessing

1. Resample 200 Hz → 256 Hz
2. Per-trial per-channel z-score
3. Zero-pad post-stimulus tail to 512 samples

## Modal commands (GPU + THINGS-MEG volume)

**3a — zero-shot eval (3000 test trials, same seed as μ-transform):**

```bash
cd neural-foundation-model
modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \
  --tokenizer brainomni \
  --calibration neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3a/config.json \
  --n-test 3000 \
  --seed 0
```

## Modal resources (finetune)

- GPU: A10G (24 GB VRAM)
- CPU: 8 cores, **64 GB RAM** (full 88k train tensor is ~30 GB if loaded at once)
- Default batch size: **32**

**Quick smoke (~2 min)** — verify loss decreases before a long run:

```bash
modal run neural_tokenizers/meg/modal/modal_meg_finetune_brainomni.py::smoke
```

Uses 1k train / 200 val trials, 3 epochs, batch 32.

**3b — full finetune (multi-hour; survives laptop close):**

```bash
modal run --detach neural_tokenizers/meg/modal/modal_meg_finetune_brainomni.py::finetune_detached \
  --finetune-mode adapt --lr 3e-05 --epochs 10 --batch-size 32 --stage 3b
```

Uses `.spawn()` — safe to close the laptop once the command returns (~5 s).
Do **not** use `::finetune` for long jobs; `.remote()` is cancelled on disconnect.

Interactive / short runs (keep laptop awake):

```bash
modal run neural_tokenizers/meg/modal/modal_meg_finetune_brainomni.py::finetune \
  --finetune-mode adapt --lr 3e-05 --epochs 10 --batch-size 32 --stage 3b
```

**Compare vs μ-transform baseline:**

```bash
python neural_tokenizers/meg/brainomni/compare_eval.py
```

## Token shape

`(B, 16, 8, 4)` — 16 latent sources, 8 temporal tokens, 4 RVQ layers.

4M registration: `meg/modality_registration.py` → `meg_tokens_brainomni`.
