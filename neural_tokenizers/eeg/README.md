## Files in this directory

**Core tokenizer modules** (satisfy the `Tokenizer` protocol in `../evaluation/protocol.py`):
- `eeg_encoder.py` — BottleneckMLP encoder backbone (Stage 1: B_6-Wi_1024 mirroring 4M's human-poses config).
- `eeg_quantizer.py` — thin wrapper around a 4M quantizer (`memcodes`, codebook_size=1024, num_codebooks=8, latent_dim=1024).
- `eeg_decoder.py` — decoder backbone mirroring the encoder.
- `eeg_tokenizer.py` — composes the three above into a class exposing `tokenize`, `decode_tokens`, `codebook_size`.
- `labram_tokenizer.py` — wraps the finetuned LaBraM VQNSP checkpoint in the same `Tokenizer` protocol. Stage 2c implementation.

**Modal pipeline scripts** (in `modal/`):
- `modal/convert_eeg_to_labram_hdf5.py` — converts THINGS-EEG2 `.npy` files to LaBraM HDF5 format with image-level 80/20 split.
- `modal/convert_eeg1_to_labram_hdf5.py` — same for THINGS-EEG1 EEGLAB derivatives.
- `modal/finetune_labram_tokenizer.py` — finetunes LaBraM's `vqnsp.pth` on THINGS-EEG HDF5 data via `torchrun`.
- `modal/train_stage1.py` — trains the Stage 1 BottleneckMLP + product VQ tokenizer.
- `modal/modal_eeg_produce_tokens.py` — runs the finetuned tokenizer over all EEG1 + EEG2 trials, writes per-subject `.npz` caches keyed by THINGS catalog image_id.
- `modal/eval_labram_tokenizer.py` — evaluates the finetuned LaBraM checkpoint on the four harness axes.
- `modal/run_eval_harness.py` — runs the full `neural_tokenizers/` eval harness on Modal against real val data.
- `modal/modal_eeg_write_coverage_json.py` — writes `eeg_coverage.json` mapping EEG1 ∪ EEG2 image_ids onto the THINGS catalog.
- `modal/check_image_coverage.py` — quick sanity check of catalog coverage by EEG1 / EEG2 / their union.
- `modal/download_eeg2_image_metadata.py` — fetches Gifford's `image_metadata.npy` from OSF if needed.
- `modal/verify_unified_shards.py` — read-only integrity check of the unified EEG shard format against per-subject `.npz` caches.

## Status

Stage 1 (BottleneckMLP + memcodes) and Stage 2c (LaBraM finetune) are both complete and evaluated on the four `../CLAUDE.md` §5 axes. LaBraM (2c) was pursued as the production tokenizer. Token caches produced by `modal_eeg_produce_tokens.py` are on the `project` volume under `/project/data/things-eeg/tokens/labram/`.

## Data location

EEG data lives on the Modal `project` volume:
- `/project/data/raw/things-eeg2/` — 10 subjects, `preprocessed_eeg_{training,test}.npy` per subject (MVNN-whitened, 17 ch, 100 Hz)
- `/project/data/raw/things-eeg1/derivatives/` — 50 subjects, EEGLAB `.set`/`.fdt`
- `/project/data/things-eeg/` — reorganized artifact tree: `preprocessed/`, `labels/`, `tokens/`

See `../../modal/download_things_eeg.py` for the download jobs.

## MVNN whitening — resolution

Gifford's preprocessed THINGS-EEG2 is MVNN-whitened (dimensionless), not raw µV. LaBraM nominally expects µV. Resolution used: empirical per-channel rescale to ≈10 µV std, applied inside the conversion script before writing HDF5. The codebook-collapse smoke test (`Unused_code` in `engine_for_vqnsp.py:117-126`) confirmed the codebook was in active use after epoch 1.

## LaBraM pipeline — quick reference

Full spec (HDF5 format, channel-naming constraints, CLI flags, resume mechanics, Stage-1 loss) lives in the root `../../CLAUDE.md` under "LaBraM Stage-1 verified pipeline spec". Key facts:
- Input: 200 Hz, 0.1–75 Hz bandpass + 50 Hz notch, 17 posterior channels, continuous `(17, N×200)` per subject
- Training: `torchrun run_vqnsp_training.py --input_size 1600 --codebook_n_emd 8192 --codebook_emd_dim 64 --quantize_kmeans_init`
- Checkpoint: `/project/checkpoints/eeg/labram/V8192_d64_ch17_sr200_train-eeg1+2_e5/checkpoint.pt`
