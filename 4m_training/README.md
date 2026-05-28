# 4M Training on THINGS (Vision + MEG/EEG)

Train [4M](https://github.com/apple/ml-4m) on pretokenized THINGS shards with RGB, depth, MEG, and EEG — without modifying `external/ml-4m/`.

> **New here? Start with [`GETTING_STARTED.md`](GETTING_STARTED.md)** — a step-by-step guide to running, training, checkpointing, and validating on your own data. This file is the architecture reference.

Volume layout: [`modal/data/README.md`](../modal/data/README.md).

## Quickstart

```bash
python 4m_training/train_4m.py demo
python 4m_training/train_4m.py validate --config 4m_training/configs/4m_things_main.yaml
python 4m_training/train_4m.py dryrun --config 4m_training/configs/4m_things_main.yaml
python 4m_training/train_4m.py train --config 4m_training/configs/4m_things_main.yaml
modal run 4m_training/modal/modal_train.py --config 4m_training/configs/4m_things_main.yaml
pytest 4m_training/ -v
```

## Files (every file justified)

| File | Why it exists |
|------|----------------|
| **`train_4m.py`** | Entry point: demo / dryrun / delegates to `run_training_4m.py` |
| **`fourm_neural_modalities.py`** | Registers the symmetric neural modalities (`tok_meg_rvq0..3`, `tok_eeg`) + masks into stock `MODALITY_INFO`; each has both an encoder and a decoder embedding (usable as input AND target) |
| **`fourm_neural_transforms.py`** | Stock `AbstractTransform` adapters; `NeuralTargetTransform` (passthrough-clip) for the splitter-materialized neural modalities |
| **`fourm_neural_embeddings.py`** | `MegRVQEncoderEmbedding` / `MegRVQDecoderEmbedding` (single codebook + axial pos) and `EegEncoderEmbedding` / `EegDecoderEmbedding` (sincos); encoder & decoder share a positional mixin |
| **`neural_trial_transform.py`** | Pick one trial: MEG `(16,8,4)→(128,4)`, EEG `(17,)`; `NeuralTargetSplitter` (coherent per-head output split); sentinel handling |
| **`neural_masking.py`** | Subclass of stock `UnifiedMasking`; zero budget when mask=0 |
| **`things_augmenter.py`** | THINGS vision shims: `ThingsImageAugmenter` (no `crop_settings` tars) + `ThingsTokTransform` (flat `(196,)` tokens, no on-disk augmentation axis) |
| **`fourm_dataloader.py`** | Patches above into stock 4M + small WDS val loader (stock val is folder-only) |
| **`validate_4m.py`** | Standalone runner for named validation tasks (masked / cross-modal) |
| **`in_loop_val.py`** | Runs `validate_4m.ValidationSuite` on the live model during training (every `eval_freq` epochs); wraps the stock trainer's `train_one_epoch` global, merging per-task loss into the epoch log |
| **`overfit_smoke.py`** | One-batch overfit sanity check (every target's loss must descend) |
| **`modal_train.py`** | Modal wrapper: `train` / `--dryrun` / `--validate` |
| **`configs/4m_things_main.yaml`** | Model, optimizer, token budgets, paths |
| **`configs/4m_things_data.yaml`** | Train/val `data_path`, domains, Dirichlet alphas |
| **`test_*.py`** | Contract tests: shard layout, stock decode, trial/masking logic, loader integration |

Train uses stock `get_train_dataloader` → `build_wds_fm_pretraining_dataloader`. Custom code is limited to modality registration, three small patches, and val-on-tars. `train_4m.run_train` imports the stock trainer as a module (not `runpy` as `__main__`) so it can wrap `train_one_epoch` for in-loop validation before driving `main(args)`; absent `in_loop_val_tasks` the launch path is identical to stock.

## Config

Production layout (what downstream users should copy):

- **`configs/4m_things_main.yaml`** — model, optimizer, `data_config:` pointer, `output_dir` on the volume
- **`configs/4m_things_data.yaml`** — `train` / `val` datasets, `data_path` under `/project/data/...`

All training data lives on the shared Modal **`project`** volume; you do not need separate repo directories per dataset. Point `data_path` at `/project/data/train/things/...` or `/project/data/train/cc12m/...`.

Edit **`4m_things_data.yaml`** for `in_domains` / `out_domains` / alphas. Keep `meg_mask` / `eeg_mask` out of domain lists (presence flags only). The neural modalities `tok_meg_rvq0..3` and `tok_eeg` are **symmetric** — list them in `in_domains` AND `out_domains` to use them as both encoder context and reconstruction targets (leak-free; see [`notes/4m_neural_modality_design.md`](../notes/4m_neural_modality_design.md)). `tok_meg` alone is an on-disk **folder**, not a modality (the validator rejects it as a domain). To try a new task without touching the main config, swap only the data file:

```bash
python 4m_training/train_4m.py train --config 4m_training/configs/4m_things_main.yaml -- \
  --data_config 4m_training/configs/4m_smoke_things_neural_in_data.yaml
```

## Validation

Validation tasks are **named masking schemes** defined in
[`configs/4m_things_val_tasks.yaml`](configs/4m_things_val_tasks.yaml); `validate_4m.py`
evaluates one, several, or all of them on a checkpoint and reports per-task loss. (4M's
built-in per-epoch eval can only validate the task you trained, so these live in a
standalone runner.) Shipped tasks:

| Task | in → out | What it measures |
|------|----------|------------------|
| `anyany_neural` | rgb+depth+meg_rvq*+eeg → rgb+depth | masked ~50% vision prediction, brain signals as context |
| `anyany_noneural` | rgb+depth → rgb+depth | masked vision prediction, no brain signals |
| `rgb2depth` | rgb → depth | cross-modal: depth from RGB |
| `depth2rgb` | depth → rgb | cross-modal: RGB from depth |

```bash
# all tasks
python 4m_training/validate_4m.py --config configs/4m_things_main.yaml \
  --tasks configs/4m_things_val_tasks.yaml --checkpoint /project/runs/.../checkpoint-last.pth
# a subset
python 4m_training/validate_4m.py ... --select rgb2depth,anyany_neural
# on Modal (GPU), against the volume's val shards
modal run 4m_training/modal/modal_train.py --validate --select rgb2depth
```

Add a task by appending an entry under `tasks:` (its `out_domains` is what the loss
scores). The shipped val tasks keep neural as encoder context only, so eval numbers stay
comparable across runs. Omit `--checkpoint` for a pipeline smoke on a randomly-initialised model.

### Modal: avoid rebuilding the image every time

The Modal image only caches **pip/apt** deps (`4m_training/modal_image.py`). Your repo is **mounted at container start**, not copied into the image, so editing YAML or Python does not trigger a multi-minute torch reinstall.

**Fast iteration (no GPU train):**

```bash
pytest 4m_training/ -v                                   # configs + synthetic shards
python 4m_training/train_4m.py validate --config 4m_training/configs/4m_things_main.yaml
modal run 4m_training/modal/modal_train.py --dryrun --config 4m_training/configs/4m_things_main.yaml
modal run 4m_training/modal/modal_smoke_train.py --case probe   # CPU, checks volume paths
```

**GPU smoke train** (only after dryrun passes): `modal run 4m_training/modal/modal_smoke_train.py --case prod_things`

Rebuild happens only if you change the package list in `modal_image.py` or Modal’s Python version.

> **Python 3.10, not 3.11+.** Stock 4M's decoder forward uses `random.sample(mod_dict.items(), …)` (`fourm/models/fm.py`), which `random.sample` rejected starting in Python 3.11. Import and `dryrun` work on newer Python, but `train` crashes on the first decoder step. `modal_image.py` pins 3.10 for this reason; keep it there unless upstream 4M changes that line.

## Token shapes

Defined in `neural_constants.py` (import constants instead of hardcoding literals):

| Modality | On-disk folder | After trial pick + split | Role | Vocab constant |
|----------|---------|------------------|------|----------------|
| `tok_meg_rvq0..3` | `tok_meg` `(n_trials, 16, 8, 4)` | `(128,)` each (one RVQ layer of the 16·8 grid) | **input + target** | `MEG_VOCAB_SIZE` |
| `tok_eeg` | `tok_eeg` `(n_trials, 17)` | `(17,)` | **input + target** | `EEG_VOCAB_SIZE` |
| `tok_rgb` | `tok_rgb` `(196,)` int16 | `(196,)` | input + target | `TOK_RGB_VOCAB_SIZE` |
| `tok_depth` | `tok_depth` `(196,)` int16 | `(196,)` | input + target | `TOK_DEPTH_VOCAB_SIZE` |

Neural modalities are **symmetric**: each is one modality with both an encoder and a decoder
embedding (`type: neural_grid`, parallel decoder branch), so it can be an encoder input, a
reconstruction target, or — as in the shipped main config — **both**. 4M's masked prediction
splits each modality's cells into input vs target disjointly, so predicting neural
regularizes the model with no input→target leak. MEG is 4 modalities (one per RVQ layer)
because RVQ has no single discrete "summed" token — each layer is its own 512-vocab head over
the `16×8` grid; EEG is one 8192-vocab head over its 17 tokens. All read the existing on-disk
folders (no repack); the loader picks one trial per sample and splits it so the 4 MEG layers
stay coherent. The reasoning (the "pick 2 of 3" design triangle + the leak-free guarantee):
[`notes/4m_neural_modality_design.md`](../notes/4m_neural_modality_design.md).

Missing neural: sentinel `(1, …)` filled with `-1`, mask `0`; presence masking zeroes
its input budget so the placeholder never reaches the encoder.
