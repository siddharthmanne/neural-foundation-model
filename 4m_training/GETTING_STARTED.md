# Getting started

A step-by-step guide to training, checkpointing, and validating 4M on your own
data with this system. For *how it works* internally, see [`README.md`](README.md);
for the data format, see [`../modal/data/README.md`](../modal/data/README.md).

**Assumptions:** you have a Modal account with the CLI authed (`modal token new`),
and your pretokenized shards are on a Modal volume in the THINGS / CC12M format
(parallel modality folders of `{image_id}.npy`).

---

## Step 1 — Tell the system where your paths are

You set paths by **editing one file**: [`lib/repo_paths.py`](lib/repo_paths.py). Open it and
change the two values in the `EDIT HERE` block. That's permanent — every terminal, every
`python ...` and `modal run` reads from this file, so there is nothing to export and nothing
to re-run.

```python
# lib/repo_paths.py
ML4M_DIR_OVERRIDE = None        # path to your 4M checkout, e.g. "/Users/you/code/ml-4m".
                                # Leave None to use the bundled external/ml-4m submodule.
PROJECT_VOLUME_NAME = "project" # the name of your Modal data + checkpoint volume.
```

Everything that needs the 4M repo (trainer, text tokenizer, the editable install on Modal)
and the Modal volume follows these automatically — locally and inside the container.

The **one** path you set per dataset is `data_path` in the **data** YAML (your shards under
the volume's `/project` mount), plus `output_dir` in the **main** YAML (where checkpoints go):
```yaml
# configs/4m_things_data.yaml
data_path: /project/data/train/things/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{000..026}.tar
# configs/4m_things_main.yaml
output_dir: /project/runs/my_experiment
```
That's everything; nothing else needs touching.

## Step 2 — Point the configs at your data

Two YAMLs drive everything (copy them to make your own experiments):

- [`configs/4m_things_main.yaml`](configs/4m_things_main.yaml) — model, optimizer, token budgets, `output_dir`.
- [`configs/4m_things_data.yaml`](configs/4m_things_data.yaml) — where the data is and what to predict.

In the **data** YAML, set `data_path` (train and val) to your shards under `/project`:
```yaml
data_path: /project/data/train/things/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{000..026}.tar
```
- The `[...]` lists the modality folders to zip together; `{000..026}` is a brace range.
- `in_domains` = what the encoder sees; `out_domains` = what gets predicted (the loss).
- **`tok_meg` / `tok_eeg` may only be inputs** (`in_domains`), never `out_domains`.
- `meg_mask` / `eeg_mask` stay out of both lists (they are presence flags, used automatically).

## Step 3 — Sanity-check before paying for a GPU

```bash
pytest 4m_training/ -q                                                  # unit + contract tests
python 4m_training/train_4m.py validate --config 4m_training/configs/4m_things_main.yaml   # config sanity, no GPU/data
modal run 4m_training/modal/modal_train.py --dryrun --config 4m_training/configs/4m_things_data.yaml  # CPU: real data flows
```
The dryrun should print sane batch shapes (e.g. `tok_meg (B,128,4)`, `tok_rgb (B,196)`).

## Step 4 — Train

Start with a tiny GPU smoke (a few steps, T4), then the real run:
```bash
modal run 4m_training/modal/modal_smoke_train.py --case prod_things           # smoke: confirms it trains
modal run 4m_training/modal/modal_train.py --config 4m_training/configs/4m_things_main.yaml   # full training (A100)
```
Checkpoints and logs land in `output_dir` (on the volume).

## Step 5 — Checkpoint and resume

- `output_dir` (main YAML) is where checkpoints are written.
- **Auto-resume**: by default a run picks up `checkpoint-last.pth` from `output_dir`, so
  re-running the same command continues where it left off. (Pause/resume works on the
  Modal image's PyTorch — handled for you.)
- **Manual** (main YAML): `resume: /project/runs/.../checkpoint-last.pth` (continue, with
  optimizer/epoch state) or `finetune: /project/runs/.../checkpoint-best.pth` (weights
  only, e.g. a new data mix). Set `auto_resume: false` to always start fresh.
- Changing the model architecture (size, domains) makes old checkpoints incompatible —
  use a fresh `output_dir`.

## Step 6 — Validate

Validation tasks are named masking schemes in
[`configs/4m_things_val_tasks.yaml`](configs/4m_things_val_tasks.yaml). Run all, or pick some:
```bash
modal run 4m_training/modal/modal_train.py --validate \
    --checkpoint /project/runs/4m_things_neural/checkpoint-last.pth        # all tasks
modal run 4m_training/modal/modal_train.py --validate --select rgb2depth,anyany_neural  # subset
```
Shipped tasks: `anyany_neural`, `anyany_noneural` (masked ~50% vision prediction, with/without
brain signals) and `rgb2depth`, `depth2rgb` (cross-modal). Each prints its own loss. Add a task
by appending an entry under `tasks:` (its `out_domains` is what the loss scores).

## Step 7 — Change hyperparameters

All in the **main** YAML — no code changes:

| Want to change | Field |
|----------------|-------|
| Model size | `model:` (see the preset list in the file) |
| Batch size | `batch_size:` |
| Learning rate | `blr:` (effective LR = `blr * batch_size / 256`) |
| Training length | `epochs:`, `epoch_size:` |
| Tokens per step | `num_input_tokens:`, `num_target_tokens:` |
| Precision | `dtype:` (`bfloat16` / `float32`) |
| Regularization | `weight_decay:`, `clip_grad:` |

What to predict / mixture weights live in the **data** YAML (`in_domains`, `out_domains`,
`input_alphas`, `target_alphas`, and `weights` when mixing datasets).


---

### Fast iteration tips
- Edit YAML or Python freely — the Modal image is **not** rebuilt (your repo is mounted at
  container start). Only changing `modal_image.py` deps/Python triggers a rebuild.
- Always `--dryrun` (CPU) before a GPU run; it's free and catches path/shape mistakes.
