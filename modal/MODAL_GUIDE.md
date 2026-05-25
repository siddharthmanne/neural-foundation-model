# When and how to write a Modal file

For any teammate who needs to run something on Modal. If you don't need Modal for your current work (e.g., you're happy on FarmShare), skip this.

## What Modal is, briefly

Modal is a service that runs your Python code on remote computers — including ones with GPUs — and bills you per second of compute used. Instead of buying a GPU machine or fighting a queue, you write a Python script with a few decorator lines saying "run this on Modal," type `modal run my_file.py`, and Modal:

1. Builds a Linux container with your Python and dependencies (the **Image**).
2. Attaches the GPU you asked for.
3. Mounts any shared storage you specified (the **Volume**).
4. Injects any API tokens you need (the **Secrets**).
5. Runs your code there.
6. Tears the container down.

You see the logs streaming in your terminal as if it were running locally. The container is gone afterwards; anything you wrote to the Volume persists.

Our team has $200 of shared credits in a Modal workspace called `neural-fm`. Any job you run inside that workspace draws from the shared pot.

## What a Modal file does

A Modal file is just a Python script with one extra ingredient: `@app.function(...)` decorators on the functions you want to run remotely instead of locally. The decorator tells Modal "this function isn't meant to run on the laptop — when called via `.remote()`, ship it to a container and execute it there."

The four pieces every Modal file needs:

| Piece | What it is | Where ours comes from |
|---|---|---|
| **App** | A label that groups related jobs in the dashboard | `app = modal.App("neural-fm")` in `modal_app.py` |
| **Image** | OS + Python packages your code needs | Defined in `modal_app.py` from `requirements.txt` |
| **Volume** (optional) | Persistent disk for things you want to keep between runs | `data_volume`, mounted at `/data` |
| **Secret** (optional) | API tokens (HF, W&B, etc.) without committing them to git | **You create your own** — see the section below |

The App, Image, and Volume are shared infrastructure — `from modal_app import ...` and you have them. **Secrets are personal** and you create your own (see "Personal secrets" below). The reason: Modal secrets are workspace-scoped, meaning a secret defined in the shared `modal_app.py` would be usable by every teammate, which is not what you want for personal credentials.

## Personal secrets (HF tokens, W&B keys, etc.)

If your Modal job needs to authenticate with Hugging Face, Weights & Biases, or any other external service, **create your own Modal secret first**. Probable best not to share tokens via a generic secret name in the shared `modal_app.py`. Will update if this changes.

One-time setup, from your laptop terminal:

```
modal secret create hf-yourname HF_TOKEN=hf_xxxxx_your_token_here
```

(`hf-liane`, `hf-brandon`, `hf-sid` — pick something obvious so we can tell whose token is whose in the dashboard.)

Then in your Modal file:

```python
import modal
from modal_app import app, image, data_volume

my_hf = modal.Secret.from_name("hf-yourname")

@app.function(image=image, volumes={"/data": data_volume}, secrets=[my_hf])
def my_job():
    import os
    token = os.environ["HF_TOKEN"]   # set automatically by the Secret
    ...
```

Same idea for W&B (`modal secret create wandb-yourname WANDB_API_KEY=...`) or anything else.

## When you'd write a new Modal file

- A training job that's too big or slow for your laptop / FarmShare.
- An evaluation that needs a particular GPU type and you don't want to wait in a queue.
- A many-config sweep where you want to spin up several containers in parallel.
- A one-off data preprocessing job that needs more RAM or disk than you have locally.

If you can do it on FarmShare without pain, do it on FarmShare.

## The minimum template

Create a new file (say `my_job.py`) next to `modal_app.py` and write:

```python
"""What this job does."""

import modal
from modal_app import app, image, data_volume

# If your job needs an API token, create your own personal secret first
# (see "Personal secrets" above) and reference it here. Otherwise delete
# this line and the `secrets=[...]` argument below.
my_hf = modal.Secret.from_name("hf-yourname")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[my_hf],         # only if you need a token; remove if not
    gpu="A10",               # T4 / L40S / A10 / A100 / H100 — pick the smallest that fits
    timeout=60 * 60,         # seconds — set this so a runaway job doesn't burn credits forever
)
def do_the_thing():
    """The actual work. Runs on a remote Modal container.

    /data is the shared Volume — files you write there persist.
    Don't write outside /data; everything else disappears when the container ends.
    """
    # your code here
    ...

    # Flush any /data writes back to the Volume before the function returns.
    data_volume.commit()


@app.local_entrypoint()
def main():
    do_the_thing.remote()
```

Run it: `modal run my_job.py`

### About `--detach`

For anything longer than about 10 minutes, run it with `--detach`:

```
modal run --detach my_job.py
```

Without `--detach`, the job dies the moment you close your terminal or your laptop goes to sleep — and the credits you spent up to that point are gone with it. With `--detach`, the job keeps running on Modal's servers regardless; you reconnect by visiting the dashboard at modal.com to see logs and progress. Use `--detach` for any real training run. Skip it for quick tests where you want to see the output live.

## The two-file pattern (for real training scripts)

The template above puts your code inline inside a Modal function. That's fine for small one-off jobs, but for actual training scripts there's a cleaner pattern: **keep your training code as a regular Python file, and write a small Modal wrapper that calls it.**

Two files instead of one:

- **`train_my_thing.py`** — pure Python. argparse, training loop, save outputs. No Modal imports anywhere. You can run it locally on your laptop or on FarmShare with a small dataset for debugging.
- **`modal_train_my_thing.py`** — ~25-line Modal wrapper. Defines image, GPU, volume, then `subprocess.run`s the training script inside the container.

**A working demo lives in this folder**: see `demo_train.py` (the "training" script) and `modal_demo.py` (the Modal wrapper that runs it).

To try it:
```
modal run modal_demo.py
```

That runs `demo_train.py` on a Modal container for ~2 seconds, writes a result file to the shared Volume at `/data/demo_output/`, and exits.

To make your own version:

1. Copy `modal_demo.py` to `modal_<your_thing>.py`.
2. Write your real training script as a sibling Python file (e.g., `train_eeg_tokenizer.py`).
3. In your new Modal wrapper file:
   - Change `add_local_python_source("demo_train")` to your script's module name.
   - Update the `subprocess.run([...])` command to pass the right args.
   - Set `gpu=`, `timeout=`, and any secrets you need.
4. `modal run modal_<your_thing>.py`.

**Why this is nice:**
- Your training code stays portable — it runs anywhere Python runs.
- The Modal wrapper is small and rarely needs changes; you edit your training code freely.
- You can test your training locally with a tiny dataset before paying for Modal compute.

IMPORTANT: Always set `timeout=` so a job can't run indefinitely.

## Things to know before you click "run"

- **Use the cheapest GPU that fits.** OOM errors are obvious; spending too much isn't. Start at T4 or A10, upgrade only when needed.
- **Run `modal_app.py::hello` first** in your workspace to confirm setup works.
- **Detached jobs keep running after you close your terminal.** Check the dashboard if you think you forgot one.
- **Anything outside `/data` disappears when the container ends.** Don't save to `/root/` or `/tmp/` expecting it to stick around.
- **Call `data_volume.commit()`** at the end of any function that writes to `/data`, or your writes won't persist to the Volume.

## Architecture gotchas (learned the hard way)

Five traps that bit us building the MEG tokenizer Modal scripts under
`neural_tokenizers/meg/modal/`. Read these before adding new scripts that
import from project subpackages.

### 1. Make every project package a *regular* package, not a namespace package
If `neural_tokenizers/__init__.py` is missing, `neural_tokenizers` becomes a
namespace package. `add_local_python_source("neural_tokenizers")` then packs
a confusing union of every matching directory on `sys.path` (e.g., if the
repo is checked out twice). Symptom: `neural_tokenizers.__path__` shows the
same dir twice, and imports succeed locally but the container can't find
submodules. Fix: add an empty `__init__.py` at the package root.

### 2. `pip_install_from_requirements("requirements.txt")` is evaluated at import time
`modal_app.py` uses a relative path. Any script that does
`from modal_app import app, image` from outside the `modal/` directory dies
with `FileNotFoundError: requirements.txt`. Workaround: don't inherit the
image. Build your own with `modal.Image.debian_slim(...).pip_install(...)`
in your script and create a separate `app = modal.App("neural-fm")` —
Modal groups by App name in the dashboard, so jobs still cluster correctly.

### 3. `modal_app.py` registers `@app.local_entrypoint() def main()`
If you import `modal_app`'s `app` and add your own `def main():` entrypoint,
Modal errors with `Duplicate local entrypoint name: main`. Use any name
other than `main` (e.g. `def run():`, `def calibrate():`).

### 4. `__file__`-based `parents[N]` paths break on the remote container
On the laptop, your script is at `<repo>/sub/dir/script.py` and
`Path(__file__).resolve().parents[3]` = `<repo>`. On the Modal container,
Modal copies your script to `/root/script.py` (top-level), so the same
expression `IndexError`s. Always wrap such `parents[N]` lookups in
`try/except IndexError`, and rely on `add_local_python_source` to make
your package importable remotely:

```python
try:
    import my_package  # noqa: F401
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    except IndexError:
        pass   # remote container — add_local_python_source handles it
```

### 5. `modal run` resolves paths from cwd, not from the script's location
`modal run neural_tokenizers/meg/modal/foo.py` from the wrong directory →
`FileNotFoundError`. The script must always be reachable as a relative path
from the cwd where you invoked `modal run`. Practical implication: always
run from the inner repo root (`neural-foundation-model/neural-foundation-model/`),
or use an absolute path.

### Bonus: `add_local_python_source` packs *.py only*
Non-Python files (JSON sidecars, weights, calibration files) shipped beside
your code DO NOT get packed by `add_local_python_source`. Either (a) put
them on a Volume the container mounts, or (b) ship them as function
arguments via `.remote()` (small files only), or (c) use
`image.add_local_file(...)` / `image.add_local_dir(...)` explicitly.
We use (a) for the THINGS image-id→concept-id mapping (lives on the
`project` Volume, plus a git-tracked local copy for laptop use).

## Where to look for more

- Working examples in this folder: `modal_app.py` (shared setup + `hello` test), `modal_demo.py` + `demo_train.py` (the two-file wrapper pattern).
- Modal docs: https://modal.com/docs
- The CS224R compute guide PDF (Stanford's Modal intro, also a good general reference): https://cs224r.stanford.edu/material/CS224R_compute_guide.pdf
