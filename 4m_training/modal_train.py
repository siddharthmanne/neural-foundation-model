"""Modal GPU wrapper for ``train_4m.py``.

Run from repo root::

    modal run 4m_training/modal_train.py --config 4m_training/configs/4m_things_main.yaml

Dryrun on the volume (no GPU, no training — fast config/data check)::

    modal run 4m_training/modal_train.py --dryrun --config 4m_training/configs/4m_things_data.yaml

Image rebuilds only when ``modal_image.py`` pip/apt deps change, not when you edit YAML or Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
import modal

def _load_modal_image():
    for path in (
        __import__("pathlib").Path("/opt/repo/4m_training/_modal_load.py"),
        __import__("pathlib").Path(__file__).resolve().parent / "_modal_load.py",
    ):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_modal_load", path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        return loader.load_modal_image()
    raise ImportError("_modal_load.py not found (expected /opt/repo/4m_training on Modal)")


_mi = _load_modal_image()
REPO = _mi.REPO
ensure_fourm = _mi.ensure_fourm
train_image = _mi.train_image
training_env = _mi.training_env

app = modal.App("train-4m-things")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"


def _run_train(dryrun: bool, condition: str = "rgb_only", large_gpu: bool = False) -> None:
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/run_scaling_experiment_2.py"),
        "--mode", "sweep",
        "--condition", condition,
    ]
    if dryrun:
        cmd.append("--test_run")
    if large_gpu:
        cmd.append("--large_gpu")

    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="A100",
    timeout=60 * 60 * 24,
    memory=64 * 1024,
)
def train(condition: str = "rgb_only", large_gpu: bool = False) -> None:
    _run_train(dryrun=False, condition=condition, large_gpu=large_gpu)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    memory=64 * 1024,
)
def train_large(condition: str = "rgb_only") -> None:
    """80GB A100 variant — use for dim=512 re-runs with bs=512."""
    _run_train(dryrun=False, condition=condition, large_gpu=True)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=32 * 1024,
)
def dryrun_job(condition: str = "rgb_only") -> None:
    """CPU-only dataloader smoke test — use this before paying for GPU train."""
    _run_train(dryrun=True, condition=condition)


def _run_analysis(condition: str | None, mode: str, loss_type: str = "rgb") -> None:
    """Run collect or fit mode of run_scaling_experiment_2.py.

    condition=None processes all conditions found under sweep_base_dir.
    """
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/run_scaling_experiment_2.py"),
        "--mode", mode,
        "--loss_type", loss_type,
    ]
    if condition:
        cmd.extend(["--condition", condition])
    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=16 * 1024,
)
def collect_job(condition: str = "") -> None:
    """Parse training logs and write results.json. Empty condition = all conditions."""
    _run_analysis(condition or None, "collect")


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=16 * 1024,
)
def fit_job(condition: str = "", loss_type: str = "rgb") -> None:
    """Fit Chinchilla scaling law. Empty condition = all conditions."""
    _run_analysis(condition or None, "fit", loss_type)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=16 * 1024,
)
def plot_job(
    sweep_dir: str = "/project/data/scaling_sweep",
    conditions: str = "",
    control: str = "rgb_only",
    treatment: str = "",
) -> None:
    """Plot per-condition and comparison plots from the scaling sweep.

    Plots are saved to {sweep_dir}/{condition}/plots/ per condition and
    {sweep_dir}/plots/ for cross-condition comparisons.
    conditions: comma-separated list to limit which conditions are plotted;
    empty string means auto-discover all conditions under sweep_dir.
    treatment: specific treatment condition for gap plot; empty = all non-control conditions.
    """
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/plot_training.py"),
        "--sweep_dir", sweep_dir,
        "--control", control,
    ]
    if treatment:
        cmd.extend(["--treatment", treatment])
    if conditions:
        cmd.extend(["--conditions", conditions])
    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 60 * 2,   # probe + 5-fold CV can take ~1-2 h on CPU
    memory=16 * 1024,
)
def eval_dinov2_job(
    shard_indices: list[int] | None = None,
    n_samples: int | None = None,
    probe_classifier: str = "linear",
    probe_n_folds: int = 5,
    probe_epochs: int = 100,
    seed: int = 0,
    run_probe: bool = True,
    run_retrieval: bool = True,
) -> None:
    """Run the DINOv2 tokenizer evaluation on THINGS val tokens."""
    import sys as _sys
    import importlib.util as _ilu
    from pathlib import Path as _Path

    # Load eval_dinov2.py by file path — avoids package/relative-import issues.
    _eval_path = f"{REPO}/neural_tokenizers/evaluation/eval_dinov2.py"
    _spec = _ilu.spec_from_file_location("eval_dinov2", _eval_path)
    _mod = _ilu.module_from_spec(_spec)

    # Make the evaluation sub-modules importable by their plain names before
    # executing the module (its top-level try/except fallback uses sys.path).
    _eval_dir = str(_Path(_eval_path).parent)
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, _eval_dir)

    _spec.loader.exec_module(_mod)

    _mod.run_eval(
        tokens_dir=_Path(f"{PROJECT}/data/val/things/tok_dinov2@224"),
        image_id_to_concept_path=_Path(
            f"{REPO}/neural_tokenizers/meg/data/image_id_to_concept.json"
        ),
        concept_to_superordinate_path=_Path(
            f"{REPO}/neural_tokenizers/meg/data/concept_id_to_superordinate.json"
        ),
        shard_indices=shard_indices,
        n_samples=n_samples,
        probe_classifier=probe_classifier,
        probe_n_folds=probe_n_folds,
        probe_epochs=probe_epochs,
        seed=seed,
        run_probe=run_probe,
        run_retrieval=run_retrieval,
    )


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=8 * 1024,
)
def download_dinov2_models() -> None:
    """Download DINOv2-B/14 and 4M VQVAE tokenizer to the project volume cache (CPU only).

    Run this once before tokenize_things_dinov2 so GPU containers start with models
    already on disk — avoids paying GPU rates during a ~5 min download.
    """
    ensure_fourm()
    import os as _os
    import sys as _sys
    import torch

    _hf_cache = f"{PROJECT}/hf_cache"
    _os.makedirs(_hf_cache, exist_ok=True)
    _os.environ["HF_HOME"] = _hf_cache
    _os.environ["TORCH_HOME"] = _hf_cache

    _sys.path.insert(0, f"{REPO}/ml-4m")
    from fourm.vq.vqvae import VQVAE

    print("Downloading DINOv2-B/14…", flush=True)
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    print("Downloading 4M DINOv2 VQVAE tokenizer…", flush=True)
    VQVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448")
    project_volume.commit()
    print("Done — models cached to project volume.", flush=True)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="T4",
    timeout=60 * 30,
    memory=16 * 1024,
)
def tokenize_things_dinov2(split: str = "train") -> None:
    """Tokenize THINGS images with DINOv2-B/14 + 4M VQVAE, writing tok_dinov2@224 shards.

    Reads raw JPEG images from /project/data/{split}/things/rgb/shard_NNN.tar
    and writes /project/data/{split}/things/tok_dinov2@224/shard_NNN.tar.
    Each entry: {key}.npy with shape (1, 256) int16.
    """
    ensure_fourm()
    import io
    import os as _os
    import sys as _sys
    import tarfile

    import numpy as np
    import torch
    import torchvision.transforms as T
    import webdataset as wds
    from pathlib import Path
    from PIL import Image

    # Cache both HuggingFace and torch.hub downloads to the project volume so they
    # survive across Modal container restarts. No naming collisions: HF uses
    # "models--owner--repo" prefixes, torch.hub uses "owner_repo_branch" and
    # "checkpoints/" — all distinct within hub/.
    _hf_cache = f"{PROJECT}/hf_cache"
    _os.makedirs(_hf_cache, exist_ok=True)
    _os.environ["HF_HOME"] = _hf_cache     # → /project/hf_cache/hub/models--...
    _os.environ["TORCH_HOME"] = _hf_cache  # → /project/hf_cache/hub/facebookresearch_dinov2_main/

    _sys.path.insert(0, f"{REPO}/ml-4m")
    from fourm.vq.vqvae import VQVAE

    device = torch.device("cuda")

    print(f"Loading DINOv2-B/14…", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()

    print(f"Loading 4M DINOv2 VQVAE tokenizer…", flush=True)
    vqvae = VQVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448").to(device).eval()

    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    images_root = Path(PROJECT) / "data" / split / "things" / "rgb"
    out_root = Path(PROJECT) / "data" / split / "things" / "tok_dinov2@224"
    out_root.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(images_root.glob("shard_*.tar"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard_*.tar files found in {images_root}. "
            "Raw THINGS JPEG images must be in WDS tar format with {{key}}.jpg entries."
        )

    for shard_path in shard_paths:
        out_path = out_root / shard_path.name
        if out_path.exists():
            print(f"  Skipping {shard_path.name} (already exists)", flush=True)
            continue

        print(f"  Processing {shard_path.name}…", flush=True)
        writer = wds.TarWriter(str(out_path))
        n = 0
        with tarfile.open(shard_path) as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".jpg"):
                    continue
                key = member.name.split(".")[0]
                buf = tf.extractfile(member).read()
                img = Image.open(io.BytesIO(buf)).convert("RGB")
                img_t = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

                with torch.no_grad():
                    feats = dino.forward_features(img_t)
                    patch = feats["x_norm_patchtokens"]              # (1, 256, 768)
                    patch = patch.reshape(1, 16, 16, 768).permute(0, 3, 1, 2)  # (1, 768, 16, 16)
                    tok = vqvae.tokenize(patch)                      # (1, 16, 16)
                    tok = tok.reshape(1, 256).cpu().numpy().astype(np.int16)    # (1, 256)

                npy_buf = io.BytesIO()
                np.save(npy_buf, tok)
                writer.write({"__key__": key, "npy": npy_buf.getvalue()})
                n += 1

        writer.close()
        print(f"    Wrote {n} samples → {out_path}", flush=True)

    project_volume.commit()
    print(f"Done. tok_dinov2@224 shards committed for {split} split.", flush=True)


@app.local_entrypoint()
def main(
    condition: str = "",
    dryrun: bool = False,
    large_gpu: bool = False,
    mode: str = "train",
    loss_type: str = "rgb",
    treatment: str = "",
) -> None:
    """
    mode: train | dryrun | collect | fit | plot | tokenize_dinov2 | eval_dinov2

    --condition defaults to "" which means all conditions for collect/fit/plot,
    and is required for train/dryrun.
    --treatment for plot mode: specific treatment condition for gap plot;
    empty = auto-generate gap plots for all non-control conditions.

    Examples:
      modal run 4m_training/modal_train.py --condition pixel_meg
      modal run 4m_training/modal_train.py --condition pixel_dinov2
      modal run 4m_training/modal_train.py --condition rgb_only_pure_all2all
      modal run 4m_training/modal_train.py --mode tokenize_dinov2
      modal run 4m_training/modal_train.py --mode collect
      modal run 4m_training/modal_train.py --mode collect --condition pixel_meg
      modal run 4m_training/modal_train.py --mode fit --loss_type depth
      modal run 4m_training/modal_train.py --mode fit --condition rgb_only --loss_type depth
      modal run 4m_training/modal_train.py --mode plot
      modal run 4m_training/modal_train.py --mode plot --treatment pixel_meg
      modal run 4m_training/modal_train.py --large_gpu --condition pixel_meg
    """
    if mode == "eval_dinov2":
        eval_dinov2_job.remote()
    elif mode == "tokenize_dinov2":
        print("Downloading DINOv2 models to project volume (CPU)…")
        download_dinov2_models.remote()
        print("Tokenizing THINGS images with DINOv2 (train + val splits)…")
        tokenize_things_dinov2.remote(split="train")
        tokenize_things_dinov2.remote(split="val")
    elif mode == "collect":
        collect_job.remote(condition=condition)
    elif mode == "fit":
        fit_job.remote(condition=condition, loss_type=loss_type)
    elif mode == "plot":
        plot_job.remote(conditions=condition, treatment=treatment)
    elif dryrun:
        print("dryrun")
        dryrun_job.remote(condition=condition)
    elif large_gpu:
        print("launching on A100-80GB with --large_gpu batch sizes")
        train_large.remote(condition=condition)
    else:
        train.remote(condition=condition, large_gpu=large_gpu)
