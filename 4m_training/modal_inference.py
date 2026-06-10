"""Modal wrapper for multi-model RGB→Depth inference comparison.

Loads THINGS samples and tokenizers once, then runs each model in sequence —
avoiding repeated Modal startup and HuggingFace download overhead.

Each generated image is saved independently so figures can be composed later.
A separate CPU function (assemble_figure_cpu) reads the saved images and builds
a grid figure without needing a GPU.

Examples:
  # Compare two model sizes, pin the same 4 samples, auto-assemble figure
  modal run 4m_training/modal_inference.py \\
    --names pixel_only_128,pixel_only_512 \\
    --checkpoints /project/output/pixel_only/2layer/dim128/checkpoints/checkpoint-latest.pth,/project/output/pixel_only/3layer/dim512/checkpoints/checkpoint-latest.pth \\
    --configs /opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-2e-2d-scaling.yaml,/opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-3e-3d-scaling.yaml \\
    --sample_key 000042,000117,000203,000389

  # CPU-only run (cheaper, slower — good for debugging)
  modal run 4m_training/modal_inference.py --cpu ...same args...

  # MEG-conditioned run for one model
  modal run 4m_training/modal_inference.py \\
    --names meg_rgb_only,meg_rgb_plus_meg \\
    --checkpoints /project/output/.../checkpoint-latest.pth,/project/output/.../checkpoint-latest.pth \\
    --configs ...meg-rvq0.yaml,...meg-rvq0.yaml \\
    --sample_key 000042,000117,000203,000389 \\
    --include_meg_for meg_rgb_plus_meg

  # Re-assemble figure from already-saved PNGs (no GPU, very fast)
  modal run 4m_training/modal_inference.py \\
    --mode plot_only --names pixel_only_128,pixel_only_512
"""

from __future__ import annotations

import importlib.util as _ilu

import modal

def _load_modal_image():
    for path in (
        __import__("pathlib").Path("/opt/repo/4m_training/_modal_load.py"),
        __import__("pathlib").Path(__file__).resolve().parent / "_modal_load.py",
    ):
        if not path.is_file():
            continue
        spec = _ilu.spec_from_file_location("_modal_load", path)
        loader = _ilu.module_from_spec(spec)
        spec.loader.exec_module(loader)
        return loader.load_modal_image()
    raise ImportError("_modal_load.py not found")

_mi = _load_modal_image()
REPO = _mi.REPO
train_image = _mi.train_image
ensure_fourm = _mi.ensure_fourm

app = modal.App("4m-inference")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

# Shared kwargs for both CPU and GPU variants
_common = dict(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=1800,
)


def _run_inference(
    names, checkpoints, configs,
    things_root, shard_idx, n_samples, sample_key,
    include_meg_for, meg_source,
    include_dinov2_for, dinov2_source,
    include_eeg_for, eeg_source,
    output_dir, tokenizer_cache,
    device_str,
):
    ensure_fourm()

    import json
    import sys as _sys
    _sys.path.insert(0, f"{REPO}/ml-4m")
    _sys.path.insert(0, f"{REPO}/4m_training/lib")

    import os as _os
    if tokenizer_cache:
        _os.makedirs(tokenizer_cache, exist_ok=True)
        _os.environ["HF_HOME"] = tokenizer_cache

    from pathlib import Path
    import sys as _sys2
    _sys2.stdout.reconfigure(line_buffering=True)

    import numpy as np
    import torch
    from PIL import Image as PILImage

    print("Importing fourm modules…", flush=True)
    import fourm_neural_modalities  # noqa: F401 — registers neural modalities
    from fourm.models.generate import (
        GenerationSampler, build_chained_generation_schedules,
        init_empty_target_modality, init_full_input_modality,
    )
    from fourm.vq.vqvae import DiVAE, VQVAE

    print("Loading run_things_inference helpers…", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_things_inference", f"{REPO}/ml-4m/run_things_inference.py"
    )
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

    device = torch.device(device_str)

    # ---- Load shared tokenizers (once for all models) ----
    print(f"Loading RGB tokenizer on {device_str}…", flush=True)
    tok_rgb_dec = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").to(device).eval()
    print("Loading depth tokenizer…", flush=True)
    tok_depth_dec = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_depth_8k_224-448").to(device).eval()
    tok_dinov2_dec = None
    if include_dinov2_for:
        print("Loading DINOv2 VQVAE tokenizer…", flush=True)
        tok_dinov2_dec = VQVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448").to(device).eval()
    print("Tokenizers ready.", flush=True)

    sample_keys_list = [k.strip() for k in sample_key.split(",")] if sample_key else None
    meg_src  = meg_source    if include_meg_for    else None
    dino_src = dinov2_source if include_dinov2_for else None
    eeg_src  = eeg_source    if include_eeg_for    else None
    print(f"Loading THINGS samples from shard {shard_idx} "
          f"(meg={meg_src}, dino={dino_src}, eeg={eeg_src})…", flush=True)
    samples = infer.load_things_samples(
        Path(things_root), shard_idx, n_samples,
        sample_keys=sample_keys_list,
        meg_source=meg_src,
        dinov2_source=dino_src,
        eeg_source=eeg_src,
    )
    print(f"  Keys: {[s['key'] for s in samples]}", flush=True)

    _os.makedirs(output_dir, exist_ok=True)

    def _save_img(arr: np.ndarray, path: Path, is_depth: bool) -> None:
        """Save float [0,1] array as PNG. Depth saved as L (grayscale) for colormap flexibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_depth:
            pil = PILImage.fromarray((arr * 255).astype(np.uint8), mode="L")
        else:
            pil = PILImage.fromarray((arr * 255).astype(np.uint8))
        pil.save(path)

    # ---- Run each model in sequence ----
    for name, checkpoint, config in zip(names, checkpoints, configs):
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {name}", flush=True)
        use_meg    = name in include_meg_for
        use_dinov2 = name in include_dinov2_for
        use_eeg    = name in include_eeg_for

        print(f"  Building model from {Path(config).name}…", flush=True)
        model, modality_info = infer.build_fm_model(Path(config), Path(checkpoint), device)
        sampler = GenerationSampler(model)

        model_dir = Path(output_dir) / name
        model_dir.mkdir(parents=True, exist_ok=True)

        image_types = ["rgb", "depth_gt", "depth_pred"]
        if use_dinov2:
            image_types = ["rgb", "dino_gt", "depth_gt", "depth_pred"]
        colormaps = {t: ("plasma" if "depth" in t else "color") for t in image_types}

        keys_processed = []
        for sample in samples:
            key = sample["key"]
            print(f"  Sample key={key}…", flush=True)

            tok_rgb_t   = torch.tensor(sample["tok_rgb"],   dtype=torch.int64).unsqueeze(0).to(device)
            tok_depth_t = torch.tensor(sample["tok_depth"], dtype=torch.int64).unsqueeze(0).to(device)

            mod_dict: dict = {"tok_rgb@224": {"tensor": tok_rgb_t}}
            mod_dict = init_full_input_modality(mod_dict, modality_info, "tok_rgb@224", device)

            meg_cond: list[str] = []
            if use_meg and "meg_rvq" in sample:
                mod_dict, meg_cond = infer.add_meg_input(
                    mod_dict, sample["meg_rvq"], modality_info, device
                )

            eeg_cond: list[str] = []
            if use_eeg and "eeg_tokens" in sample:
                mod_dict, eeg_cond = infer.add_eeg_input(
                    mod_dict, sample["eeg_tokens"], modality_info, device
                )

            dino_cond: list[str] = []
            if use_dinov2 and "tok_dinov2" in sample:
                tok_dinov2_t = torch.tensor(
                    sample["tok_dinov2"].reshape(256), dtype=torch.int64
                ).unsqueeze(0).to(device)
                mod_dict["tok_dinov2@224"] = {"tensor": tok_dinov2_t}
                mod_dict = init_full_input_modality(mod_dict, modality_info, "tok_dinov2@224", device)
                dino_cond = ["tok_dinov2@224"]

            mod_dict = init_empty_target_modality(
                mod_dict, modality_info, "tok_depth@224",
                batch_size=1, num_tokens=196, device=device,
            )
            schedule = build_chained_generation_schedules(
                cond_domains=["tok_rgb@224"] + meg_cond + eeg_cond + dino_cond,
                target_domains=["tok_depth@224"],
                tokens_per_target=[196],
                autoregression_schemes=["roar"],
                decoding_steps=[1],
                token_decoding_schedules=["linear"],
                temps=[0.01],
                temp_schedules=["constant"],
                cfg_scales=[2.0],
                cfg_schedules=["constant"],
                cfg_grow_conditioning=False,
                modality_info=modality_info,
            )
            print(f"    Generating…", flush=True)
            with torch.no_grad():
                mod_dict = sampler.generate(mod_dict, schedule, seed=42, verbose=True)

            print(f"    Decoding and saving…", flush=True)
            pred_tokens = mod_dict["tok_depth@224"]["tensor"]

            rgb_img    = infer.decode_to_image(tok_rgb_dec,   tok_rgb_t,   is_rgb=True)
            gt_depth   = infer.decode_to_image(tok_depth_dec, tok_depth_t, is_rgb=False)
            pred_depth = infer.decode_to_image(tok_depth_dec, pred_tokens, is_rgb=False)

            _save_img(rgb_img,    model_dir / f"{key}_rgb.png",        is_depth=False)
            _save_img(gt_depth,   model_dir / f"{key}_depth_gt.png",   is_depth=True)
            _save_img(pred_depth, model_dir / f"{key}_depth_pred.png", is_depth=True)

            if use_dinov2:
                dino_gt = infer.decode_dinov2(tok_dinov2_dec, tok_dinov2_t)
                _save_img(dino_gt, model_dir / f"{key}_dino_gt.png", is_depth=False)

            keys_processed.append(key)

        # Metadata for assemble_figure_cpu
        meta = {
            "model_name":  name,
            "checkpoint":  checkpoint,
            "config":      config,
            "keys":        keys_processed,
            "image_types": image_types,
            "colormaps":   colormaps,
            "conditions": {
                "meg":    meg_source    if use_meg    else None,
                "dinov2": dinov2_source if use_dinov2 else None,
                "eeg":    eeg_source    if use_eeg    else None,
            },
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Wrote meta.json for {name}.", flush=True)

        del model, sampler
        if device_str == "cuda":
            torch.cuda.empty_cache()

    project_volume.commit()
    print(f"\nAll images committed to volume at {output_dir}/")


@app.function(**_common, gpu="T4")
def inference_job_gpu(
    names, checkpoints, configs, things_root, shard_idx, n_samples,
    sample_key, include_meg_for, meg_source,
    include_dinov2_for, dinov2_source,
    include_eeg_for, eeg_source,
    output_dir, tokenizer_cache,
):
    _run_inference(
        names, checkpoints, configs, things_root, shard_idx, n_samples,
        sample_key, include_meg_for, meg_source,
        include_dinov2_for, dinov2_source,
        include_eeg_for, eeg_source,
        output_dir, tokenizer_cache,
        device_str="cuda",
    )


@app.function(**_common)
def inference_job_cpu(
    names, checkpoints, configs, things_root, shard_idx, n_samples,
    sample_key, include_meg_for, meg_source,
    include_dinov2_for, dinov2_source,
    include_eeg_for, eeg_source,
    output_dir, tokenizer_cache,
):
    _run_inference(
        names, checkpoints, configs, things_root, shard_idx, n_samples,
        sample_key, include_meg_for, meg_source,
        include_dinov2_for, dinov2_source,
        include_eeg_for, eeg_source,
        output_dir, tokenizer_cache,
        device_str="cpu",
    )


@app.function(image=train_image, volumes={PROJECT: project_volume}, timeout=600, memory=4 * 1024)
def assemble_figure_cpu(
    output_dir: str,
    model_names: list[str],
    sample_keys: list[str] | None = None,
    n_keys: int = 2,
    transpose: bool = False,
) -> None:
    """Assemble a single comparison figure across all models.

    Default layout (transpose=False):
      rows    = reference rows (rgb, depth_gt) + one depth_pred row per model
      columns = selected sample keys  (default: first n_keys=2)

    Transposed layout (transpose=True):
      rows    = selected sample keys
      columns = rgb | depth_gt | depth_pred per model

    Depth images are rendered with the plasma colormap.
    Output: {output_dir}/comparison_figure.png
    """
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as PILImage

    output_dir_path = Path(output_dir)

    # ── Load per-model metadata ───────────────────────────────────────────
    metas = {}
    for model_name in model_names:
        meta_path = output_dir_path / model_name / "meta.json"
        if not meta_path.exists():
            print(f"  Skipping {model_name}: meta.json not found", flush=True)
            continue
        metas[model_name] = json.loads(meta_path.read_text())

    if not metas:
        print("No valid model directories found — nothing to plot.", flush=True)
        return

    # Keys to display: explicit list, or first n_keys from the first model
    first_meta = next(iter(metas.values()))
    display_keys = sample_keys or first_meta["keys"][:n_keys]
    print(f"  Displaying keys: {display_keys}", flush=True)

    # Reference images (rgb, depth_gt) are identical across models — use first model
    ref_dir = output_dir_path / next(iter(metas))
    ref_types = ["rgb", "depth_gt"]
    ref_labels = {"rgb": "RGB", "depth_gt": "Depth GT"}

    model_names_ordered = list(metas.keys())

    def _show(ax, img_path: Path, itype: str) -> None:
        if not img_path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return
        arr = np.array(PILImage.open(img_path))
        ax.imshow(arr, cmap="plasma", vmin=0, vmax=255) if "depth" in itype else ax.imshow(arr)
        ax.axis("off")

    def _side_label(ax, text: str, bold: bool = False) -> None:
        ax.text(
            -0.08, 0.5, text,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold" if bold else "normal",
            va="center", ha="right", clip_on=False,
        )

    if not transpose:
        # rows = ref types + models,  cols = keys
        n_rows = len(ref_types) + len(model_names_ordered)
        n_cols = len(display_keys)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False)

        for row, itype in enumerate(ref_types):
            for col, key in enumerate(display_keys):
                _show(axes[row][col], ref_dir / f"{key}_{itype}.png", itype)
            _side_label(axes[row][0], ref_labels[itype], bold=True)

        for col, key in enumerate(display_keys):
            axes[0][col].set_title(f"key {key}", fontsize=13)

        for m_idx, (model_name, meta) in enumerate(metas.items()):
            row = len(ref_types) + m_idx
            for col, key in enumerate(display_keys):
                _show(axes[row][col], output_dir_path / model_name / f"{key}_depth_pred.png", "depth")
            cond = meta.get("conditions", {})
            cond_str = " ".join(f"+{k}" for k, v in cond.items() if v)
            _side_label(axes[row][0], model_name + (f"\n{cond_str}" if cond_str else ""))

    else:
        # rows = keys,  cols = ref types + models
        n_rows = len(display_keys)
        n_cols = len(ref_types) + len(model_names_ordered)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.5 * n_rows), squeeze=False)

        for row, key in enumerate(display_keys):
            _side_label(axes[row][0], f"key {key}", bold=True)
            for col, itype in enumerate(ref_types):
                _show(axes[row][col], ref_dir / f"{key}_{itype}.png", itype)
            for m_idx, (model_name, meta) in enumerate(metas.items()):
                col = len(ref_types) + m_idx
                _show(axes[row][col], output_dir_path / model_name / f"{key}_depth_pred.png", "depth")

        for col, itype in enumerate(ref_types):
            axes[0][col].set_title(ref_labels[itype], fontsize=13)
        for m_idx, (model_name, meta) in enumerate(metas.items()):
            col = len(ref_types) + m_idx
            cond = meta.get("conditions", {})
            cond_str = " ".join(f"+{k}" for k, v in cond.items() if v)
            axes[0][col].set_title(model_name + (f"\n{cond_str}" if cond_str else ""), fontsize=13)

    plt.tight_layout()
    out = output_dir_path / "comparison_figure.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}", flush=True)

    project_volume.commit()
    print("Figure assembly complete.")


@app.function(image=train_image, volumes={PROJECT: project_volume}, gpu="T4", timeout=60 * 60 * 3, memory=16 * 1024)
def average_depth_gt_gpu(
    splits: list[str] | None = None,
    things_base: str = f"{PROJECT}/data",
    output_dir: str = f"{PROJECT}/output/inference",
    batch_size: int = 8,
    tokenizer_cache: str = f"{PROJECT}/hf_cache",
) -> None:
    """Decode every THINGS tok_depth shard and save the per-pixel mean as a plasma plot.

    Iterates all tok_depth shards in {things_base}/{split}/things/tok_depth/ for each
    requested split (default: train + val), decodes with DiVAE (25 steps), accumulates
    a float64 running sum, then saves {output_dir}/average_depth_gt.png.

    ~26k images at batch_size=8 takes roughly 30–60 min on a T4.
    """
    ensure_fourm()

    import io
    import os as _os
    import sys as _sys
    import tarfile
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image as PILImage

    _sys.path.insert(0, f"{REPO}/ml-4m")
    if tokenizer_cache:
        _os.makedirs(tokenizer_cache, exist_ok=True)
        _os.environ["HF_HOME"] = tokenizer_cache

    from fourm.vq.vqvae import DiVAE

    device = torch.device("cuda")
    print("Loading depth tokenizer…", flush=True)
    tok_depth_dec = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_depth_8k_224-448").to(device).eval()

    if splits is None:
        splits = ["train", "val"]

    running_sum   = None   # (H, W) float64 accumulator
    running_count = 0

    for split in splits:
        depth_dir = Path(things_base) / split / "things" / "tok_depth"
        shard_paths = sorted(depth_dir.glob("shard_*.tar"))
        if not shard_paths:
            print(f"  No shards found in {depth_dir} — skipping", flush=True)
            continue
        print(f"  {split}: {len(shard_paths)} shards", flush=True)

        for shard_path in shard_paths:
            token_batch: list[np.ndarray] = []

            with tarfile.open(shard_path) as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".npy")]
                for member in members:
                    buf = tf.extractfile(member).read()
                    arr = np.load(io.BytesIO(buf)).astype(np.int64).reshape(196)
                    token_batch.append(arr)

                    if len(token_batch) == batch_size:
                        tokens_t = torch.tensor(
                            np.stack(token_batch), dtype=torch.long
                        ).reshape(-1, 14, 14).to(device)
                        with torch.no_grad():
                            imgs = tok_depth_dec.decode_tokens(tokens_t, timesteps=25, image_size=224)
                        imgs_np = imgs.squeeze(1).cpu().float().numpy()  # (B, 224, 224)
                        for img in imgs_np:
                            lo, hi = img.min(), img.max()
                            if hi > lo:
                                img = (img - lo) / (hi - lo)
                            if running_sum is None:
                                running_sum = img.astype(np.float64)
                            else:
                                running_sum += img.astype(np.float64)
                            running_count += 1
                        token_batch = []

            # flush remaining partial batch
            if token_batch:
                tokens_t = torch.tensor(
                    np.stack(token_batch), dtype=torch.long
                ).reshape(-1, 14, 14).to(device)
                with torch.no_grad():
                    imgs = tok_depth_dec.decode_tokens(tokens_t, timesteps=25, image_size=224)
                imgs_np = imgs.squeeze(1).cpu().float().numpy()
                for img in imgs_np:
                    lo, hi = img.min(), img.max()
                    if hi > lo:
                        img = (img - lo) / (hi - lo)
                    if running_sum is None:
                        running_sum = img.astype(np.float64)
                    else:
                        running_sum += img.astype(np.float64)
                    running_count += 1

            print(f"    {shard_path.name}: {running_count} total so far", flush=True)

    if running_count == 0 or running_sum is None:
        print("No depth maps decoded — nothing to save.", flush=True)
        return

    avg = (running_sum / running_count).astype(np.float32)  # [0, 1]
    print(f"  Averaged {running_count} depth maps.", flush=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(avg, cmap="plasma", vmin=0, vmax=1)
    ax.set_title(f"Average GT depth  (n={running_count:,})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    _os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir) / "average_depth_gt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}", flush=True)

    project_volume.commit()


@app.local_entrypoint()
def main(
    names: str = "",
    checkpoints: str = "",
    configs: str = "",
    things_root: str = f"{PROJECT}/data/val/things",
    shard_idx: int = 0,
    n_samples: int = 4,
    sample_key: str = "",
    include_meg_for: str = "",
    meg_source: str = "tok_meg_avg",
    include_dinov2_for: str = "",
    dinov2_source: str = "tok_dinov2@224",
    include_eeg_for: str = "",
    eeg_source: str = "tok_eeg",
    output_dir: str = f"{PROJECT}/output/inference",
    tokenizer_cache: str = f"{PROJECT}/hf_cache",
    cpu: bool = False,
    no_plot: bool = False,
    mode: str = "infer",
    n_plot_keys: int = 2,
    transpose: bool = False,
) -> None:
    """
    --names              Comma-separated labels for each model (used as output filenames)
    --checkpoints        Comma-separated checkpoint .pth paths (same order as --names)
    --configs            Comma-separated model YAML config paths (same order as --names)
    --sample_key         Comma-separated shard keys to pin across all models (e.g. 000042,000117);
                         also selects which keys appear in the figure when used with plot_only
    --include_meg_for    Comma-separated subset of --names that should use RGB+MEG input
    --meg_source         MEG shard folder: tok_meg_avg (default) or tok_meg
    --include_dinov2_for Comma-separated subset of --names that should use RGB+DINOv2 input
    --dinov2_source      DINOv2 shard folder (default: tok_dinov2@224)
    --include_eeg_for    Comma-separated subset of --names that should use RGB+EEG input
    --eeg_source         EEG shard folder (default: tok_eeg)
    --output_dir         Directory in the project volume where PNGs are saved
    --cpu                Run inference on CPU instead of T4 GPU (slower but cheaper)
    --no_plot            Skip auto-assembling grid figures after inference
    --n_plot_keys        Number of images (columns) to show in the figure (default: 2);
                         ignored when --sample_key is given explicitly
    --mode               infer (default) | plot_only | avg_depth
                         avg_depth: decode all THINGS tok_depth shards and save average depth map
    --transpose          Transpose comparison figure (rows=keys, cols=rgb|depth_gt|models)
    """
    names_list       = [n.strip() for n in names.split(",")       if n.strip()]
    checkpoints_list = [c.strip() for c in checkpoints.split(",") if c.strip()]
    configs_list     = [c.strip() for c in configs.split(",")     if c.strip()]

    if mode == "avg_depth":
        print("Computing average THINGS ground truth depth (train + val)…")
        # things_root is e.g. /project/data/val/things — derive the base from it
        things_base = str(__import__("pathlib").Path(things_root).parents[1])
        average_depth_gt_gpu.remote(
            output_dir=output_dir,
            things_base=things_base,
            tokenizer_cache=tokenizer_cache,
        )
        return

    if mode == "plot_only":
        if not names_list:
            raise SystemExit("--names is required for plot_only mode")
        keys_list = [k.strip() for k in sample_key.split(",") if k.strip()] or None
        print(f"Assembling comparison figure for: {names_list}")
        assemble_figure_cpu.remote(
            output_dir=output_dir,
            model_names=names_list,
            sample_keys=keys_list,
            n_keys=n_plot_keys,
            transpose=transpose,
        )
        return

    if not names_list:
        raise SystemExit("--names is required (comma-separated model labels)")
    if len(checkpoints_list) != len(names_list):
        raise SystemExit("--checkpoints must have the same number of entries as --names")
    if len(configs_list) != len(names_list):
        raise SystemExit("--configs must have the same number of entries as --names")

    meg_for_set    = {n.strip() for n in include_meg_for.split(",")    if n.strip()}
    dinov2_for_set = {n.strip() for n in include_dinov2_for.split(",") if n.strip()}
    eeg_for_set    = {n.strip() for n in include_eeg_for.split(",")    if n.strip()}
    for label, s in [
        ("--include_meg_for",    meg_for_set),
        ("--include_dinov2_for", dinov2_for_set),
        ("--include_eeg_for",    eeg_for_set),
    ]:
        unknown = s - set(names_list)
        if unknown:
            raise SystemExit(f"{label} references unknown names: {unknown}")

    kwargs = dict(
        names=names_list,
        checkpoints=checkpoints_list,
        configs=configs_list,
        things_root=things_root,
        shard_idx=shard_idx,
        n_samples=n_samples,
        sample_key=sample_key,
        include_meg_for=meg_for_set,
        meg_source=meg_source,
        include_dinov2_for=dinov2_for_set,
        dinov2_source=dinov2_source,
        include_eeg_for=eeg_for_set,
        eeg_source=eeg_source,
        output_dir=output_dir,
        tokenizer_cache=tokenizer_cache,
    )

    print(f"Running {len(names_list)} model(s) on {'CPU' if cpu else 'T4 GPU'}: {names_list}")
    if cpu:
        inference_job_cpu.remote(**kwargs)
    else:
        inference_job_gpu.remote(**kwargs)

    if not no_plot:
        keys_list = [k.strip() for k in sample_key.split(",") if k.strip()] or None
        print("Assembling comparison figure…")
        assemble_figure_cpu.remote(
            output_dir=output_dir,
            model_names=names_list,
            sample_keys=keys_list,
            n_keys=n_plot_keys,
            transpose=transpose,
        )
