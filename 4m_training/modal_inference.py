"""Modal wrapper for multi-model RGB→Depth inference comparison.

Loads THINGS samples and tokenizers once, then runs each model in sequence —
avoiding repeated Modal startup and HuggingFace download overhead.

Examples:
  # Compare two model sizes, pin the same 4 samples
  modal run 4m_training/modal_inference.py \\
    --names pixel_only_128,pixel_only_512 \\
    --checkpoints /project/output/pixel_only/2layer/dim128/checkpoints/checkpoint-latest.pth,/project/output/pixel_only/3layer/dim512/checkpoints/checkpoint-latest.pth \\
    --configs /opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-2e-2d-scaling.yaml,/opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-3e-3d-scaling.yaml \\
    --sample_key 000042,000117,000203,000389

  # CPU-only run (cheaper, slower — good for debugging)
  modal run 4m_training/modal_inference.py --cpu ...same args...

  # Same samples, add MEG-conditioned run for one model
  modal run 4m_training/modal_inference.py \\
    --names meg_rgb_only,meg_rgb_plus_meg \\
    --checkpoints /project/output/pixel_meg_rvq0/3layer/dim512/checkpoints/checkpoint-latest.pth,/project/output/pixel_meg_rvq0/3layer/dim512/checkpoints/checkpoint-latest.pth \\
    --configs /opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-3e-3d-scaling-meg-rvq0.yaml,/opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-3e-3d-scaling-meg-rvq0.yaml \\
    --sample_key 000042,000117,000203,000389 \\
    --include_meg_for meg_rgb_plus_meg
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
    include_meg_for, meg_source, output_dir, tokenizer_cache,
    device_str,
):
    ensure_fourm()

    import sys as _sys
    _sys.path.insert(0, f"{REPO}/ml-4m")
    _sys.path.insert(0, f"{REPO}/4m_training/lib")

    import os as _os
    if tokenizer_cache:
        _os.makedirs(tokenizer_cache, exist_ok=True)
        _os.environ["HF_HOME"] = tokenizer_cache

    from pathlib import Path
    import sys as _sys2
    _sys2.stdout.reconfigure(line_buffering=True)  # flush every line — Modal buffers by default

    import matplotlib
    matplotlib.use("Agg")  # headless backend — must be set before importing pyplot
    import matplotlib.pyplot as plt

    import torch

    print("Importing fourm modules…", flush=True)
    import fourm_neural_modalities  # noqa: F401 — registers neural modalities
    from fourm.models.generate import (
        GenerationSampler, build_chained_generation_schedules,
        init_empty_target_modality, init_full_input_modality,
    )
    from fourm.vq.vqvae import DiVAE

    print("Loading run_things_inference helpers…", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_things_inference", f"{REPO}/ml-4m/run_things_inference.py"
    )
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

    device = torch.device(device_str)

    # ---- Load shared resources (once for all models) ----
    print(f"Loading RGB tokenizer on {device_str}…", flush=True)
    tok_rgb_dec = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").to(device).eval()
    print("Loading depth tokenizer…", flush=True)
    tok_depth_dec = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_depth_8k_224-448").to(device).eval()
    print("Tokenizers ready.", flush=True)

    sample_keys_list = [k.strip() for k in sample_key.split(",")] if sample_key else None
    meg_src = meg_source if include_meg_for else None
    print(f"Loading THINGS samples from shard {shard_idx} (meg_src={meg_src})…", flush=True)
    samples = infer.load_things_samples(
        Path(things_root), shard_idx, n_samples,
        sample_keys=sample_keys_list, meg_source=meg_src,
    )
    print(f"  Keys: {[s['key'] for s in samples]}", flush=True)

    _os.makedirs(output_dir, exist_ok=True)

    # ---- Run each model in sequence ----
    for name, checkpoint, config in zip(names, checkpoints, configs):
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {name}", flush=True)
        use_meg = name in include_meg_for

        print(f"  Building model from {Path(config).name}…", flush=True)
        model, modality_info = infer.build_fm_model(Path(config), Path(checkpoint), device)
        sampler = GenerationSampler(model)

        n = len(samples)
        input_label = f"RGB+MEG ({meg_source})" if use_meg else "RGB"
        fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n), squeeze=False)
        col_titles = ["RGB (decoded)", "Depth GT", f"Depth pred\n({input_label} → depth)"]

        for row, sample in enumerate(samples):
            print(f"  Sample {row+1}/{n} key={sample['key']}…", flush=True)
            tok_rgb_t   = torch.tensor(sample["tok_rgb"],   dtype=torch.int64).unsqueeze(0).to(device)
            tok_depth_t = torch.tensor(sample["tok_depth"], dtype=torch.int64).unsqueeze(0).to(device)

            mod_dict: dict = {"tok_rgb@224": {"tensor": tok_rgb_t}}
            mod_dict = init_full_input_modality(mod_dict, modality_info, "tok_rgb@224", device)

            meg_cond: list[str] = []
            if use_meg and "meg_rvq" in sample:
                mod_dict, meg_cond = infer.add_meg_input(
                    mod_dict, sample["meg_rvq"], modality_info, device
                )

            mod_dict = init_empty_target_modality(
                mod_dict, modality_info, "tok_depth@224",
                batch_size=1, num_tokens=196, device=device,
            )
            schedule = build_chained_generation_schedules(
                cond_domains=["tok_rgb@224"] + meg_cond,
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
            print(f"    Generating ({len(schedule)}-step schedule)…", flush=True)
            with torch.no_grad():
                mod_dict = sampler.generate(mod_dict, schedule, seed=42, verbose=True)

            print(f"    Decoding tokens…", flush=True)
            pred_tokens = mod_dict["tok_depth@224"]["tensor"]
            rgb_img    = infer.decode_to_image(tok_rgb_dec,   tok_rgb_t,   is_rgb=True)
            gt_depth   = infer.decode_to_image(tok_depth_dec, tok_depth_t, is_rgb=False)
            pred_depth = infer.decode_to_image(tok_depth_dec, pred_tokens, is_rgb=False)

            for col, (img, title) in enumerate(zip([rgb_img, gt_depth, pred_depth], col_titles)):
                ax = axes[row][col]
                ax.imshow(img) if col == 0 else ax.imshow(img, cmap="plasma")
                if row == 0:
                    ax.set_title(title, fontsize=10)
                ax.axis("off")
            axes[row][0].set_ylabel(f"key {sample['key']}", fontsize=8,
                                    rotation=0, labelpad=50, va="center")
            print(f"    Row {row+1} done.", flush=True)

        print(f"  Saving figure…", flush=True)
        fig.suptitle(name, fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = Path(output_dir) / f"{name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}", flush=True)

        del model, sampler
        if device_str == "cuda":
            torch.cuda.empty_cache()

    project_volume.commit()
    print(f"\nAll outputs committed to volume at {output_dir}/")


@app.function(**_common, gpu="T4")
def inference_job_gpu(
    names, checkpoints, configs, things_root, shard_idx, n_samples,
    sample_key, include_meg_for, meg_source, output_dir, tokenizer_cache,
):
    _run_inference(
        names, checkpoints, configs, things_root, shard_idx, n_samples,
        sample_key, include_meg_for, meg_source, output_dir, tokenizer_cache,
        device_str="cuda",
    )


@app.function(**_common)
def inference_job_cpu(
    names, checkpoints, configs, things_root, shard_idx, n_samples,
    sample_key, include_meg_for, meg_source, output_dir, tokenizer_cache,
):
    _run_inference(
        names, checkpoints, configs, things_root, shard_idx, n_samples,
        sample_key, include_meg_for, meg_source, output_dir, tokenizer_cache,
        device_str="cpu",
    )


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
    output_dir: str = f"{PROJECT}/output/inference",
    tokenizer_cache: str = f"{PROJECT}/hf_cache",
    cpu: bool = False,
) -> None:
    """
    --names          Comma-separated labels for each model (used as output filenames)
    --checkpoints    Comma-separated checkpoint .pth paths (same order as --names)
    --configs        Comma-separated model YAML config paths (same order as --names)
    --sample_key     Comma-separated shard keys to pin across all models (e.g. 000042,000117)
    --include_meg_for Comma-separated subset of --names that should use RGB+MEG input
    --meg_source     MEG shard folder: tok_meg_avg (default) or tok_meg
    --output_dir     Directory in the project volume where PNGs are saved
    --cpu            Run on CPU instead of T4 GPU (slower but cheaper; good for debugging)
    """
    names_list       = [n.strip() for n in names.split(",")      if n.strip()]
    checkpoints_list = [c.strip() for c in checkpoints.split(",") if c.strip()]
    configs_list     = [c.strip() for c in configs.split(",")    if c.strip()]

    if not names_list:
        raise SystemExit("--names is required (comma-separated model labels)")
    if len(checkpoints_list) != len(names_list):
        raise SystemExit("--checkpoints must have the same number of entries as --names")
    if len(configs_list) != len(names_list):
        raise SystemExit("--configs must have the same number of entries as --names")

    meg_for_set = {n.strip() for n in include_meg_for.split(",") if n.strip()}
    unknown = meg_for_set - set(names_list)
    if unknown:
        raise SystemExit(f"--include_meg_for references unknown names: {unknown}")

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
        output_dir=output_dir,
        tokenizer_cache=tokenizer_cache,
    )

    print(f"Running {len(names_list)} model(s) on {'CPU' if cpu else 'T4 GPU'}: {names_list}")
    if cpu:
        inference_job_cpu.remote(**kwargs)
    else:
        inference_job_gpu.remote(**kwargs)
