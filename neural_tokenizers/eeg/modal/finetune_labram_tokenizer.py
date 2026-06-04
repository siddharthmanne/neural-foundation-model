"""
Finetune LaBraM's vqnsp.pth on THINGS-EEG2 (and optionally EEG1) HDF5 data.

Reads manifests produced by convert_eeg_to_labram_hdf5.py and/or
convert_eeg1_to_labram_hdf5.py, patches run_vqnsp_training.py at runtime to
inject the correct HDF5 paths (the script has no --data_path flag; paths are
hardcoded at lines 149-162), strips optimizer/scaler from the checkpoint for a
clean finetune LR schedule, and launches via torchrun.

Run dry run (1 epoch, sub-01 EEG2 only):
    modal run neural_tokenizers/eeg/modal/finetune_labram_tokenizer.py

Run full training (10 epochs, all EEG2 subjects):
    modal run neural_tokenizers/eeg/modal/finetune_labram_tokenizer.py::run_full

Run full training with EEG1 + EEG2 combined (the shipped recipe):
    modal run neural_tokenizers/eeg/modal/finetune_labram_tokenizer.py::run_full_combined
"""

import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import warnings
from datetime import datetime

import modal
from _app import app, data_volume


def _download_labram(dest: str = "/tmp/LaBraM") -> str:
    """Download LaBraM source archive if not already present. Returns dest path."""
    if not os.path.isdir(dest):
        print("Downloading LaBraM source (~few MB)...")
        tmp_archive = "/tmp/labram.tar.gz"
        urllib.request.urlretrieve(
            "https://github.com/935963004/LaBraM/archive/refs/heads/main.tar.gz",
            tmp_archive,
        )
        with tarfile.open(tmp_archive, "r:gz") as tf:
            tf.extractall("/tmp/labram_extract")
        shutil.move("/tmp/labram_extract/LaBraM-main", dest)
        os.remove(tmp_archive)
        print("LaBraM downloaded.")
    return dest

finetune_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel")
    .pip_install([
        "einops",
        "timm==0.6.12",
        "tensorboardX",
        "h5py",
        "mne",
        "scipy",
        "numpy<2",
        "pyhealth",
    ])
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)


def _patch_run_vqnsp(
    original_path: str,
    train_hdf5_paths: list[str],
    val_hdf5_paths: list[str],
    out_path: str,
) -> None:
    """
    Replace only the datasets_train, datasets_val, and time_window assignments
    in run_vqnsp_training.py. Each variable is replaced independently so that
    intermediate calls (build_pretraining_dataset, etc.) are preserved.
    """
    with open(original_path) as f:
        lines = f.readlines()

    def _find_assignment_span(lines_list, var, search_from=0):
        """Return (start_idx, end_idx_inclusive) of `var = ...` or `var = [...]`."""
        for i in range(search_from, len(lines_list)):
            stripped = lines_list[i].strip()
            if stripped.startswith(f"{var} =") or stripped.startswith(f"{var}="):
                depth = 0
                started = False
                for j in range(i, len(lines_list)):
                    for ch in lines_list[j]:
                        if ch in "([":
                            depth += 1
                            started = True
                        elif ch in ")]" and started:
                            depth -= 1
                    if started and depth == 0:
                        return i, j
                return i, i  # single-line (no brackets) fallback
        return None, None

    def _replace_var(lines_list, var, new_repr, search_from=0):
        """Replace the assignment of var with new_repr, preserving indentation."""
        start, end = _find_assignment_span(lines_list, var, search_from)
        if start is None:
            return lines_list, None
        indent_str = lines_list[start][: len(lines_list[start]) - len(lines_list[start].lstrip())]
        new_line = f"{indent_str}{var} = {new_repr}\n"
        return lines_list[:start] + [new_line] + lines_list[end + 1 :], start

    # All subjects share the same 17-channel layout → ONE montage group.
    # LaBraM's build_pretraining_dataset creates one DataLoader per inner list;
    # the engine only trains on data_loader_train_list[0]. Putting all files in
    # a single group means one DataLoader with all subjects concatenated via
    # ShockDataset's cumulative indexing.
    train_repr = repr([train_hdf5_paths])
    val_repr   = repr([val_hdf5_paths])
    # time_window=8 → window_size = 8 * 200 = 1600 samples. This matches the
    # original vqnsp.pth pretraining config (--input_size 1600) and avoids the
    # decoder shape bug: with time_window=1, quantize is (B, D, 17, 1) and
    # t=1 == decoder.patch_size=1, so input_time_window is misread as a=17
    # instead of t=1, expanding pos_embed to 17*17+1=290 vs 18 actual tokens.
    # With time_window=8, t=8 != patch_size=1, so the correct branch fires.
    time_repr  = repr([8])

    lines, train_pos = _replace_var(lines, "datasets_train", train_repr)
    if train_pos is None:
        raise RuntimeError(f"Could not locate 'datasets_train' in {original_path}.")

    lines, val_pos = _replace_var(lines, "datasets_val", val_repr, search_from=train_pos + 1)
    if val_pos is None:
        raise RuntimeError(f"Could not locate 'datasets_val' in {original_path}.")

    lines, _ = _replace_var(lines, "time_window", time_repr)  # optional; no error if absent

    with open(out_path, "w") as f:
        f.writelines(lines)


def _collect_hdf5_paths(
    eeg2_subjects: list[str] | None,
    include_eeg1: bool,
) -> tuple[list[str], list[str]]:
    """Collect train/val HDF5 paths from EEG2 (and optionally EEG1) manifests.

    Returns (train_paths, val_paths).
    """
    train_paths: list[str] = []
    val_paths: list[str]   = []

    # EEG2
    manifest2_path = "/project/data/labram_input/things-eeg2/manifest.json"
    if not os.path.exists(manifest2_path):
        raise FileNotFoundError(
            f"EEG2 manifest not found at {manifest2_path}. "
            "Run convert_things_eeg2 first."
        )
    with open(manifest2_path) as f:
        manifest2 = json.load(f)

    subjects2 = eeg2_subjects if eeg2_subjects is not None else list(manifest2.keys())
    missing2 = [s for s in subjects2 if s not in manifest2]
    if missing2:
        raise ValueError(f"EEG2 subjects not found in manifest: {missing2}")

    train_paths += [manifest2[s]["train"] for s in subjects2]
    val_paths   += [manifest2[s]["val"]   for s in subjects2]
    print(f"EEG2: {len(subjects2)} subjects")

    # EEG1 (optional)
    if include_eeg1:
        manifest1_path = "/project/data/labram_input/things-eeg1/manifest.json"
        if not os.path.exists(manifest1_path):
            raise FileNotFoundError(
                f"EEG1 manifest not found at {manifest1_path}. "
                "Run convert_things_eeg1 first."
            )
        with open(manifest1_path) as f:
            manifest1 = json.load(f)
        subjects1 = list(manifest1.keys())
        train_paths += [manifest1[s]["train"] for s in subjects1]
        val_paths   += [manifest1[s]["val"]   for s in subjects1]
        print(f"EEG1: {len(subjects1)} subjects")

    print(f"Total HDF5 files: {len(train_paths)} train, {len(val_paths)} val")
    return train_paths, val_paths


@app.function(
    image=finetune_image,
    volumes={"/project": data_volume},
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=3600 * 24,
)
def finetune_labram(
    dry_run: bool = False,
    eeg2_subjects: list[str] | None = None,
    include_eeg1: bool = False,
    warmup_epochs: int = 0,
) -> str:
    """
    Finetune LaBraM vqnsp on THINGS-EEG data.

    dry_run=True:       1 epoch, EEG2 sub-01 only (smoke test).
    dry_run=False:      10 epochs on all specified subjects (shipped: checkpoint-5).
    include_eeg1=True:  add all available EEG1 subjects to the training set.
    warmup_epochs:      linear LR warmup epochs. Use 3+ for combined runs
                        (6x more batches per epoch → more room for early divergence).

    Returns the path to the final checkpoint on the project volume.
    """
    import torch

    warnings.filterwarnings("ignore")

    labram_root = _download_labram()

    if dry_run:
        eeg2_subjects = ["sub-01"]
        include_eeg1 = False
        epochs = 1
        warmup_epochs = 0
    else:
        epochs = 10

    train_hdf5_paths, val_hdf5_paths = _collect_hdf5_paths(eeg2_subjects, include_eeg1)

    # Strip optimizer and scaler so we start with a clean LR schedule.
    # --resume with the full checkpoint would hard-set start_epoch=1 but
    # would also restore the optimizer state from the original pretraining
    # run, which had a different dataset distribution and LR trajectory.
    os.makedirs("/project/data/labram_tokenizer", exist_ok=True)
    model_only_path = "/project/data/labram_tokenizer/vqnsp_model_only.pth"
    if not os.path.exists(model_only_path):
        # git clone inside the image may fail silently (|| true).
        # Fall back to downloading the checkpoint directly from GitHub and
        # caching it on the volume so subsequent runs skip this download.
        vqnsp_src = os.path.join(labram_root, "checkpoints/vqnsp.pth")
        if not os.path.exists(vqnsp_src):
            vqnsp_src = "/project/data/labram_tokenizer/vqnsp_original.pth"
            if not os.path.exists(vqnsp_src):
                _url = (
                    "https://raw.githubusercontent.com/935963004/LaBraM"
                    "/main/checkpoints/vqnsp.pth"
                )
                print(f"Downloading vqnsp.pth from GitHub (~95 MB)...")
                urllib.request.urlretrieve(_url, vqnsp_src)
                print("Download complete.")
                data_volume.commit()
        ckpt = torch.load(vqnsp_src, map_location="cpu")
        torch.save({"model": ckpt["model"]}, model_only_path)
        print(f"Stripped checkpoint saved to {model_only_path}")
    else:
        print(f"Using existing stripped checkpoint at {model_only_path}")

    # Codebook health check: verify the checkpoint has an initialized, L2-normalized
    # codebook before spending GPU time. If either value drifts the pretrained
    # codebook has been silently destroyed (wrong file, partial save, etc.).
    _ckpt_state = torch.load(model_only_path, map_location="cpu")["model"]
    _initted = _ckpt_state.get("quantize.embedding.initted")
    _weight  = _ckpt_state.get("quantize.embedding.weight")
    if _initted is not None:
        _initted_val = _initted.item()
        print(f"[codebook] initted={_initted_val} (must be 1.0)")
        if _initted_val != 1.0:
            raise RuntimeError(f"Codebook not initialized (initted={_initted_val}). Check checkpoint.")
    else:
        print("[codebook] WARNING: quantize.embedding.initted not found in checkpoint")
    if _weight is not None:
        _norm_mean = _weight.norm(dim=-1).mean().item()
        print(f"[codebook] weight norm mean={_norm_mean:.4f} (should be ≈1.0)")
    _cluster_size = _ckpt_state.get("quantize.embedding.cluster_size")
    if _cluster_size is not None:
        _cs_sum = _cluster_size.sum().item()
        _cs_nonzero = (_cluster_size > 0).sum().item()
        print(f"[codebook] cluster_size.sum={_cs_sum:.1f} (expect ~258k)")
        print(f"[codebook] cluster_size nonzero={_cs_nonzero}/8192 (expect 8192)")
        if _cs_nonzero < 8000:
            print(f"[codebook] WARNING: {8192 - _cs_nonzero} codes have zero cluster_size "
                  "— EMA state may have been reset, first batches will be ill-conditioned")
    else:
        print("[codebook] WARNING: quantize.embedding.cluster_size not found in checkpoint")
    del _ckpt_state

    # Build a working directory that has all LaBraM source files so that
    # relative imports inside run_vqnsp_training.py resolve correctly.
    labram_src = "/tmp/labram_src"
    if os.path.exists(labram_src):
        shutil.rmtree(labram_src)
    shutil.copytree(labram_root, labram_src)

    patched_script = os.path.join(labram_src, "run_vqnsp_training_patched.py")
    _patch_run_vqnsp(
        original_path=os.path.join(labram_src, "run_vqnsp_training.py"),
        train_hdf5_paths=train_hdf5_paths,
        val_hdf5_paths=val_hdf5_paths,
        out_path=patched_script,
    )
    print(f"Patched script written to {patched_script}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_label = f"dry-run-{timestamp}" if dry_run else f"full-{timestamp}"
    output_dir = f"/project/data/labram_tokenizer/run-{run_label}/"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        patched_script,
        "--resume",          model_only_path,
        "--output_dir",      output_dir,
        "--input_size",      "1600",
        "--codebook_n_emd",  "8192",
        "--codebook_emd_dim","64",
        "--quantize_kmeans_init",
        "--batch_size",      "64",
        "--epochs",          str(epochs),
        "--warmup_epochs",   str(warmup_epochs),
        "--lr",              "5e-5",
        "--num_workers",     "4",
        "--save_ckpt_freq",  "1",
    ]

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=labram_src)

    # Find the last checkpoint written by LaBraM's save logic.
    # LaBraM may write checkpoint-{epoch}.pth and/or checkpoint.pth (latest).
    ckpt_files = sorted(
        [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".pth") and (
                f.startswith("checkpoint-") or f == "checkpoint.pth"
            )
        ],
        key=lambda p: os.path.getmtime(p),
    )
    print(f"Files in output_dir: {os.listdir(output_dir)}")

    if not ckpt_files:
        raise RuntimeError(
            f"No checkpoint files found in {output_dir} after training. "
            "Check the torchrun logs above for errors."
        )

    final_ckpt = ckpt_files[-1]
    print(f"Training complete. Final checkpoint: {final_ckpt}")

    data_volume.commit()
    return final_ckpt


@app.local_entrypoint()
def finetune_standalone():
    """Dry run: 1 epoch on EEG2 sub-01 only."""
    print(
        "Starting LaBraM dry-run finetune (1 epoch, sub-01).\n"
        "Watch the 'Unused_code' log line from engine_for_vqnsp.py.\n"
        "  Unused_code near 0   -> codebook in use, scale OK, proceed to full run.\n"
        "  Unused_code near 8192 -> scale mismatch, see CLAUDE.md MVNN section.\n"
    )
    final_ckpt = finetune_labram.remote(dry_run=True)
    print(f"Dry run complete. Checkpoint: {final_ckpt}")


@app.local_entrypoint()
def run_full():
    """Full finetune: 10 epochs on all 10 EEG2 subjects."""
    print("Starting LaBraM full finetune (10 epochs, all EEG2 subjects).")
    final_ckpt = finetune_labram.remote(dry_run=False, include_eeg1=False)
    print(f"Full run complete. Checkpoint: {final_ckpt}")


@app.local_entrypoint()
def run_full_combined():
    """Full finetune: 10 epochs on EEG2 (10 subjects) + EEG1 (50 subjects) combined.

    This is the shipped recipe — checkpoint-5 from this run is the production
    tokenizer (slug V8192_d64_ch17_sr200_train-eeg1+2_e5).
    """
    print(
        "Starting LaBraM combined finetune (10 epochs, EEG2 + EEG1).\n"
        "EEG1 is µV-scale (not MVNN-whitened), which may improve codebook coverage.\n"
        "Requires convert_eeg1_to_labram_hdf5 to have run first."
    )
    final_ckpt = finetune_labram.remote(dry_run=False, include_eeg1=True, warmup_epochs=3)
    print(f"Combined run complete. Checkpoint: {final_ckpt}")
