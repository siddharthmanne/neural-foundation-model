"""
Run the 4-axis tokenizer evaluation harness on Modal against real THINGS-EEG2 val data.

Evaluates whichever checkpoints exist on the volume:
  - Stage 1 (BottleneckMLP + product VQ): /project/data/stage1_tokenizer/checkpoint.pt
  - LaBraM finetuned: latest checkpoint under /project/data/labram_tokenizer/

Prints a side-by-side comparison table across reconstruction, codebook,
sequence, and probe axes.

Run:
    modal run neural_tokenizers/eeg/modal/run_eval_harness.py
"""

import os
import shutil
import tarfile
import urllib.request

import modal
from _app import app, data_volume


def _download_labram(dest: str = "/tmp/LaBraM") -> str:
    """Download LaBraM source archive if not already present. Returns dest path."""
    if not os.path.isdir(dest):
        print("Downloading LaBraM source...")
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

# LaBraM finetune container has everything we need for both tokenizers.
eval_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel")
    .pip_install([
        "scipy",
        "numpy<2",
        "einops",
        "timm==0.6.12",
        "tensorboardX",
        "h5py",
        "mne",
        "pyhealth",
    ])
    .add_local_python_source("neural_tokenizers")
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)


@app.function(
    image=eval_image,
    volumes={"/project": data_volume},
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=3600 * 2,
)
def run_eval(n_val_samples: int = 1000, labram_ckpt_path: str | None = None) -> dict:
    """
    Evaluate both tokenizers on a held-out slice of THINGS-EEG2 val data.

    n_val_samples: number of val trials to use (keeps the run fast).
    labram_ckpt_path: explicit LaBraM checkpoint to evaluate. If None, uses
        the latest finetuned checkpoint found under /project/data/labram_tokenizer/.
        Pass the original pretrained path to compare against finetune quality.
    """
    import numpy as np
    import sys
    import torch
    from neural_tokenizers.evaluation import EvalConfig, evaluate

    labram_root = _download_labram()
    if labram_root not in sys.path:
        sys.path.insert(0, labram_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running eval on {device}")

    # Load val data (sub-01, val split)
    signal, labels = _load_val_data(
        subject="sub-01",
        n_samples=n_val_samples,
    )
    signal = signal.to(device)
    labels = labels.to(device)
    print(f"Val data: {signal.shape}, {labels.shape}")

    config = EvalConfig(
        sample_rate_hz=100.0,
        device=device,
        batch_size=128,
        probe_epochs=100,
        probe_top_k=(1, 5),
    )

    results = {}

    # --- Stage 1 ---
    stage1_ckpt = "/project/data/stage1_tokenizer/checkpoint.pt"
    if os.path.exists(stage1_ckpt):
        from neural_tokenizers.eeg.eeg_tokenizer import Stage1EEGTokenizer
        tok_s1 = Stage1EEGTokenizer.load(stage1_ckpt)
        tok_s1 = tok_s1.to(device)
        tok_s1.eval()
        print("\n=== Evaluating Stage 1 tokenizer ===")
        report_s1 = evaluate(tok_s1, signal, labels, config)
        print(report_s1)
        results["stage1"] = _report_to_dict(report_s1)
    else:
        print(f"Stage 1 checkpoint not found at {stage1_ckpt}, skipping.")

    # --- LaBraM ---
    labram_ckpt = labram_ckpt_path or _find_latest_labram_ckpt()
    if labram_ckpt:
        import json as _json
        from neural_tokenizers.eeg.labram_tokenizer import LaBraMTokenizer
        manifest_path = "/project/data/labram_input/things-eeg2/manifest.json"
        ch_names = None
        if os.path.exists(manifest_path):
            with open(manifest_path) as _f:
                _manifest = _json.load(_f)
            first_subject = next(iter(_manifest))
            ch_names = _manifest[first_subject].get("ch_names")
        tok_lb = LaBraMTokenizer(ckpt_path=labram_ckpt, ch_names=ch_names, device=device)
        print(f"\n=== Evaluating LaBraM tokenizer ({labram_ckpt}) ===")
        report_lb = evaluate(tok_lb, signal, labels, config)
        print(report_lb)
        results["labram"] = _report_to_dict(report_lb)
    else:
        print("No LaBraM checkpoint found, skipping.")

    # Print comparison
    if len(results) > 1:
        print("\n=== Comparison ===")
        _print_comparison(results)

    # Save results JSON to volume
    import json
    out_path = "/project/data/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    data_volume.commit()
    return results


def _load_val_data(subject: str, n_samples: int) -> tuple:
    import numpy as np
    import torch

    n_train_images = 16540
    n_test_images = 200
    total_images = n_train_images + n_test_images

    rng = np.random.default_rng(20200220)
    shuffled = rng.permutation(total_images)
    n_train_split = int(total_images * 0.8)
    train_image_ids = set(shuffled[:n_train_split].tolist())

    subject_dir = f"/project/data/raw/things-eeg2/{subject}"
    test_npy = _find_npy(subject_dir, "test")
    eeg_test = np.load(test_npy, allow_pickle=True).item()["preprocessed_eeg_data"]
    # (200, 80, 17, 100)

    # Use test images (200 unique stimuli, 200 classes) for the probe
    trials, labels_list = [], []
    for img_idx in range(eeg_test.shape[0]):
        global_id = n_train_images + img_idx
        if global_id not in train_image_ids:
            for rep in range(eeg_test.shape[1]):
                trials.append(eeg_test[img_idx, rep])
                labels_list.append(img_idx)

    trials_arr = np.array(trials, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)

    # Subsample if needed
    if len(trials_arr) > n_samples:
        idx = np.random.default_rng(42).choice(len(trials_arr), n_samples, replace=False)
        idx.sort()
        trials_arr = trials_arr[idx]
        labels_arr = labels_arr[idx]

    return torch.tensor(trials_arr), torch.tensor(labels_arr)


def _find_npy(subject_dir: str, pattern: str) -> str:
    for fname in os.listdir(subject_dir):
        if pattern in fname and fname.endswith(".npy"):
            return os.path.join(subject_dir, fname)
    raise FileNotFoundError(f"No '{pattern}' .npy in {subject_dir}")


def _find_latest_labram_ckpt() -> str | None:
    base = "/project/data/labram_tokenizer"
    if not os.path.isdir(base):
        return None
    candidates = []
    for run_dir in os.listdir(base):
        run_path = os.path.join(base, run_dir)
        if not os.path.isdir(run_path):
            continue
        for fname in os.listdir(run_path):
            if fname.startswith("checkpoint-") and fname.endswith(".pth"):
                candidates.append((os.path.getmtime(os.path.join(run_path, fname)),
                                   os.path.join(run_path, fname)))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def _report_to_dict(report) -> dict:
    out = {}
    for axis in report.axes():
        out[axis.name] = axis.values
    return out


def _print_comparison(results: dict) -> None:
    all_metrics = {}
    for name, d in results.items():
        for axis, vals in d.items():
            for k, v in vals.items():
                key = f"{axis}/{k}"
                all_metrics.setdefault(key, {})[name] = v

    models = list(results.keys())
    header = f"{'metric':<40s}" + "".join(f"  {m:<12s}" for m in models)
    print(header)
    print("-" * len(header))
    for metric, model_vals in sorted(all_metrics.items()):
        row = f"{metric:<40s}"
        for m in models:
            row += f"  {model_vals.get(m, float('nan')):>12.4f}"
        print(row)


@app.local_entrypoint()
def eval_standalone():
    results = run_eval.remote()
    print("Evaluation complete.")


@app.local_entrypoint()
def eval_original_pretrained():
    """Eval the original vqnsp.pth (no finetune) to isolate whether codebook
    collapse is caused by the train/inference window mismatch or by finetuning."""
    original_ckpt = "/project/data/labram_tokenizer/vqnsp_model_only.pth"
    print(f"Evaluating original pretrained checkpoint: {original_ckpt}")
    results = run_eval.remote(labram_ckpt_path=original_ckpt)
    print("Evaluation complete.")
