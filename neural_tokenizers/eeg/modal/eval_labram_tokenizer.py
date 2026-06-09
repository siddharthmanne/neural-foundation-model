"""
Run the four-axis neural_tokenizers eval harness on a finetuned LaBraM checkpoint.

Loads THINGS-EEG2 test data (preprocessed_eeg_test.npy, 200 images × 80 reps),
applies the same per-subject µV calibration used during training, and reports:
  - Reconstruction fidelity (MSE + spectral MSE per band)
  - Codebook utilization (perplexity + dead-code fraction)
  - Linear probe (top-1 / top-5 on 200 test-image labels)
  - Token-sequence statistics (unigram/bigram entropy, run length)

Run:
    modal run neural_tokenizers/eeg/modal/eval_labram_tokenizer.py
"""

import json
import os
import shutil
import sys
import tarfile
import urllib.request

import modal
import numpy as np

from _app import app, data_volume


SLUG = "V8192_d64_ch17_sr200_train-eeg1+2_e5"
CHECKPOINT_PATH = f"/project/checkpoints/eeg/labram/{SLUG}/checkpoint.pt"
EEG2_RAW_ROOT = "/project/data/raw/things-eeg2"
EEG1_STATS_PATH = "/project/data/things-eeg/labels/eeg1_per_channel_std.json"

# Resolve paths relative to this file (on the client at image-build time).
_this_dir = os.path.dirname(os.path.abspath(__file__))            # .../eeg/modal/
_eeg_dir  = os.path.dirname(_this_dir)                            # .../eeg/
_nt_dir   = os.path.dirname(_eeg_dir)                             # .../neural_tokenizers/

eval_image = (
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
        os.path.join(_this_dir, "_app.py"),
        "/root/_app.py",
    )
    .add_local_dir(
        local_path=_nt_dir,
        remote_path="/neural_tokenizers",
    )
)


def _download_labram(dest: str = "/tmp/LaBraM") -> str:
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


def _load_test_data(n_subjects: int = 10, avg_per_image: bool = False):
    """Load and calibrate THINGS-EEG2 test trials from n_subjects.

    Args:
        n_subjects: number of EEG2 subjects to include
        avg_per_image: if True, average all 80 reps per (subject, image) into
            one cleaner trial. Yields 200 classes × n_subjects trials, much
            higher SNR for diagnosing whether the §5 probe failure is a
            signal-noise issue vs a tokenizer-throws-away-signal issue.

    Returns:
        signal: (N, 17, 100) float32 tensor — calibrated µV-scale EEG at 100 Hz
        labels: (N,) long tensor           — image index in [0, 200)
    """
    import torch

    with open(EEG1_STATS_PATH) as f:
        eeg1_stats = json.load(f)
    eeg1_ch_std = np.array(eeg1_stats["per_channel_std"], dtype=np.float32)  # (17,)

    all_signals = []
    all_labels = []

    for sub_idx in range(1, n_subjects + 1):
        sub = f"sub-{sub_idx:02d}"
        test_path  = os.path.join(EEG2_RAW_ROOT, sub, "preprocessed_eeg_test.npy")
        train_path = os.path.join(EEG2_RAW_ROOT, sub, "preprocessed_eeg_training.npy")
        if not os.path.exists(test_path):
            print(f"[{sub}] test file not found, skipping")
            continue

        test_data  = np.load(test_path,  allow_pickle=True).item()["preprocessed_eeg_data"]
        train_data = np.load(train_path, allow_pickle=True).item()["preprocessed_eeg_data"]

        # Pool all trials to compute per-channel std (same formula as training conversion).
        train_flat = train_data.reshape(-1, 17, 100)                   # (66160, 17, 100)
        all_flat_for_std = np.concatenate(
            [train_flat, test_data.reshape(-1, 17, 100)], axis=0
        )
        eeg2_ch_std = all_flat_for_std.std(axis=(0, 2)).clip(min=1e-6)  # (17,)
        scale = eeg1_ch_std / eeg2_ch_std                              # (17,)

        # test_data shape: (200, 80, 17, 100). Decide here whether to average reps.
        if avg_per_image:
            # Average across 80 reps per image, then calibrate → (200, 17, 100)
            avg = test_data.mean(axis=1)                               # (200, 17, 100)
            calibrated = (avg * scale[None, :, None]).astype(np.float32)
            labels = np.arange(200)                                    # (200,)
        else:
            test_flat  = test_data.reshape(-1, 17, 100)                # (16000, 17, 100)
            calibrated = (test_flat * scale[None, :, None]).astype(np.float32)
            labels = np.repeat(np.arange(200), 80)                     # (16000,)

        all_signals.append(calibrated)
        all_labels.append(labels)
        print(f"[{sub}] {len(calibrated)} test trials, scale mean={scale.mean():.2f}"
              + (" (averaged 80 reps per image)" if avg_per_image else ""))

    signal = torch.from_numpy(np.concatenate(all_signals, axis=0))
    labels = torch.from_numpy(np.concatenate(all_labels, axis=0)).long()
    print(f"Total: {len(signal)} trials, {signal.shape}")
    return signal, labels


class _FilteredTokenizer:
    """Wraps LaBraMTokenizer to tokenize via the F2-CORRECTED production path.

    Replicates convert_eeg_to_labram_hdf5 training preprocessing (and the F2
    fix to modal_eeg_produce_tokens): upsample 100->200 Hz, then 0.1-75 Hz FIR
    bandpass + 50 Hz notch at 200 Hz, then encode with the base model. The
    stock wrapper upsamples but does NOT filter, so comparing the two answers
    "does removing the train/inference filter skew change the tokens?".

    decode/codebook delegate to the base tokenizer unchanged.
    """

    def __init__(self, base):
        self.base = base
        self.codebook_size = base.codebook_size

    def tokenize(self, x):
        import warnings
        import mne
        import scipy.signal
        import torch

        arr = x.detach().cpu().numpy()
        up = scipy.signal.resample_poly(arr, up=2, down=1, axis=-1)  # (B, 17, 200)
        B, C, T = up.shape
        flat = up.reshape(B * C, T).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat = mne.filter.filter_data(
                flat, sfreq=200.0, l_freq=0.1, h_freq=75.0,
                method="fir", fir_window="hamming", verbose=False,
            )
            flat = mne.filter.notch_filter(flat, Fs=200.0, freqs=50.0, verbose=False)
        x200 = torch.tensor(flat.reshape(B, C, T), dtype=torch.float32,
                            device=self.base.device)
        # Same 8-patch tiling the wrapper uses (input_size=1600 native length).
        x_nats = x200.unsqueeze(2).expand(-1, -1, 8, -1).contiguous()
        with torch.no_grad():
            tokens_all = self.base.model.get_codebook_indices(
                x_nats, input_chans=self.base.input_chans
            )
            tokens = tokens_all[:, ::8]
        return tokens.long().cpu()

    def decode_tokens(self, tokens):
        return self.base.decode_tokens(tokens)


@app.function(
    image=eval_image,
    volumes={"/project": data_volume},
    timeout=7200,
    cpu=4,
    memory=32768,
)
def eval_labram(
    checkpoint_path: str = CHECKPOINT_PATH,
    n_eval_subset: int = 20_000,
    avg_per_image: bool = False,
    apply_filter: bool = False,
):
    import torch  # container-only import

    labram_root = _download_labram()
    os.environ["LABRAM_ROOT"] = labram_root
    if labram_root not in sys.path:
        sys.path.insert(0, labram_root)

    # neural_tokenizers is mounted at /neural_tokenizers; make it importable.
    mount_root = "/neural_tokenizers"
    parent = os.path.dirname(mount_root)  # "/"
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Deferred imports (need LaBraM on path first).
    from neural_tokenizers.eeg.labram_tokenizer import LaBraMTokenizer
    from neural_tokenizers.evaluation import EvalConfig, evaluate

    print(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = LaBraMTokenizer(ckpt_path=checkpoint_path, device="cpu")
    if apply_filter:
        tokenizer = _FilteredTokenizer(tokenizer)
        print("Tokenizer loaded (F2-CORRECTED path: 0.1-75 bandpass + 50 notch at 200 Hz).")
    else:
        print("Tokenizer loaded (stock path: no filter, matches shipped tokens).")

    signal, labels = _load_test_data(n_subjects=10, avg_per_image=avg_per_image)
    print(f"Loaded: {signal.shape}, labels: {labels.shape}, classes: {labels.unique().numel()}")

    # Subsample to keep CPU eval tractable.
    if len(signal) > n_eval_subset:
        rng = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(signal), generator=rng)[:n_eval_subset]
        signal = signal[idx]
        labels = labels[idx]
        print(f"Subsampled to {len(signal)} trials for CPU tractability")

    config = EvalConfig(
        sample_rate_hz=100.0,
        device="cpu",
        batch_size=256,
        seed=0,
        probe_epochs=100,
        probe_lr=1e-2,
        probe_weight_decay=1e-4,
        probe_top_k=(1, 5),
        probe_test_frac=0.2,
        psd_nperseg=50,
    )

    print("\nRunning eval harness (all four axes)...")
    report = evaluate(tokenizer, signal, labels, config)

    print("\n" + "=" * 60)
    print(str(report))
    print("=" * 60)

    return {axis.name: axis.values for axis in report.axes()}


@app.local_entrypoint()
def main(
    checkpoint_path: str = CHECKPOINT_PATH,
    n_eval_subset: int = 20_000,
    avg_per_image: bool = False,
    apply_filter: bool = False,
):
    result = eval_labram.remote(
        checkpoint_path=checkpoint_path,
        n_eval_subset=n_eval_subset,
        avg_per_image=avg_per_image,
        apply_filter=apply_filter,
    )
    print("\nFinal metrics:")
    for axis, values in result.items():
        print(f"\n[{axis}]")
        for k, v in values.items():
            print(f"  {k:<40s} {v:.4f}")
