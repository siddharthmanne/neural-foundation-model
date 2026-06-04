"""
Train Stage 1 EEG tokenizer (BottleneckMLP + product VQ) on THINGS-EEG2.

Loads sub-01 through sub-10 from the project volume, applies image-level
80/20 split (same seed as the LaBraM HDF5 conversion so splits are consistent),
trains for 20 epochs, saves checkpoint and config.

Run:
    modal run neural_tokenizers/eeg/modal/train_stage1.py
"""

import os
import modal
from _app import app, data_volume

stage1_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch", "numpy"])
    .add_local_python_source("neural_tokenizers")
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)


@app.function(
    image=stage1_image,
    volumes={"/project": data_volume},
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=3600 * 4,
)
def train_stage1(
    subjects: list[str] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> str:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from neural_tokenizers.eeg.eeg_tokenizer import Stage1EEGTokenizer

    if subjects is None:
        # Start with sub-01 only; can extend to all subjects after validation.
        subjects = ["sub-01"]

    n_train_images = 16540
    n_test_images = 200
    total_images = n_train_images + n_test_images

    rng = np.random.default_rng(20200220)
    shuffled = rng.permutation(total_images)
    n_train_split = int(total_images * 0.8)
    train_image_ids = set(shuffled[:n_train_split].tolist())

    raw_root = "/project/data/raw/things-eeg2"
    all_train_X, all_val_X = [], []

    for subject in subjects:
        subject_dir = os.path.join(raw_root, subject)
        train_npy = _find_npy(subject_dir, "training")
        test_npy = _find_npy(subject_dir, "test")

        eeg_train = np.load(train_npy, allow_pickle=True).item()["preprocessed_eeg_data"]
        eeg_test = np.load(test_npy, allow_pickle=True).item()["preprocessed_eeg_data"]
        # shapes: (16540, 4, 17, 100), (200, 80, 17, 100)

        for img_idx in range(eeg_train.shape[0]):
            dest = all_train_X if img_idx in train_image_ids else all_val_X
            for rep in range(eeg_train.shape[1]):
                dest.append(eeg_train[img_idx, rep])

        for img_idx in range(eeg_test.shape[0]):
            global_id = n_train_images + img_idx
            dest = all_train_X if global_id in train_image_ids else all_val_X
            for rep in range(eeg_test.shape[1]):
                dest.append(eeg_test[img_idx, rep])

        print(f"[{subject}] loaded {eeg_train.shape[0]} train + {eeg_test.shape[0]} test images")

    train_X = torch.tensor(np.array(all_train_X, dtype=np.float32))
    val_X = torch.tensor(np.array(all_val_X, dtype=np.float32))
    print(f"Total: {train_X.shape[0]} train trials, {val_X.shape[0]} val trials")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Stage1EEGTokenizer(channels=17, time=100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(train_X), batch_size=batch_size, shuffle=True, num_workers=4
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            _, loss, _ = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg = total_loss / len(train_loader)

        # Quick val loss (no_grad, single pass)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for start in range(0, min(2048, len(val_X)), batch_size):
                xv = val_X[start : start + batch_size].to(device)
                _, lv, _ = model(xv)
                val_loss += lv.item()
        val_loss /= max(1, min(2048, len(val_X)) // batch_size)
        print(f"Epoch {epoch}/{epochs}  train_loss={avg:.4f}  val_loss={val_loss:.4f}")

    out_dir = "/project/data/stage1_tokenizer"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    model.save(ckpt_path)
    print(f"Saved Stage 1 checkpoint to {ckpt_path}")

    data_volume.commit()
    return ckpt_path


def _find_npy(subject_dir: str, pattern: str) -> str:
    for fname in os.listdir(subject_dir):
        if pattern in fname and fname.endswith(".npy"):
            return os.path.join(subject_dir, fname)
    raise FileNotFoundError(
        f"No file matching '{pattern}' in {subject_dir}. "
        f"Contents: {os.listdir(subject_dir)}"
    )


@app.local_entrypoint()
def train_standalone():
    ckpt = train_stage1.remote()
    print(f"Stage 1 training complete. Checkpoint: {ckpt}")
