"""Tokenizer evaluation for pre-computed DINOv2 tokens on THINGS images.

Runs the four-axis harness (codebook, probe, sequence, retrieval) on 4M's
DINOv2-B/14 VQVAE tokens.  No GPU, no model downloads, no raw image features
needed — the eval works entirely from pre-computed token IDs stored in the
THINGS WDS shards.

Probe and retrieval use bag-of-codes features: a normalised histogram over the
8192-code vocabulary per sample.  The `raw` bracket uses a 1-dim dummy signal
(no real features available) and will land at chance — this is expected and
correct.  Compare `tokens_all` vs `random` for the main signal-vs-noise verdict.

Label chain:
    shard_key → image_id (strip leading zeros)
              → concept_id  (image_id_to_concept.json)
              → superordinate_index 0-26  (concept_id_to_superordinate.json)

Usage::

    python eval_dinov2.py \\
        --tokens_dir /project/data/val/things/tok_dinov2@224

Run --help for all options.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# ── Import axis modules by file path so this script works both as a package
# member (relative imports) and as a standalone file (sys.path fallback).
# We deliberately bypass __init__.py / evaluate() because that function
# calls tokenize_all() which requires a real tokenizer.tokenize() — we have
# only pre-computed token IDs and call each axis function directly instead.

_HERE = Path(__file__).resolve().parent

try:
    from .protocol import EvalConfig, MetricResult, TokenizerReport
    from .codebook import compute_codebook_metrics_from_tokens
    from .sequence import compute_sequence_metrics
    from .probe import compute_probe_metrics
    from .retrieval import compute_retrieval_metrics
except ImportError:
    sys.path.insert(0, str(_HERE))
    from protocol import EvalConfig, MetricResult, TokenizerReport          # type: ignore[no-redef]
    from codebook import compute_codebook_metrics_from_tokens               # type: ignore[no-redef]
    from sequence import compute_sequence_metrics                           # type: ignore[no-redef]
    from probe import compute_probe_metrics                                 # type: ignore[no-redef]
    from retrieval import compute_retrieval_metrics                         # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal adapter — satisfies Tokenizer protocol from pre-computed token IDs
# ─────────────────────────────────────────────────────────────────────────────

class _PrecomputedTokenAdapter:
    """Token-ID-only adapter — no model, no GPU.

    tokenize() is intentionally not implemented.  The eval script passes
    tokens directly to each axis function.  Without tokens_to_embedding the
    probe falls back to bag-of-codes features (histogram over the 8192-vocab).
    """

    def __init__(self, codebook_size: int) -> None:
        self.codebook_size = codebook_size

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "_PrecomputedTokenAdapter: pass tokens= directly, do not call tokenize()."
        )

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "_PrecomputedTokenAdapter: no decoder — reconstruction skipped."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _read_shard(tar_path: Path) -> dict[str, np.ndarray]:
    """Return {key: array} for every .npy entry in a single-modality shard."""
    samples: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".npy"):
                continue
            key = member.name.split(".")[0]
            buf = tf.extractfile(member).read()
            samples[key] = np.load(io.BytesIO(buf))
    return samples


def load_tokens_from_shards(
    tokens_dir: Path,
    shard_indices: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Load pre-computed DINOv2 tokens from WDS shards.

    On-disk format: {key}.npy with shape (1, 256) int16 per sample.
    The leading 1 is the crop axis (single center crop, no augmentation).

    Returns {key: (1, 256) int16}.  If shard_indices is None, all
    shard_NNN.tar files found in tokens_dir are loaded.
    """
    tokens_dir = Path(tokens_dir)
    if shard_indices is None:
        tar_paths = sorted(tokens_dir.glob("shard_*.tar"))
    else:
        tar_paths = [tokens_dir / f"shard_{i:03d}.tar" for i in shard_indices]

    if not tar_paths:
        raise FileNotFoundError(f"No shard tars found in {tokens_dir}")

    all_samples: dict[str, np.ndarray] = {}
    for p in tar_paths:
        if not p.exists():
            raise FileNotFoundError(f"Shard not found: {p}")
        all_samples.update(_read_shard(p))
        print(f"  loaded {p.name}  ({len(all_samples)} total keys)", flush=True)

    return all_samples


def load_things_labels(
    image_id_to_concept_path: Path,
    concept_to_superordinate_path: Path,
    keys: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Map THINGS shard keys → 27-way superordinate category labels.

    Chain:
        shard_key → image_id (strip leading zeros, e.g. "000042" → "42")
                  → concept_id   (image_id_to_concept.json)
                  → superordinate_index 0-26  (concept_id_to_superordinate.json)

    Returns:
        labels:          (N,) int64 tensor, values in [0, 26]
        category_names:  list of 27 strings
    """
    with open(image_id_to_concept_path) as f:
        img_map: dict[str, int] = json.load(f)["image_id_to_concept_id"]
    with open(concept_to_superordinate_path) as f:
        cat_data = json.load(f)
    cat_map: dict[str, int] = cat_data["concept_id_to_superordinate_index"]
    category_names: list[str] = cat_data["category_names"]

    label_list: list[int] = []
    valid_keys: list[str] = []
    for key in keys:
        image_id = str(int(key))   # strip leading zeros: "000042" → "42"
        concept_id = img_map.get(image_id)
        if concept_id is None:
            continue   # image not in MEG trial set
        superordinate = cat_map.get(str(concept_id))
        if superordinate is None:
            continue   # concept has no superordinate label (892/1854 concepts covered)
        label_list.append(superordinate)
        valid_keys.append(key)

    n_skipped = len(keys) - len(valid_keys)
    if n_skipped:
        print(f"  Skipped {n_skipped}/{len(keys)} samples: not in MEG trials or "
              f"concept lacks superordinate label (27-way labels cover ~892/1854 concepts)",
              flush=True)
    if not valid_keys:
        raise RuntimeError("No keys resolved to labels — check JSON alignment.")

    return torch.tensor(label_list, dtype=torch.long), category_names, valid_keys


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    tokens_dir: Path | Sequence[Path],
    image_id_to_concept_path: Path,
    concept_to_superordinate_path: Path,
    shard_indices: Sequence[int] | None = None,
    n_samples: int | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    probe_classifier: str = "linear",
    probe_n_folds: int = 5,
    probe_epochs: int = 100,
    seed: int = 0,
    run_probe: bool = True,
    run_retrieval: bool = True,
) -> TokenizerReport:
    """Run the DINOv2 token evaluation.

    Each axis function is called directly with pre-computed tokens — this
    bypasses tokenize_all() and works without a live tokenizer model.

    tokens_dir may be a single Path or a list of Paths (e.g. train + val).
    Keys must be globally unique across all directories.
    """
    adapter = _PrecomputedTokenAdapter(codebook_size=8192)

    # ── Load tokens ───────────────────────────────────────────────────────
    dirs = [tokens_dir] if isinstance(tokens_dir, Path) else list(tokens_dir)
    token_dict: dict[str, np.ndarray] = {}
    for d in dirs:
        print(f"Loading tokens from {d} …", flush=True)
        token_dict.update(load_tokens_from_shards(d, shard_indices))
    all_keys = sorted(token_dict.keys())
    print(f"  {len(all_keys)} total samples", flush=True)

    if n_samples is not None and n_samples < len(all_keys):
        rng = random.Random(seed)
        all_keys = rng.sample(all_keys, n_samples)
        print(f"  subsampled to {len(all_keys)} (seed={seed})", flush=True)

    # ── Load labels ───────────────────────────────────────────────────────
    print("Loading THINGS labels …", flush=True)
    labels, category_names, all_keys = load_things_labels(
        image_id_to_concept_path, concept_to_superordinate_path, all_keys
    )
    n_classes = len(category_names)
    chance = 1.0 / n_classes
    print(f"  {len(all_keys)} labeled samples, {n_classes} categories — chance = {chance:.4f}", flush=True)

    # ── Build token tensor (N, 256) and dummy signal (N, 1, 1) ───────────
    token_arrays = [token_dict[k].reshape(256).astype(np.int64) for k in all_keys]
    tokens = torch.tensor(np.stack(token_arrays), dtype=torch.long)   # (N, 256)
    # probe/retrieval require a (B, C, T) signal; 1-dim dummy → `raw` at chance
    signal = torch.zeros(len(all_keys), 1, 1, dtype=torch.float32)
    print(f"  tokens {tuple(tokens.shape)}", flush=True)

    # ── Config ────────────────────────────────────────────────────────────
    config = EvalConfig(
        sample_rate_hz=1.0,        # not meaningful for vision; required field
        device=device,
        batch_size=batch_size,
        seed=seed,
        run_reconstruction=False,
        run_codebook=True,
        run_sequence=True,
        run_probe=run_probe,
        run_retrieval=run_retrieval,
        probe_classifier=probe_classifier,
        probe_n_folds=probe_n_folds,
        probe_epochs=probe_epochs,
        probe_rvq_layers=(None,),  # single-codebook VQVAE, not RVQ
    )

    # ── Run each axis with pre-computed tokens ────────────────────────────
    # We do NOT call evaluate() from __init__.py because that function always
    # calls tokenize_all() before passing tokens to the axis functions.
    print("\nRunning codebook axis …", flush=True)
    report = TokenizerReport()
    report.codebook = compute_codebook_metrics_from_tokens(tokens, adapter.codebook_size)

    print("Running sequence axis …", flush=True)
    report.sequence = compute_sequence_metrics(tokens, adapter.codebook_size)

    if run_probe:
        print("Running probe axis …", flush=True)
        report.probe = compute_probe_metrics(
            adapter, signal, labels, config, tokens=tokens
        )

    if run_retrieval:
        print("Running retrieval axis …", flush=True)
        report.retrieval = compute_retrieval_metrics(
            adapter, signal, labels, config, tokens=tokens
        )

    print("\n" + str(report))
    print(
        f"\nChance (27-way): {chance:.4f} — compare `bal_acc_tokens_all_mean`.\n"
        "`raw` bracket uses 1-dim dummy signal and is expected to be at chance."
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_REPO = _HERE.parents[1]   # neural-foundation-model/


def _default(name: str) -> Path:
    print(_REPO)
    return _REPO / "neural_tokenizers" / "meg" / "data" / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tokens_dir", type=Path, nargs="+",
        default=[Path("/home/users/liubr/projects/neural-image-foundation/tmp/train/tok_dinov2@224"), Path("/home/users/liubr/projects/neural-image-foundation/tmp/val/tok_dinov2@224")],
        help="One or more directories containing shard_NNN.tar files (e.g. train + val)",
    )
    parser.add_argument(
        "--image_id_to_concept", type=Path,
        default=_default("image_id_to_concept.json"),
    )
    parser.add_argument(
        "--concept_to_superordinate", type=Path,
        default=_default("concept_id_to_superordinate.json"),
    )
    parser.add_argument(
        "--shard_indices", type=int, nargs="+", default=None, metavar="N",
        help="Shard indices to load (default: all found)",
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--probe_classifier", default="linear",
        choices=["linear", "mlp"],
        help="'cnn' not supported without tokens_to_embedding",
    )
    parser.add_argument("--probe_n_folds", type=int, default=5)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_probe", action="store_true")
    parser.add_argument("--no_retrieval", action="store_true")
    args = parser.parse_args()

    run_eval(
        tokens_dir=args.tokens_dir if len(args.tokens_dir) > 1 else args.tokens_dir[0],
        image_id_to_concept_path=args.image_id_to_concept,
        concept_to_superordinate_path=args.concept_to_superordinate,
        shard_indices=args.shard_indices,
        n_samples=args.n_samples,
        device=args.device,
        batch_size=args.batch_size,
        probe_classifier=args.probe_classifier,
        probe_n_folds=args.probe_n_folds,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
        run_probe=not args.no_probe,
        run_retrieval=not args.no_retrieval,
    )


if __name__ == "__main__":
    main()
