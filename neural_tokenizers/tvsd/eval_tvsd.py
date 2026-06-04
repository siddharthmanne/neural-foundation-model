"""Run the §5.3 linear probe on TVSD monkey MUA data.

Since no TVSD tokenizer exists yet, this script uses a RandomTokenizer stub
so the probe framework can still compute the `raw` (upper bound) and `random`
(lower bound) feature sets. The `tokens_*` metrics will match `random` by
construction — that is expected.

Usage (from repo root, fourm env):
    python -m neural_tokenizers.tvsd.eval_tvsd
    python -m neural_tokenizers.tvsd.eval_tvsd --monkey monkeyN --region V4

The script prints a MetricResult dict with keys like:
    bal_acc_raw_mean / _std      ← primary number of interest
    bal_acc_random_mean / _std   ← should be ~3.7% (1/27)
    top1_raw_mean / _std
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from neural_tokenizers.evaluation.probe import compute_probe_metrics
from neural_tokenizers.evaluation.protocol import EvalConfig
from neural_tokenizers.stubs.tokenizers import RandomTokenizer
from neural_tokenizers.tvsd.data import ASSEMBLY_ROOT, load_tvsd
from neural_tokenizers.tvsd.labels import STIMULUS_ROOT, apply_image_ids, build_tvsd_label_map


def run(
    monkey: str = "monkeyF",
    region: str = "IT",
    timebin_ms: int | str = 10,
    assembly_root: str = ASSEMBLY_ROOT,
    stimulus_root: str = STIMULUS_ROOT,
    n_folds: int = 5,
    seed: int = 0,
    max_trials: int | None = None,
) -> dict:
    print(f"Loading TVSD signal: monkey={monkey} region={region} timebin_ms={timebin_ms}")
    signal, image_ids = load_tvsd(
        root=assembly_root, monkey=monkey, region=region, timebin_ms=timebin_ms
    )
    print(f"  signal shape: {tuple(signal.shape)}")

    print("Resolving superordinate labels …")
    label_map = build_tvsd_label_map(monkey=monkey, stimulus_root=stimulus_root)
    labels_np = apply_image_ids(label_map, image_ids)

    valid = labels_np >= 0
    n_valid = int(valid.sum())
    print(f"  valid trials (non-catch, mapped concept): {n_valid} / {len(labels_np)}")

    signal = signal[valid]
    labels = torch.from_numpy(labels_np[valid])

    if max_trials is not None and max_trials < len(labels):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(labels), size=max_trials, replace=False)
        idx.sort()
        signal = signal[idx]
        labels = labels[idx]
        print(f"  subsampled to {max_trials} trials for smoke test")

    B, C, T = signal.shape
    stub = RandomTokenizer(codebook_size=256, seq_len=16, signal_shape=(C, T), seed=seed)

    config = EvalConfig(
        sample_rate_hz=100.0,  # 10ms bins = 100 Hz effective rate
        probe_n_folds=n_folds,
        probe_class_weighted=True,
        probe_top_k=(1, 5),
        seed=seed,
    )

    print(f"Running probe: {n_folds}-fold CV, {B} trials, {27} classes …")
    result = compute_probe_metrics(stub, signal, labels, config)
    return result.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monkey", default="monkeyF", choices=["monkeyF", "monkeyN"])
    ap.add_argument("--region", default="IT", choices=["IT", "V1", "V4", "Full"])
    ap.add_argument("--timebin-ms", default=10, type=lambda x: x if x == "default" else int(x))
    ap.add_argument("--n-folds", default=5, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--max-trials", default=None, type=int,
                    help="Subsample to N trials (smoke test). Omit for full run.")
    args = ap.parse_args()

    metrics = run(
        monkey=args.monkey,
        region=args.region,
        timebin_ms=args.timebin_ms,
        n_folds=args.n_folds,
        seed=args.seed,
        max_trials=args.max_trials,
    )

    print("\n--- Results ---")
    print(json.dumps(metrics, indent=2))

    chance = 1 / 27
    raw = metrics.get("bal_acc_raw_mean", float("nan"))
    rand = metrics.get("bal_acc_random_mean", float("nan"))
    print(f"\nbal_acc  raw={raw:.4f}  random={rand:.4f}  chance={chance:.4f}")


if __name__ == "__main__":
    main()
