"""Compare BrainOmni eval reports against the μ-transform baseline."""

from __future__ import annotations

import json
from pathlib import Path


def load_report(path: str | Path) -> dict[str, dict[str, float]]:
    return json.loads(Path(path).read_text())


def compare(baseline: dict, candidate: dict) -> dict[str, dict[str, float]]:
    """Return per-axis deltas (candidate - baseline) for shared metrics."""
    deltas: dict[str, dict[str, float]] = {}
    for axis in baseline:
        if axis not in candidate:
            continue
        deltas[axis] = {
            k: candidate[axis][k] - baseline[axis].get(k, 0.0)
            for k in candidate[axis]
            if isinstance(candidate[axis][k], (int, float))
        }
    return deltas


def main():
    repo = Path(__file__).resolve().parents[2]  # neural_tokenizers/
    mu_path = repo / "meg/mu_transform/runs/V256_mu255_clip0.5-99.5_per_channel_s0/evals/eval_ntest=n3000_s0.json"
    bo_path = repo / "meg/brainomni/runs/V512_rvq4_win512_sf256_3a/evals/eval_ntest=n3000_s0.json"
    if not mu_path.exists() or not bo_path.exists():
        print("Missing eval JSON — run harness first.")
        return
    mu = load_report(mu_path)
    bo = load_report(bo_path)
    deltas = compare(mu, bo)
    print("BrainOmni vs μ-transform (delta = BrainOmni - μ):")
    for axis, metrics in deltas.items():
        print(f"  [{axis}]")
        for k, v in metrics.items():
            print(f"    {k:<32s} {v:+.4f}")


if __name__ == "__main__":
    main()
