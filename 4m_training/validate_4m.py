"""Run named validation tasks on a 4M model — masked prediction and cross-modal.

4M's built-in per-epoch eval can only validate the *same* task you trained
(its val loaders derive masking from the train dataset of the same name), so it
can't express several distinct validation tasks. This standalone runner builds a
val loader per task (each with its own in/out domains) and runs 4M's eval forward,
reporting per-task loss. Pick tasks with ``--select`` or run them all.

    python 4m_training/validate_4m.py --config configs/4m_things_main.yaml \
        --tasks configs/4m_things_val_tasks.yaml          # checkpoint from the main YAML
    python 4m_training/validate_4m.py ... --checkpoint /path/to.pth   # one-off override
    python 4m_training/validate_4m.py ... --select rgb2depth,anyany_neural

Tasks are defined in the --tasks YAML; see configs/4m_things_val_tasks.yaml. The
checkpoint comes from the main YAML's ``val_checkpoint:`` field unless ``--checkpoint``
is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE / "lib"))  # library modules live in lib/

import fourm_neural_modalities  # noqa: F401,E402 — register modalities/transforms
from fourm_dataloader import _wds_eval_loader, patch_pretrain_utils
from neural_constants import THINGS_IMAGE_SIZE
from repo_paths import TEXT_TOKENIZER
from token_accounting import format_tokens_seen, read_tokens_seen

patch_pretrain_utils()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _resolve_checkpoint(cli_checkpoint: Path | None, main_cfg: dict) -> Path | None:
    """Pick the checkpoint to validate: CLI ``--checkpoint`` overrides the
    ``val_checkpoint:`` field in the main YAML; absent both, return ``None``
    (random-weight pipeline smoke). Keeps the path varying knob in config so a
    plain ``--validate`` run needs no flags."""
    if cli_checkpoint is not None:
        return cli_checkpoint
    yaml_checkpoint = main_cfg.get("val_checkpoint")
    return Path(yaml_checkpoint) if yaml_checkpoint else None


def _task_dataset_config(task: dict, defaults: dict) -> dict:
    """A task entry -> the dataset_config the WDS val loader expects."""
    mods = ",".join(task["modalities"])
    data_path = f"{defaults['val_root']}/[{mods}]/{defaults['shards']}"
    in_n = len(task["in_domains"].split("-"))
    out_n = len(task["out_domains"].split("-"))
    return {
        "type": "multimodal",
        "use_wds": True,
        "data_path": data_path,
        "in_domains": task["in_domains"],
        "out_domains": task["out_domains"],
        "main_augment_domain": defaults.get("main_augment_domain", "tok_rgb"),
        "tok_train_aug": False,
        "aligned_captions": False,
        "input_alphas": "-".join(["1.0"] * in_n),
        "target_alphas": "-".join(["1.0"] * out_n),
    }


def build_model(in_union: list[str], out_union: list[str], model_name: str, input_size: int):
    """Build a 4M model with encoder embeddings for all inputs, decoders for all targets."""
    from fourm.utils import create_model
    from train_4m import _build_modality_info

    modality_info = _build_modality_info(sorted(set(in_union) | set(out_union)), input_size)

    def _emb(mod: str, key: str):
        info = modality_info[mod]
        if info["type"] == "img":
            return info[key](patch_size=info.get("patch_size", 16), image_size=input_size)
        return info[key]()

    enc = {m: _emb(m, "encoder_embedding") for m in in_union}
    dec = {m: _emb(m, "decoder_embedding") for m in out_union}
    return create_model(
        model_name, encoder_embeddings=enc, decoder_embeddings=dec,
        modality_info=modality_info, num_register_tokens=0,
    )


def load_checkpoint(model, ckpt_path: Path) -> dict:
    """Load weights only; tolerate architecture supersets (strict=False). Returns the
    full checkpoint dict so callers can read non-weight metadata (e.g. tokens-seen)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [ckpt] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return ckpt


def _build_task_loader(task_cfg: dict, input_size: int, in_range, out_range,
                       text_tokenizer, num_workers: int, batch_size: int):
    from fourm.data.modality_info import MODALITY_TRANSFORMS
    from fourm.data.pretrain_utils import setup_sampling_mod_info
    from things_augmenter import ThingsImageAugmenter
    from train_4m import _build_modality_info

    in_d = sorted(task_cfg["in_domains"].split("-"))
    out_d = sorted(task_cfg["out_domains"].split("-"))
    all_d = sorted(set(in_d) | set(out_d))
    # Restrict masking modality_info to THIS task's domains (+ its alphas).
    full = _build_modality_info(all_d, input_size)
    mod_info, sampling_weights = setup_sampling_mod_info(task_cfg, full)
    augmenter = ThingsImageAugmenter(
        target_size=input_size, no_aug=True, main_domain=task_cfg["main_augment_domain"]
    )
    loader = _wds_eval_loader(
        data_path=task_cfg["data_path"], all_domains=all_d, modality_info=mod_info,
        modality_transforms=MODALITY_TRANSFORMS, image_augmenter=augmenter,
        text_tokenizer=text_tokenizer, input_tokens_range=in_range,
        target_tokens_range=out_range, num_workers=num_workers, batch_size=batch_size,
        sampling_weights=sampling_weights,
    )
    return loader, all_d, out_d


@torch.no_grad()
def evaluate_task(model, loader, device, all_domains, out_domains, n_in, n_out, loss_type, dtype, n_batches):
    """Average per-target loss over up to n_batches val batches.

    Loss is reported ONLY on the task's ``out_domains`` (its prediction targets).
    The model may carry extra decoder heads — e.g. ``tok_rgb`` when only depth is
    this task's target — whose loss is 0 and would otherwise dilute the average
    (the bug: rgb2depth showing (depth + 0)/2). The task ``loss`` is the mean over
    ``out_domains`` alone, so a single-target task reports exactly that target.
    """
    model.eval()
    out_set = set(out_domains)
    totals: dict[str, float] = {}
    count = 0
    for i, x in enumerate(loader):
        if n_batches and i >= n_batches:
            break
        mod_dict = {
            mod: {k: v.to(device) for k, v in d.items()}
            for mod, d in x.items()
            if mod in all_domains
        }
        autocast = torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)
        with autocast:
            _, mod_loss = model(mod_dict, num_encoder_tokens=n_in,
                                num_decoder_tokens=n_out, loss_type=loss_type)
        target_losses = {mod: float(l.mean()) for mod, l in mod_loss.items() if mod in out_set}
        for mod, value in target_losses.items():
            totals[f"{mod}_loss"] = totals.get(f"{mod}_loss", 0.0) + value
        if target_losses:
            totals["loss"] = totals.get("loss", 0.0) + sum(target_losses.values()) / len(target_losses)
        count += 1
    if count == 0:
        return {"loss": float("nan"), "n_batches": 0}
    stats = {k: v / count for k, v in totals.items()}
    stats["n_batches"] = count
    return stats


class ValidationSuite:
    """Run the named validation tasks on a model → ``{task_name: stats}``.

    The single home for "run the task suite on a model". It is reusable across
    calls: deterministic WDS loaders are built lazily and cached, and the set of
    runnable tasks is resolved once against the model's actual domains. Two
    callers share it, differing only in *where the model comes from* and *how the
    result is formatted*:

      * ``run_validation`` (CLI) — builds a model + loads a checkpoint, runs once.
      * ``in_loop_val`` (training) — runs on the live model every eval epoch.
    """

    def __init__(self, tasks_cfg: dict, *, input_size: int, text_tokenizer,
                 loss_type: str, batch_size: int, n_batches: int,
                 select: list[str] | None = None):
        self.tasks_cfg = tasks_cfg
        self.tasks = tasks_cfg["tasks"]
        names = select or list(self.tasks)
        unknown = [n for n in names if n not in self.tasks]
        if unknown:
            raise SystemExit(f"unknown task(s) {unknown}; available: {list(self.tasks)}")
        self.names = names
        self.input_size = input_size
        self.text_tokenizer = text_tokenizer
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_in = tasks_cfg.get("fixed_eval_input_tokens", 128)
        self.n_out = tasks_cfg.get("fixed_eval_target_tokens", 128)
        self._cache: dict[str, tuple] = {}
        self._runnable: list[str] | None = None

    def _loader_for(self, name: str):
        if name not in self._cache:
            task_cfg = _task_dataset_config(self.tasks[name], self.tasks_cfg)
            self._cache[name] = _build_task_loader(
                task_cfg, self.input_size, (self.n_in, self.n_in), (self.n_out, self.n_out),
                self.text_tokenizer, num_workers=0, batch_size=self.batch_size,
            )
        return self._cache[name]

    def _resolve_runnable(self, model) -> list[str]:
        """Keep only tasks the model can run: encoder needs every in_domain, decoder
        every out_domain. Reads the (possibly DDP-wrapped) model's embedding keys."""
        m = getattr(model, "module", model)
        enc = set(getattr(m, "encoder_embeddings", {}).keys())
        dec = set(getattr(m, "decoder_embeddings", {}).keys())
        runnable = []
        for name in self.names:
            task = self.tasks[name]
            missing = (set(task["in_domains"].split("-")) - enc) | (
                set(task["out_domains"].split("-")) - dec)
            if missing:
                print(f"[val suite] skipping '{name}': model lacks {sorted(missing)}")
            else:
                runnable.append(name)
        return runnable

    def run(self, model, device, dtype) -> dict[str, dict[str, Any]]:
        if self._runnable is None:
            self._runnable = self._resolve_runnable(model)
        results: dict[str, dict[str, Any]] = {}
        for name in self._runnable:
            loader, all_d, out_d = self._loader_for(name)
            results[name] = evaluate_task(
                model, loader, device, all_d, out_d, self.n_in, self.n_out,
                self.loss_type, dtype, self.n_batches,
            )
        return results


def run_validation(
    main_cfg: dict, tasks_cfg: dict, select: list[str] | None,
    checkpoint: Path | None, device: str, batch_size: int, n_batches: int,
) -> dict[str, dict[str, Any]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    from tokenizers import Tokenizer

    tok_cfg = main_cfg.get("text_tokenizer_path")
    tok = Tokenizer.from_file(str(_resolve(tok_cfg) if tok_cfg else TEXT_TOKENIZER))
    suite = ValidationSuite(
        tasks_cfg, input_size=main_cfg.get("input_size", THINGS_IMAGE_SIZE),
        text_tokenizer=tok, loss_type=main_cfg.get("loss_type", "mod"),
        batch_size=batch_size, n_batches=n_batches, select=select,
    )

    # The standalone model spans every modality across the selected tasks (so the
    # suite never skips); the in-loop caller instead reuses the live training model.
    tasks = tasks_cfg["tasks"]
    in_union = sorted({d for n in suite.names for d in tasks[n]["in_domains"].split("-")})
    out_union = sorted({d for n in suite.names for d in tasks[n]["out_domains"].split("-")})
    print(f"model in={in_union} out={out_union} device={device.type}")
    model = build_model(in_union, out_union, main_cfg["model"],
                        main_cfg.get("input_size", THINGS_IMAGE_SIZE)).to(device)
    tokens_seen = None
    if checkpoint:
        tokens_seen = read_tokens_seen(load_checkpoint(model, checkpoint))

    results = suite.run(model, device, dtype)
    print(format_tokens_seen(tokens_seen))
    for name, stats in results.items():
        pretty = "  ".join(f"{k}={v:.4f}" for k, v in stats.items() if k != "n_batches")
        print(f"[{name}] {pretty}  (n_batches={stats['n_batches']})")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="main yaml (model, input_size, tokenizer)")
    p.add_argument("--tasks", type=Path, required=True, help="val-tasks yaml")
    p.add_argument("--select", type=str, default=None, help="comma-separated task names (default: all)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="model checkpoint (weights); overrides `checkpoint:` in the tasks YAML. "
                        "Omit both for a random-weight pipeline smoke.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-batches", type=int, default=4, help="batches per task (0 = whole val set)")
    args = p.parse_args()

    main_cfg = yaml.safe_load(_resolve(str(args.config)).read_text())
    tasks_cfg = yaml.safe_load(_resolve(str(args.tasks)).read_text())
    select = args.select.split(",") if args.select else None
    checkpoint = _resolve_checkpoint(args.checkpoint, main_cfg)

    results = run_validation(
        main_cfg, tasks_cfg, select, checkpoint, args.device, args.batch_size, args.n_batches,
    )
    print("\n=== validation summary ===")
    for name, stats in results.items():
        print(f"  {name:>16s}: loss={stats.get('loss', float('nan')):.4f}  (n_batches={stats['n_batches']})")


if __name__ == "__main__":
    main()
