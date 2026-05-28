"""Modal verification: TokenAccountant's count == a manual mask count, on the REAL
masking pipeline — and zero MEG/EEG tokens are counted for placeholder samples.

Why this exists: lib/token_accounting.py counts the *actual* selected tokens from
4M's per-modality masks rather than the stock closed-form product, so placeholder
MEG/EEG samples (whose Dirichlet budget is zeroed and not redistributed) are not
falsely counted. This script proves that on synthetic-but-real shards run through the
production ``PresenceAwareUnifiedMasking``:

  1. For real / placeholder / mixed neural shards, the accountant's running total
     equals an INDEPENDENT manual count of ``~input_mask`` / ``~target_mask`` over the
     exact same batches (cross-checked against ``decoder_attention_mask`` for targets).
  2. PLACEHOLDER shards: every neural modality contributes 0 input and 0 target
     tokens — the module counts no MEG/EEG for placeholders.
  3. REAL shards: neural modalities DO contribute (sanity that real neural is counted).
  4. End-to-end: with the hook registered the production way and a real model forward,
     the accountant (populated only by the firing hook) still matches the manual count.

Run (CPU, no volume, no GPU cost):

    modal run 4m_training/modal/modal_verify_token_count.py
    modal run 4m_training/modal/modal_verify_token_count.py --n-examples 8
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import modal


def _load_modal_image():
    for path in (
        Path("/opt/repo/4m_training/modal/_modal_load.py"),
        Path(__file__).resolve().parent / "_modal_load.py",
    ):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_modal_load", path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        return loader.load_modal_image()
    raise ImportError("_modal_load.py not found (expected /opt/repo/4m_training on Modal)")


_mi = _load_modal_image()
REPO = _mi.REPO
ensure_fourm = _mi.ensure_fourm
train_image = _mi.train_image

app = modal.App("verify-token-count")


# ── container-side: build shards, run the real masking, compare counts ────────


def _prepare_env() -> None:
    """Make fourm + our lib importable in-process (mirrors training_env, no subprocess)."""
    import os
    import sys

    ensure_fourm()
    # We import fourm IN-PROCESS (no subprocess), so the editable install's .pth is not
    # auto-loaded into this interpreter's sys.path — add the ml-4m dir explicitly, the same
    # path training_env() puts on PYTHONPATH for subprocesses.
    for p in (f"{REPO}/4m_training", f"{REPO}/4m_training/lib", _mi.ML4M_CONTAINER):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("FOURM_ML4M_DIR", _mi.ML4M_CONTAINER)
    for k, v in {
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
    }.items():
        os.environ.setdefault(k, v)


def _npy_bytes(arr) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _make_tar(path: Path, entries: list[tuple[str, bytes]]) -> None:
    import io
    import tarfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, data in entries:
            info = tarfile.TarInfo(name=f"{key}.npy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _build_shards(root: Path, ids: list[str], neural_mode: str) -> str:
    """Write THINGS-style shards. ``neural_mode``: 'real' | 'placeholder' | 'mixed'.

    Placeholder samples carry the sentinel array (-1) AND a 0 presence mask, exactly as
    the production data does for image ids without MEG/EEG. Returns the WDS data_path.
    """
    import numpy as np

    from neural_constants import (
        EEG_TRIAL_SHAPE,
        EEG_VOCAB_SIZE,
        MEG_TRIAL_SHAPE,
        MEG_VOCAB_SIZE,
        NEURAL_SENTINEL_VALUE,
        TOK_DEPTH_VOCAB_SIZE,
        TOK_RGB_TOKENS_PER_IMAGE,
        TOK_RGB_VOCAB_SIZE,
    )

    rng = np.random.default_rng(0)

    def is_real(i: int) -> bool:
        return {"real": True, "placeholder": False, "mixed": i % 2 == 0}[neural_mode]

    folders: dict[str, list[tuple[str, bytes]]] = {
        m: [] for m in ("tok_rgb", "tok_depth", "tok_meg", "tok_eeg", "meg_mask", "eeg_mask")
    }
    for i, key in enumerate(ids):
        real = is_real(i)
        folders["tok_rgb"].append((key, _npy_bytes(
            rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))))
        folders["tok_depth"].append((key, _npy_bytes(
            rng.integers(0, TOK_DEPTH_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))))
        if real:
            meg = rng.integers(0, MEG_VOCAB_SIZE, (4, *MEG_TRIAL_SHAPE), dtype=np.int16)
            eeg = rng.integers(0, EEG_VOCAB_SIZE, (2, *EEG_TRIAL_SHAPE), dtype=np.int16)
        else:
            meg = np.full((1, *MEG_TRIAL_SHAPE), NEURAL_SENTINEL_VALUE, dtype=np.int16)
            eeg = np.full((1, *EEG_TRIAL_SHAPE), NEURAL_SENTINEL_VALUE, dtype=np.int16)
        folders["tok_meg"].append((key, _npy_bytes(meg)))
        folders["tok_eeg"].append((key, _npy_bytes(eeg)))
        flag = np.array([1 if real else 0], dtype=np.uint8)
        folders["meg_mask"].append((key, _npy_bytes(flag)))
        folders["eeg_mask"].append((key, _npy_bytes(flag)))

    for mod, entries in folders.items():
        _make_tar(root / mod / "shard_000.tar", entries)
    return f"{root}/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/shard_{{000..000}}.tar"


# Modalities whose tokens must NOT be counted for placeholder samples.
def _neural_names():
    from neural_constants import EEG_MODALITY, MEG_RVQ_MODALITIES

    return set(MEG_RVQ_MODALITIES) | {EEG_MODALITY}


def _build_loader(data_path: str, n_in: int, n_out: int, batch_size: int):
    from fourm.data.modality_info import MODALITY_TRANSFORMS
    from fourm.data.pretrain_utils import setup_sampling_mod_info

    from fourm_dataloader import _wds_eval_loader
    from neural_constants import EEG_MODALITY, MEG_RVQ_MODALITIES, THINGS_IMAGE_SIZE
    from repo_paths import TEXT_TOKENIZER
    from things_augmenter import ThingsImageAugmenter
    from tokenizers import Tokenizer
    from train_4m import _build_modality_info

    in_domains = sorted(["tok_rgb", "tok_depth", *MEG_RVQ_MODALITIES, EEG_MODALITY])
    all_domains = in_domains  # symmetric: neural is both encoder input and decoder target
    ds_cfg = {
        "in_domains": "-".join(in_domains),
        "out_domains": "-".join(in_domains),
        # Uniform Dirichlet so neural reliably receives budget in the 'real' scenario.
        "input_alphas": "-".join(["1.0"] * len(in_domains)),
        "target_alphas": "-".join(["1.0"] * len(in_domains)),
    }
    full = _build_modality_info(all_domains, THINGS_IMAGE_SIZE)
    mod_info, sampling_weights = setup_sampling_mod_info(ds_cfg, full)
    tok = Tokenizer.from_file(str(TEXT_TOKENIZER))
    augmenter = ThingsImageAugmenter(
        target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
    )
    loader = _wds_eval_loader(
        data_path=data_path, all_domains=all_domains, modality_info=mod_info,
        modality_transforms=MODALITY_TRANSFORMS, image_augmenter=augmenter,
        text_tokenizer=tok, input_tokens_range=(n_in, n_in),
        target_tokens_range=(n_out, n_out), num_workers=1, batch_size=batch_size,
        sampling_weights=sampling_weights,
    )
    return loader, all_domains, mod_info


def _manual_count(mod_dict) -> dict[str, tuple[int, int, int | None]]:
    """Independent per-modality count from the masks 4M produced for this batch.

    Returns ``{modality: (input_selected, target_selected, decoder_attention_sum)}``.
    For image / neural-grid modalities ``decoder_attention_mask`` independently encodes
    the target budget, so it cross-checks ``~target_mask`` from a different field.
    """
    rows: dict[str, tuple[int, int, int | None]] = {}
    for name, v in mod_dict.items():
        if not isinstance(v, dict) or "input_mask" not in v:
            continue
        in_sel = int((~v["input_mask"]).sum())
        tm = v.get("target_mask")
        tgt_sel = int((~tm).sum()) if tm is not None else 0
        dec = v.get("decoder_attention_mask")
        dec_sum = int(dec.sum()) if dec is not None else None
        rows[name] = (in_sel, tgt_sel, dec_sum)
    return rows


def _run_scenario(neural_mode: str, n_examples: int, n_in: int, n_out: int) -> dict:
    """Drive one shard scenario through the real masking and compare counts."""
    import tempfile
    import types

    import torch

    from token_accounting import TokenAccountant

    neural = _neural_names()
    acc = TokenAccountant()
    manual_in = manual_tgt = 0
    neural_in = neural_tgt = vision_in = vision_tgt = 0
    fake_module = types.SimpleNamespace(training=True)

    with tempfile.TemporaryDirectory() as tmp:
        ids = [f"{i:09d}" for i in range(n_examples)]
        data_path = _build_shards(Path(tmp) / "things", ids, neural_mode)
        loader, all_domains, _ = _build_loader(data_path, n_in, n_out, batch_size=1)

        print(f"\n=== scenario: {neural_mode} ({n_examples} examples, n_in={n_in} n_out={n_out}) ===")
        header = f"{'example':>8}  {'modality':>14}  {'in':>4} {'tgt':>4} {'decAttn':>7}  {'neural?':>7}"
        seen = 0
        for ex, batch in enumerate(loader):
            if ex >= n_examples:
                break
            seen += 1
            rows = _manual_count(batch)
            ex_in = ex_tgt = 0
            print(header if ex == 0 else "")
            for name in sorted(rows):
                in_sel, tgt_sel, dec_sum = rows[name]
                is_n = name in neural
                # Cross-check: decoder_attention_mask sum must equal ~target_mask count
                # for these (img / neural-grid) modalities.
                assert dec_sum is None or dec_sum == tgt_sel, (
                    f"{name}: decoder_attention({dec_sum}) != target_selected({tgt_sel})")
                ex_in += in_sel
                ex_tgt += tgt_sel
                if is_n:
                    neural_in += in_sel
                    neural_tgt += tgt_sel
                else:
                    vision_in += in_sel
                    vision_tgt += tgt_sel
                print(f"{ex:>8}  {name:>14}  {in_sel:>4} {tgt_sel:>4} "
                      f"{('-' if dec_sum is None else dec_sum):>7}  {str(is_n):>7}")
            manual_in += ex_in
            manual_tgt += ex_tgt

            # The module: call the hook exactly as PyTorch's forward-pre-hook would.
            acc.hook(fake_module, (batch,))
            # Running totals must match the manual cumulative sum after every example.
            t = acc.totals()
            assert t["input"] == manual_in and t["target"] == manual_tgt, (
                f"running mismatch @ex{ex}: module={t} manual=(in={manual_in},tgt={manual_tgt})")
            print(f"{'':>8}  {'EXAMPLE TOTAL':>14}  {ex_in:>4} {ex_tgt:>4}  "
                  f"| module running: in={t['input']} tgt={t['target']} total={t['total']}")

    final = acc.totals()
    assert seen == n_examples, f"{neural_mode}: expected {n_examples} batches, got {seen}"
    assert final["input"] == manual_in, f"{neural_mode}: input {final['input']} != manual {manual_in}"
    assert final["target"] == manual_tgt, f"{neural_mode}: target {final['target']} != manual {manual_tgt}"

    # The training-gate: an eval forward (module.training False) must not be counted.
    acc.hook(types.SimpleNamespace(training=False), (batch,))
    assert acc.totals() == final, "eval forward was counted (training gate broken)"

    print(f"--- {neural_mode}: module total == manual total "
          f"(input={final['input']}, target={final['target']}, total={final['total']}); "
          f"neural in/tgt={neural_in}/{neural_tgt}, vision in/tgt={vision_in}/{vision_tgt}")
    return {
        "mode": neural_mode, "module": final,
        "manual": {"input": manual_in, "target": manual_tgt},
        "neural": {"input": neural_in, "target": neural_tgt},
        "vision": {"input": vision_in, "target": vision_tgt},
    }


def _run_end_to_end(n_examples: int, n_in: int, n_out: int) -> str:
    """Faithful path: register the hook the production way and run a real model forward,
    confirming the hook-populated accountant matches an independent manual count."""
    import tempfile
    import types

    import torch

    from token_accounting import TokenAccountant
    from validate_4m import build_model

    acc = TokenAccountant()
    manual_in = manual_tgt = 0
    with tempfile.TemporaryDirectory() as tmp:
        ids = [f"{i:09d}" for i in range(n_examples)]
        data_path = _build_shards(Path(tmp) / "things", ids, "mixed")
        loader, all_domains, _ = _build_loader(data_path, n_in, n_out, batch_size=1)

        model = build_model(all_domains, all_domains, "fm_tiny_6e_6d_swiglu_nobias", 224)
        model.train()  # so the hook's training gate is open
        model.register_forward_pre_hook(acc.hook)  # the real registration path

        for ex, batch in enumerate(loader):
            if ex >= n_examples:
                break
            rows = _manual_count(batch)
            manual_in += sum(r[0] for r in rows.values())
            manual_tgt += sum(r[1] for r in rows.values())
            with torch.no_grad():  # fire the hook via a genuine forward, no graph
                model(batch, num_encoder_tokens=n_in, num_decoder_tokens=n_out, loss_type="mod")

    final = acc.totals()
    assert final["input"] == manual_in and final["target"] == manual_tgt, (
        f"end-to-end mismatch: module={final} manual=(in={manual_in},tgt={manual_tgt})")
    msg = (f"end-to-end (real forward, registered hook): module total == manual "
           f"(input={final['input']}, target={final['target']}, total={final['total']})")
    print(f"--- {msg}")
    return msg


@app.function(image=train_image, timeout=60 * 15, memory=16 * 1024)
def verify(n_examples: int = 6) -> str:
    _prepare_env()

    from fourm_dataloader import patch_pretrain_utils

    patch_pretrain_utils()  # install PresenceAwareUnifiedMasking + neural modalities

    t0 = time.time()
    n_in, n_out = 128, 128
    results = {m: _run_scenario(m, n_examples, n_in, n_out)
               for m in ("real", "placeholder", "mixed")}

    # (2) Placeholder shards: NO MEG/EEG tokens counted.
    ph = results["placeholder"]
    assert ph["neural"]["input"] == 0 and ph["neural"]["target"] == 0, (
        f"placeholder counted neural tokens: {ph['neural']}")
    assert ph["vision"]["input"] > 0 and ph["vision"]["target"] > 0, "placeholder lost vision tokens"

    # (3) Real shards: neural IS counted (sanity).
    rl = results["real"]
    assert rl["neural"]["input"] > 0 or rl["neural"]["target"] > 0, "real neural never counted"

    # (4) Faithful end-to-end via a real model forward.
    e2e = _run_end_to_end(n_examples, n_in, n_out)

    lines = [
        "ALL CHECKS PASSED",
        f"  real:        module={rl['module']}  neural in/tgt={rl['neural']['input']}/{rl['neural']['target']}",
        f"  placeholder: module={ph['module']}  neural in/tgt=0/0 (correctly excluded)",
        f"  mixed:       module={results['mixed']['module']}",
        f"  {e2e}",
        f"  elapsed {time.time() - t0:.1f}s",
    ]
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    return summary


@app.local_entrypoint()
def main(n_examples: int = 6) -> None:
    print(verify.remote(n_examples=n_examples))
