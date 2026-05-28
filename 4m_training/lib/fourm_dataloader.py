"""Minimal hooks so stock 4M dataloaders work with THINGS neural shards.

We do **not** reimplement the WebDataset pipeline. Instead we patch three
integration points and delegate everything else to upstream 4M:

  * ``UnifiedMasking`` → ``PresenceAwareUnifiedMasking`` (meg/eeg absence)
  * ``PreTokenizedImageAugmenter`` → ``ThingsImageAugmenter`` (optional crop_settings)
  * ``get_val_dataloader`` → WDS branch when ``use_wds: true`` (stock val is folder-only)

Train uses stock ``get_train_dataloader`` → ``build_wds_fm_pretraining_dataloader`` unchanged.
Neural transforms are stock ``AbstractTransform`` subclasses registered in
``MODALITY_TRANSFORMS`` via ``fourm_neural_modalities``.
"""

from __future__ import annotations

import copy
from functools import partial

import webdataset as wds
from fourm.data import pretrain_utils
from fourm.data import unified_datasets as ud
from fourm.data import image_augmenter as ia
from fourm.data.modality_transforms import CropSettingsTransform, IdentityTransform
from fourm.data.pretrain_utils import cfgs_get
from fourm.data.unified_datasets import (
    default_collate,
    filter_metadata,
    map,
    multi_tarfile_samples,
    remove_extensions,
    rename_modalities,
    tok_to_int64,
    wds_decoder,
)
import torchvision.transforms as transforms
from fourm.data.modality_transforms import UnifiedDataTransform

from fourm.utils import logger as _fourm_logger

import fourm_neural_modalities
from neural_masking import PresenceAwareUnifiedMasking
from things_augmenter import ThingsImageAugmenter

_PRESENCE_PATHS = {"meg_mask": "meg_mask", "eeg_mask": "eeg_mask"}

# Stock 4M hardcodes ``print_freq = 10`` in its train/eval loops (no CLI/YAML knob), which is
# noisy for long epochs. We override it process-wide via the ``MetricLogger.log_every`` patch
# below; train_4m sets this from the main YAML's ``print_freq`` field. None = leave stock (10).
_LOG_PRINT_FREQ: int | None = None


def set_log_print_freq(n: int | None) -> None:
    """Set the progress-print interval (steps) for the stock trainer's MetricLogger."""
    global _LOG_PRINT_FREQ
    _LOG_PRINT_FREQ = int(n) if n else None

# Picks ONE trial per sample and splits it into the neural output heads (coherent across
# the 4 MEG RVQ layers). None until a loader is built; set for train vs eval trial sampling.
_NEURAL_SPLITTER = None

_ORIG_UNIFIED_MASKING = ud.UnifiedMasking
_ORIG_PRETOKENIZED_AUG = ia.PreTokenizedImageAugmenter
_ORIG_GET_VAL = pretrain_utils.get_val_dataloader
_ORIG_RENAME_MODALITIES = ud.rename_modalities
_ORIG_LOG_EVERY = _fourm_logger.MetricLogger.log_every
_PATCHED = False


def _log_every(self, iterable, print_freq, *args, **kwargs):
    """``MetricLogger.log_every`` with a configurable print interval (see ``set_log_print_freq``)."""
    if _LOG_PRINT_FREQ is not None:
        print_freq = max(1, _LOG_PRINT_FREQ)
    return _ORIG_LOG_EVERY(self, iterable, print_freq, *args, **kwargs)


class _SeqSafeRandom:
    """Proxy for the stdlib ``random`` module that lists ``sample`` populations.

    Stock 4M's decoder does ``random.sample(mod_dict.items(), …)``; ``random.sample``
    rejected dict views / sets starting in Python 3.11. Swapping ``fm.random`` for
    this proxy keeps the model forward working on 3.11+ without editing ``ml-4m`` —
    every other attribute delegates to the real module unchanged.
    """

    def __init__(self, real):
        self._real = real

    def sample(self, population, k):
        return self._real.sample(list(population), k)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _set_neural_target_splitter(training: bool) -> None:
    """Install the per-sample neural-output trial splitter (train vs eval sampling)."""
    global _NEURAL_SPLITTER
    from neural_trial_transform import NeuralTargetSplitter

    _NEURAL_SPLITTER = NeuralTargetSplitter(training=training)


def _rename_modalities(sample, modality_paths):
    """Stock rename assumes every path exists; THINGS omits optional ``crop_settings`` tars.

    Also the single point where neural OUTPUT heads are materialized: rename fans the
    ``tok_meg`` / ``tok_eeg`` folder out to the output modalities (aliasing one array),
    then the splitter picks one trial and slices the per-head targets — so the 4 MEG RVQ
    heads stay coherent. No-op when no neural output modality is requested.
    """
    present = {out: src for out, src in modality_paths.items() if src in sample}
    if not present:
        return sample
    renamed = _ORIG_RENAME_MODALITIES(sample, present)
    if _NEURAL_SPLITTER is not None:
        _NEURAL_SPLITTER(renamed)
    return renamed


def _extend_modality_paths(modality_info: dict) -> dict[str, str]:
    paths = {mod: modality_info[mod].get("path", None) or mod for mod in modality_info}
    paths.update(_PRESENCE_PATHS)
    return paths


def _wds_eval_loader(
    data_path: str,
    all_domains: list[str],
    modality_info: dict,
    modality_transforms: dict,
    image_augmenter,
    text_tokenizer,
    input_tokens_range,
    target_tokens_range,
    num_workers: int,
    batch_size: int,
    sampling_weights=None,
    modality_name_map=None,
):
    """Deterministic WDS val loader — only gap stock 4M does not cover.

    Reuses the same transform stack as ``build_wds_fm_pretraining_dataloader``
    (``UnifiedDataTransform`` + patched ``PresenceAwareUnifiedMasking``).
    Pipeline differs only in shard source (``SimpleShardList``, no shuffle/repeat).
    """
    fourm_neural_modalities.register(training=False)
    _set_neural_target_splitter(training=False)  # eval: deterministic trial 0

    modality_paths = _extend_modality_paths(modality_info)
    modality_transforms = copy.deepcopy(modality_transforms)
    if any(modality_info[d].get("pretokenized", False) for d in all_domains):
        modality_transforms["crop_settings"] = CropSettingsTransform()
    modality_transforms["__key__"] = IdentityTransform()

    transform = transforms.Compose(
        [
            UnifiedDataTransform(
                transforms_dict=modality_transforms,
                image_augmenter=image_augmenter,
            ),
            PresenceAwareUnifiedMasking(
                modality_info=modality_info,
                text_tokenizer=text_tokenizer,
                input_tokens_range=input_tokens_range,
                target_tokens_range=target_tokens_range,
                sampling_weights=sampling_weights,
            ),
        ]
    )

    datapipe = wds.DataPipeline(
        wds.SimpleShardList(data_path),
        partial(multi_tarfile_samples, modality_name_map=modality_name_map),
        wds.decode(wds_decoder),
        wds.map(remove_extensions),
        map(filter_metadata),
        map(tok_to_int64),
        # Tolerant rename: a task's shards may omit optional folders (e.g. meg_mask
        # for a vision-only val task), so skip paths not present in the sample.
        map(partial(_rename_modalities, modality_paths=modality_paths)),
        map(transform),
        wds.batched(batch_size, collation_fn=default_collate, partial=False),
    )
    return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)


def _neural_get_val_dataloader(
    dataset_config,
    dataset_name,
    train_configs,
    modality_info,
    sampling_weights,
    text_tokenizer,
    input_size,
    num_input_tokens,
    num_target_tokens,
    min_input_tokens,
    min_target_tokens,
    fixed_eval,
    fixed_eval_input_tokens,
    fixed_eval_target_tokens,
    dist_eval,
    num_tasks,
    num_workers,
    batch_size,
    pin_mem,
):
    """Stock val loader, plus WDS tars when ``use_wds: true`` in val config."""
    from fourm.data.modality_info import MODALITY_TRANSFORMS
    from fourm.data.modality_transforms import CaptionTransform
    from fourm.data.image_augmenter import CenterCropImageAugmenter

    dataset_type = cfgs_get("type", dataset_config, dataset_name, train_configs)
    use_wds = cfgs_get("use_wds", dataset_config, dataset_name, train_configs, False)

    if not (dataset_type == "multimodal" and use_wds):
        return _ORIG_GET_VAL(
            dataset_config,
            dataset_name,
            train_configs,
            modality_info,
            sampling_weights,
            text_tokenizer,
            input_size,
            num_input_tokens,
            num_target_tokens,
            min_input_tokens,
            min_target_tokens,
            fixed_eval,
            fixed_eval_input_tokens,
            fixed_eval_target_tokens,
            dist_eval,
            num_tasks,
            num_workers,
            batch_size,
            pin_mem,
        )

    in_domains = sorted(
        cfgs_get("in_domains", dataset_config, dataset_name, train_configs).split("-")
    )
    out_domains = sorted(
        cfgs_get("out_domains", dataset_config, dataset_name, train_configs).split("-")
    )
    all_domains = sorted(set(in_domains) | set(out_domains))

    modality_transforms = MODALITY_TRANSFORMS
    if "caption" in modality_transforms:
        modality_transforms["caption"] = CaptionTransform(
            aligned_captions=cfgs_get(
                "aligned_captions", dataset_config, dataset_name, train_configs, True
            )
        )

    main_augment_domain = cfgs_get(
        "main_augment_domain", dataset_config, dataset_name, train_configs
    )
    is_pretokenized = any(
        modality_info[mod].get("pretokenized", False) for mod in modality_info
    )
    eval_image_augmenter = (
        ThingsImageAugmenter(
            target_size=input_size, no_aug=True, main_domain=main_augment_domain
        )
        if is_pretokenized
        else CenterCropImageAugmenter(
            target_size=input_size, main_domain=main_augment_domain
        )
    )

    if fixed_eval:
        input_tokens_range = (fixed_eval_input_tokens, fixed_eval_input_tokens)
        target_tokens_range = (fixed_eval_target_tokens, fixed_eval_target_tokens)
    else:
        num_input_tokens = dataset_config.get("num_input_tokens", num_input_tokens)
        num_target_tokens = dataset_config.get("num_target_tokens", num_target_tokens)
        min_input_tokens = dataset_config.get("min_input_tokens", min_input_tokens)
        min_target_tokens = dataset_config.get("min_target_tokens", min_target_tokens)
        min_input_tokens = (
            num_input_tokens if min_input_tokens is None else min_input_tokens
        )
        min_target_tokens = (
            num_target_tokens if min_target_tokens is None else min_target_tokens
        )
        input_tokens_range = (min_input_tokens, num_input_tokens)
        target_tokens_range = (min_target_tokens, num_target_tokens)

    print(
        "Warning: Eval stats may vary slightly as the masking applied on images is random."
    )
    return _wds_eval_loader(
        data_path=cfgs_get("data_path", dataset_config, dataset_name, train_configs),
        all_domains=all_domains,
        modality_info=modality_info,
        modality_transforms=modality_transforms,
        image_augmenter=eval_image_augmenter,
        text_tokenizer=text_tokenizer,
        input_tokens_range=input_tokens_range,
        target_tokens_range=target_tokens_range,
        num_workers=num_workers,
        batch_size=batch_size,
        sampling_weights=sampling_weights,
        modality_name_map=cfgs_get(
            "modality_name_map", dataset_config, dataset_name, train_configs
        ),
    )


def patch_pretrain_utils() -> None:
    """Install neural hooks into stock 4M (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    # Stock WDS + folder datasets instantiate these classes by reference.
    ud.UnifiedMasking = PresenceAwareUnifiedMasking
    ia.PreTokenizedImageAugmenter = ThingsImageAugmenter
    pretrain_utils.PreTokenizedImageAugmenter = ThingsImageAugmenter
    ud.rename_modalities = _rename_modalities
    _fourm_logger.MetricLogger.log_every = _log_every  # configurable progress-print interval

    # Make the decoder forward's random.sample(dict.items()) work on Python 3.11+.
    from fourm.models import fm as _fm

    if not isinstance(_fm.random, _SeqSafeRandom):
        _fm.random = _SeqSafeRandom(_fm.random)

    fourm_neural_modalities.register(training=True)
    _set_neural_target_splitter(training=True)  # train: random trial per sample

    pretrain_utils.get_val_dataloader = _neural_get_val_dataloader
    import fourm.data as fd

    fd.get_val_dataloader = _neural_get_val_dataloader
    _PATCHED = True


def unpatch_pretrain_utils() -> None:
    """Restore stock 4M (for tests)."""
    global _PATCHED, _NEURAL_SPLITTER
    _NEURAL_SPLITTER = None
    ud.UnifiedMasking = _ORIG_UNIFIED_MASKING
    ia.PreTokenizedImageAugmenter = _ORIG_PRETOKENIZED_AUG
    pretrain_utils.PreTokenizedImageAugmenter = _ORIG_PRETOKENIZED_AUG
    ud.rename_modalities = _ORIG_RENAME_MODALITIES
    _fourm_logger.MetricLogger.log_every = _ORIG_LOG_EVERY
    pretrain_utils.get_val_dataloader = _ORIG_GET_VAL
    import fourm.data as fd

    fd.get_val_dataloader = _ORIG_GET_VAL

    from fourm.models import fm as _fm

    if isinstance(_fm.random, _SeqSafeRandom):
        _fm.random = _fm.random._real
    _PATCHED = False
