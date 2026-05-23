"""Extract BrainOmni sensor metadata (pos, sensor_type) from MNE info.

Mirrors ``factory/utils.py::extract_pos_sensor_type`` without importing the
full BrainOmni factory stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch

from ..meg_config import MEG_DATA

# BrainOmni factory.brain_constant.SENSOR_TYPE_DICT
SENSOR_TYPE_EEG = 0
SENSOR_TYPE_MAG = 1
SENSOR_TYPE_GRAD = 2


@dataclass(frozen=True)
class SensorMetadata:
    """Fixed per-dataset sensor layout for THINGS-MEG."""

    pos: torch.Tensor          # (C, 6) float32 — xyz + orientation
    sensor_type: torch.Tensor  # (C,) int64 — 0=EEG, 1=MAG, 2=GRAD

    @property
    def n_channels(self) -> int:
        return int(self.sensor_type.shape[0])

    def batch(self, batch_size: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Expand to (B, C, 6) and (B, C) on ``device``."""
        pos = self.pos.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        st = self.sensor_type.unsqueeze(0).expand(batch_size, -1).to(device)
        return pos, st


def extract_pos_sensor_type_from_info(info) -> tuple[np.ndarray, np.ndarray]:
    """Build pos (C, 6) and sensor_type (C,) from an MNE Info object."""
    pos_list: list[np.ndarray] = []
    sensor_type_list: list[int] = []
    for ch in info["chs"]:
        kind = int(ch["kind"])
        if kind not in (1, 2):
            raise ValueError(f"Unknown sensor kind {kind} for channel {ch['ch_name']}")
        coil_type = str(ch["coil_type"])
        if kind == 2:
            pos_list.append(np.hstack([ch["loc"][:3], np.zeros(3, dtype=np.float64)]))
            sensor_type_list.append(SENSOR_TYPE_EEG)
        else:
            xyz = ch["loc"][:3]
            dir_idx = 3
            if "PLANAR" in coil_type:
                dir_idx = 1
            direction = ch["loc"][3 * dir_idx : 3 * (dir_idx + 1)]
            pos_list.append(np.hstack([xyz, direction]))
            if "MAG" in coil_type:
                sensor_type_list.append(SENSOR_TYPE_MAG)
            else:
                sensor_type_list.append(SENSOR_TYPE_GRAD)
    pos = np.stack(pos_list).astype(np.float32)
    sensor_type = np.array(sensor_type_list, dtype=np.int64)
    return pos, sensor_type


def load_things_meg_sensor_metadata(
    data_dir: str = MEG_DATA.data_dir,
    subject: str = "P1",
) -> SensorMetadata:
    """Read sensor layout from one THINGS-MEG epoch file (layout is shared)."""
    import mne

    mne.set_log_level("ERROR")
    epo_path = os.path.join(data_dir, f"preprocessed_{subject}-epo.fif")
    if not os.path.exists(epo_path):
        raise FileNotFoundError(f"Cannot load sensor metadata — missing {epo_path}")
    epochs = mne.read_epochs(epo_path, preload=False, verbose="ERROR")
    pos, sensor_type = extract_pos_sensor_type_from_info(epochs.info)
    if pos.shape[0] != MEG_DATA.n_channels:
        raise ValueError(
            f"Expected {MEG_DATA.n_channels} channels, got {pos.shape[0]} from {epo_path}"
        )
    return SensorMetadata(
        pos=torch.from_numpy(pos),
        sensor_type=torch.from_numpy(sensor_type),
    )
