"""Find and load FALCON M2 sessions through the harness's own loader.

A "run" is one NWB file (e.g. Run1_20201030). A "session" is one recording
day; the harness scores R² per day, pooling that day's runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.dataloaders import load_nwb


CONFIG = FalconConfig(task=FalconTask.m2)  # 96 channels, 2 outputs, 20 ms bins
DANDISET = "000953"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Run:
    path: Path
    key: str                # harness key, e.g. "Run1_20201030"
    spikes: np.ndarray      # (T, 96) uint8 spike counts per 20 ms bin
    velocity: np.ndarray    # (T, 2) float32 finger velocity (index, mrs)
    trial_change: np.ndarray  # (T,) bool, True on the first bin of a trial
    eval_mask: np.ndarray   # (T,) bool, True where the bin is scored

    @property
    def date(self) -> str:
        return self.key.split("_")[1]


def files(split: str, root: Path = DATA_ROOT) -> list[Path]:
    """split: 'held-in-calib', 'held-out-calib' or 'held-in-minival'."""
    found = sorted((root / DANDISET / f"sub-MonkeyN-{split}").glob("*.nwb"))
    found = [f for f in found if not f.name.endswith(".partial.nwb")]
    if not found:
        raise FileNotFoundError(f"No {split} files under {root}. Run: python -m neurofalcon.download")
    return found


def load(path: Path) -> Run:
    spikes, vel, trial_change, eval_mask = load_nwb(path, FalconTask.m2)
    return Run(path, CONFIG.hash_dataset(path.stem), spikes, vel, trial_change, eval_mask)


def load_split(split: str, root: Path = DATA_ROOT) -> list[Run]:
    return [load(p) for p in files(split, root)]
