"""Day 2: recalibration curve. YOURS TO WRITE. This file is only the contract.

The question the curve answers: how much labelled data from a new day does a
decoder need to recover? The x axis is seconds of held-out calibration data
used to adapt; the y axis is R² on that day's remaining bins. Zero seconds is
the zero-shot score from neurofalcon.baseline.

Decisions left to you (they're the part you'll be asked about):
  - what adapts: everything, a readout layer, or input normalisation only
  - the budgets to sweep, and whether to repeat with different seeds
  - how to keep the adaptation data and the scored data disjoint in time
"""
from __future__ import annotations

from typing import Callable, Sequence

from neurofalcon.sessions import Run


def recalibration_curve(
    base_decoder,
    held_out_runs: Sequence[Run],
    budgets_s: Sequence[float],
    adapt: Callable,
) -> list[dict]:
    """Return one row per (day, budget): {"day", "budget_s", "r2"}."""
    raise NotImplementedError("Recalibration loop: yours to write")
