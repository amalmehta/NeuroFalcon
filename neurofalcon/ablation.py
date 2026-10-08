"""Day 3: ablation. YOURS TO WRITE. This file is only the contract.

CONTEXT doesn't say which component gets ablated, so this file only fixes
the output shape the writeup will read. Each row is one variant scored the
same way as the baseline (scoring.score on held-out calibration runs, plus
scoring.minival).
"""
from __future__ import annotations


def run_ablation() -> list[dict]:
    """Return rows like {"variant", "held_in_r2", "held_out_r2"}."""
    raise NotImplementedError("Ablation: yours to write")
