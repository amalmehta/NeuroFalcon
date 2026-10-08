"""Score a decoder the way the FALCON harness does.

`score` streams files through the harness's own `FalconEvaluator.predict_files`
and computes R² with its `compute_metrics_regression`: variance-weighted R²
over eval_mask bins, one value per recording day (runs pooled), then mean and
std across days. Use it on labelled held-out *calibration* files to get a
local zero-shot number; the official held-out eval files live only on EvalAI.

`minival` runs the harness's official local path end to end.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import r2_score
from falcon_challenge.evaluator import FalconEvaluator
from falcon_challenge.interface import BCIDecoder

from neurofalcon.download import DATA_ROOT
from neurofalcon.sessions import CONFIG

RESULTS = Path(__file__).resolve().parent.parent / "results"


def score(decoder: BCIDecoder, paths: Sequence[Path]) -> dict:
    ev = FalconEvaluator(eval_remote=False, split="m2", dataloader_workers=0)
    preds, targets, masks, _, _ = ev.predict_files(decoder, list(paths))
    # Same ordering the harness uses: by date, then run.
    keys = sorted(preds, key=lambda k: (k.split("_")[1], k))
    lens = defaultdict(list)
    for k in keys:
        lens[k.split("_")[1]].append(len(masks[k]))
    pred = np.concatenate([preds[k][: len(masks[k])] for k in keys])
    tgt = np.concatenate([targets[k][masks[k]] for k in keys])
    mask = np.concatenate([masks[k] for k in keys])
    m = FalconEvaluator.compute_metrics_regression(pred, tgt, mask, lens)
    # Per-day values for reporting; their mean equals the harness's "R2 Mean".
    per_day = {}
    for day in sorted(lens):
        day_keys = [k for k in keys if k.endswith(day)]
        p = np.concatenate([preds[k][: len(masks[k])][masks[k]] for k in day_keys])
        t = np.concatenate([targets[k][masks[k]] for k in day_keys])
        per_day[day] = float(r2_score(t, p, multioutput="variance_weighted"))
    return {"r2_mean": float(m["R2 Mean"]), "r2_std": float(m["R2 Std."]), "per_day": per_day}


def minival(decoder: BCIDecoder) -> dict:
    """Official local evaluation (held-in minival) via FalconEvaluator.evaluate."""
    out = RESULTS / "harness"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EVAL_DATA_PATH", str(DATA_ROOT))
    os.environ["PREDICTION_PATH_LOCAL"] = str(out / "minival_prediction.pkl")
    os.environ["GT_PATH"] = str(out / "minival_gt.pkl")
    ev = FalconEvaluator(eval_remote=False, split="m2", dataloader_workers=0)
    result = ev.evaluate(decoder, phase="minival")
    return {k: float(v) for k, v in result["submission_result"]["minival_split_m2"].items()}
