"""Build the EvalAI submission model: submission/decoder.pt.

    python -m neurofalcon.submission [--method full]

FALCON lets a decoder use each held-out day's calibration data. The file
holds one network per held-out day, recalibrated on all of that day's
calibration runs with `--method` (see recalibration.py), plus "default", the
held-in pretrained GRU, for held-in days. GRUDecoder picks the network from
each eval file's date.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neurofalcon import gru, recalibration, sessions

OUT = Path(__file__).resolve().parent.parent / "submission" / "decoder.pt"


def build(method: str = "full", seed: int = 0, out: Path = OUT) -> Path:
    base = gru.pretrained(seeds=seed + 1)[seed]
    nets = {"default": base}
    for day, runs in recalibration.by_day(sessions.load_split("held-out-calib")).items():
        net = gru.clone(base)
        if method == "renorm":
            net.set_input_stats(torch.cat([torch.as_tensor(r.spikes) for r in runs]).numpy())
        elif method != "zero-shot":
            params = {"input layer": net.inp.parameters(), "readout": net.out.parameters(),
                      "full": net.parameters()}[method]
            gru.train(net, runs, params=params, steps=recalibration.ADAPT_STEPS, lr=1e-3, seed=seed)
        nets[day] = net
        print(f"{day}: {method} on {sum(len(r.spikes) for r in runs) * 0.02:.0f} s", flush=True)
    out.parent.mkdir(exist_ok=True)
    gru.save(nets, out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build submission/decoder.pt")
    p.add_argument("--method", default="full", choices=["zero-shot", "renorm", "input layer", "readout", "full"])
    p.add_argument("--seed", type=int, default=0)
    print(build(**vars(p.parse_args())))
