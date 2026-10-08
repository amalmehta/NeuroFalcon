"""Day 1: fit the Wiener filter on held-in days and score it zero-shot.

    python -m neurofalcon.baseline

Writes results/baseline.json and results/Zero-Shot Wiener by Day.png.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurofalcon import scoring, sessions
from neurofalcon.scoring import RESULTS
from neurofalcon.wiener import WienerDecoder, WienerFilter

LAST_HELD_IN = date(2020, 10, 28)
CHART = RESULTS / "Zero-Shot Wiener by Day.png"


def days_after_held_in(day: str) -> int:
    return (date(int(day[:4]), int(day[4:6]), int(day[6:])) - LAST_HELD_IN).days


def chart(result: dict) -> None:
    held_out = result["zero_shot_held_out"]["per_day"]
    x = [days_after_held_in(d) for d in held_out]
    y = list(held_out.values())
    held_in = result["minival"]["Held In R2 Mean"]
    ink, muted, blue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.bar([str(v) for v in x], y, width=0.55, color=blue, zorder=2)
    ax.axhline(held_in, color=muted, lw=1, ls="--", zorder=1)
    ax.text(len(x) - 0.5, held_in, f"held-in minival {held_in:.2f}", ha="right", va="bottom",
            color=muted, fontsize=8)
    for i, v in enumerate(y):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", color=ink, fontsize=8)
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    ax.set_xlabel("Days after the last held-in session", color=muted, fontsize=9)
    ax.set_ylabel("R² (finger velocity)", color=muted, fontsize=9)
    ax.set_title("Zero-shot Wiener filter on held-out M2 days", color=ink, fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0cfca")
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color="#ecebe7", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(CHART, facecolor=surface)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--history", type=int, default=10, help="lag bins (20 ms each)")
    args = p.parse_args(argv)

    train = sessions.load_split("held-in-calib")
    wf = WienerFilter(history=args.history).fit(train)
    print(f"fit on {len(train)} held-in runs, history={wf.history}, ridge alpha={wf.alpha:g}")

    result = {
        "decoder": {"name": "Wiener filter", "history_bins": wf.history, "ridge_alpha": wf.alpha},
        "minival": scoring.minival(WienerDecoder(wf)),
        "zero_shot_held_out": scoring.score(WienerDecoder(wf), sessions.files("held-out-calib")),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.json").write_text(json.dumps(result, indent=2))
    chart(result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
