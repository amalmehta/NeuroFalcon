"""Day 2: recalibration curve.

How much labelled data from a new day does a decoder need to recover?

    python -m neurofalcon.recalibration     # uses models/ from `python -m neurofalcon.gru`

Protocol, per held-out day:
  - every calibration run is cut in half. The first halves, in time order,
    are the adaptation pool. The second halves are the test set. The two
    never overlap.
  - a budget of B seconds takes the first B*50 bins of the pool.
  - every model predicts each whole run from its start (so the test half
    sees the same causal context it would when streamed), and R² is
    variance-weighted over that day's test bins, the harness's metric.
  - the curve's x axis is B; B = 0 is zero-shot on the same test bins.

Adaptation methods (the ablation is the comparison between them):
  zero-shot       no change
  renorm          re-z-score inputs with the budget's spike statistics (no labels)
  input layer     fine-tune only `inp` (96 -> 32), labels used
  readout         fine-tune only `out` (64 -> 2)
  renorm + readout  renorm, then fine-tune `out`
  full            fine-tune every weight
  scratch         train a fresh GRU on the budget only (no held-in pretraining)
  Wiener refit    fit a new Wiener filter on the budget only, ridge strength
                  chosen on a time-ordered 80/20 split of the budget
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import r2_score

from neurofalcon import gru, sessions
from neurofalcon.scoring import RESULTS
from neurofalcon.sessions import Run
from neurofalcon.wiener import WienerFilter

BIN_S = 0.02
BUDGETS_S = (2.5, 5.0, 10.0, 20.0)
ADAPT_STEPS = 300
SCRATCH_STEPS = 600  # same as pretraining
METHODS = ("renorm", "renorm + readout", "input layer", "readout", "full", "scratch", "Wiener refit")


def _slice(r: Run, a: int, b: int) -> Run:
    return replace(r, spikes=r.spikes[a:b], velocity=r.velocity[a:b],
                   trial_change=r.trial_change[a:b], eval_mask=r.eval_mask[a:b])


def by_day(runs: Sequence[Run]) -> dict[str, list[Run]]:
    days: dict[str, list[Run]] = {}
    for r in sorted(runs, key=lambda r: r.key):
        days.setdefault(r.date, []).append(r)
    return dict(sorted(days.items()))


def pool(day_runs: Sequence[Run]) -> list[Run]:
    return [_slice(r, 0, len(r.spikes) // 2) for r in day_runs]


def take(pool_runs: Sequence[Run], seconds: float) -> list[Run]:
    """First `seconds` of the pool, in time order."""
    need, out = int(round(seconds / BIN_S)), []
    for r in pool_runs:
        if need <= 0:
            break
        out.append(_slice(r, 0, min(need, len(r.spikes))))
        need -= len(out[-1].spikes)
    if need > 0:
        raise ValueError(f"pool has less than {seconds} s")
    return out


def day_r2(predict: Callable[[np.ndarray], np.ndarray], day_runs: Sequence[Run]) -> float:
    """Variance-weighted R² on the second half of every run of one day, pooled."""
    p, t = [], []
    for r in day_runs:
        half = len(r.spikes) // 2
        m = r.eval_mask[half:]
        p.append(predict(r.spikes)[half:][m])
        t.append(r.velocity[half:][m])
    return float(r2_score(np.concatenate(t), np.concatenate(p), multioutput="variance_weighted"))


def adapt(method: str, base: gru.GRUNet, data: Sequence[Run], seed: int):
    """Return a predict function for one (method, budget) cell."""
    if method == "Wiener refit":
        return WienerFilter(alpha=None).fit(data).predict  # ridge strength picked inside the budget
    if method == "scratch":
        return gru.fit_gru(data, seed=seed, steps=SCRATCH_STEPS).predict
    net = gru.clone(base)
    if method.startswith("renorm"):
        net.set_input_stats(np.concatenate([r.spikes for r in data]))
        if method == "renorm":
            return net.predict
    params = {"input layer": net.inp.parameters(), "readout": net.out.parameters(),
              "renorm + readout": net.out.parameters(), "full": net.parameters()}[method]
    return gru.train(net, data, params=params, steps=ADAPT_STEPS, lr=1e-3, seed=seed).predict


def recalibration_curve(
    bases: Sequence[gru.GRUNet],
    held_out_runs: Sequence[Run],
    budgets_s: Sequence[float] = BUDGETS_S,
    methods: Sequence[str] = METHODS,
    wiener: WienerFilter | None = None,
    existing: Sequence[dict] = (),
    on_row: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """One row per (seed, day, method, budget): {"seed", "day", "method", "budget_s", "r2"}.

    `bases` holds one pretrained GRU per seed; `wiener` is the held-in
    Wiener filter used for its zero-shot row. Cells already in `existing`
    are kept and not recomputed; `on_row` gets all rows after each new one
    (used to checkpoint).
    """
    rows = list(existing)
    done = {(r["seed"], r["day"], r["method"], r["budget_s"]) for r in rows}

    def add(seed, day, method, budget, predict):
        if (seed, day, method, budget) in done:
            return
        r2 = day_r2(predict(), runs)
        rows.append(dict(seed=seed, day=day, method=method, budget_s=budget, r2=r2))
        done.add((seed, day, method, budget))
        print(f"seed {seed} {day} {budget:>4}s {method:<16} R² {r2:+.3f}", flush=True)
        if on_row:
            on_row(rows)

    for seed, base in enumerate(bases):
        for day, runs in by_day(held_out_runs).items():
            add(seed, day, "zero-shot", 0.0, lambda: base.predict)
            if wiener is not None and seed == 0:
                add(seed, day, "Wiener zero-shot", 0.0, lambda: wiener.predict)
            for b in budgets_s:
                data = take(pool(runs), b)
                for m in methods:
                    if m == "Wiener refit" and seed > 0:
                        continue  # deterministic; one seed is enough
                    add(seed, day, m, b, lambda: adapt(m, base, data, seed))
    return rows


def summarise(rows: Sequence[dict]) -> dict:
    """Mean over days, then mean ± std over seeds, per (method, budget)."""
    out: dict = {}
    keys = sorted({(r["method"], r["budget_s"]) for r in rows}, key=lambda k: (k[0], k[1]))
    for m, b in keys:
        per_seed = {}
        for r in rows:
            if r["method"] == m and r["budget_s"] == b:
                per_seed.setdefault(r["seed"], []).append(r["r2"])
        vals = [np.mean(v) for v in per_seed.values()]
        out.setdefault(m, {})[str(b)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "seeds": len(vals)}
    return out


def chart(summary: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe7"
    # Fixed categorical order from the reference palette; colour follows the method.
    colours = {"renorm + readout": "#008300", "renorm": "#eda100", "readout": "#1baf7a", "input layer": "#eb6834",
               "full": "#2a78d6", "Wiener refit": "#4a3aa7", "scratch": "#e87ba4"}  # legend order ≈ top-to-bottom at 20 s
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    zs = summary["zero-shot"]["0.0"]["mean"]
    ax.axhline(zs, color=muted, lw=1, ls="--")
    edge = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    ax.text(0.01, zs, f"GRU zero-shot {zs:.2f}", color=muted, fontsize=8, va="bottom", transform=edge)
    if "Wiener zero-shot" in summary:
        wz = summary["Wiener zero-shot"]["0.0"]["mean"]
        ax.axhline(wz, color=muted, lw=1, ls=":")
        ax.text(0.01, wz, f"Wiener zero-shot {wz:.2f}", color=muted, fontsize=8, va="top", transform=edge)
    for m, c in colours.items():
        if m not in summary:
            continue
        b = sorted(float(k) for k in summary[m])
        y = np.array([summary[m][str(k)]["mean"] for k in b])
        s = np.array([summary[m][str(k)]["std"] for k in b])
        ax.plot(b, y, color=c, lw=2, marker="o", ms=5, label=m, zorder=3)
        if s.any():
            ax.fill_between(b, y - s, y + s, color=c, alpha=0.12, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS_S, [f"{b:g}" for b in BUDGETS_S])
    ax.set_xlabel("Seconds of data from the new day (renorm uses spikes only)", color=muted, fontsize=9)
    ax.set_ylabel("R² on the day's held-back half", color=muted, fontsize=9)
    ax.set_title("Recalibration curve, mean of 4 held-out days", color=ink, fontsize=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d0cfca")
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Recalibration curve on held-out M2 days.")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--chart-only", action="store_true", help="redraw the chart from results/recalibration.json")
    p.add_argument("--fresh", action="store_true", help="recompute every cell instead of reusing saved ones")
    args = p.parse_args(argv)
    if args.chart_only:
        result = json.loads((RESULTS / "recalibration.json").read_text())
        chart(result["summary"], RESULTS / "Recalibration Curve.png")
        return result
    path = RESULTS / "recalibration.json"
    existing = [] if args.fresh or not path.exists() else json.loads(path.read_text())["rows"]
    RESULTS.mkdir(exist_ok=True)

    def save(rows):
        result = {"protocol": "per day: first half of each run = adaptation pool, second half = test; "
                              "budget = first B seconds of the pool; R² variance-weighted per day, mean over days",
                  "budgets_s": list(BUDGETS_S), "summary": summarise(rows), "rows": rows}
        path.write_text(json.dumps(result, indent=2))
        return result

    bases = gru.pretrained(args.seeds)
    rows = recalibration_curve(bases, sessions.load_split("held-out-calib"),
                               wiener=WienerFilter().fit(sessions.load_split("held-in-calib")),
                               existing=existing, on_row=save)
    result = save(rows)
    chart(result["summary"], RESULTS / "Recalibration Curve.png")
    return result


if __name__ == "__main__":
    main()
