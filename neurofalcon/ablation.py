"""Day 3: ablation. What has to change for a decoder to work on a new day?

    python -m neurofalcon.ablation     # reads results/recalibration.json

The ablated component is what recalibration is allowed to update: nothing,
input statistics only (no labels), the input layer, the readout, every weight,
or nothing pretrained at all (a GRU trained from scratch on the budget). The
Wiener filter refit on the same budget is the classical reference. All rows
come from one run of `python -m neurofalcon.recalibration`, so every method
sees the same days, budgets, held-back test bins and pretrained seeds.

Writes results/Ablation.md.
"""
from __future__ import annotations

import json

from neurofalcon.scoring import RESULTS

ORDER = ("zero-shot", "Wiener zero-shot", "renorm", "renorm + readout", "input layer", "readout", "full",
         "scratch", "Wiener refit")
WHAT = {
    "zero-shot": "pretrained GRU, nothing changes",
    "Wiener zero-shot": "held-in Wiener filter, nothing changes",
    "renorm": "re-z-score inputs with the new day's spikes (no labels)",
    "input layer": "fine-tune the 96→32 input layer only",
    "readout": "fine-tune the 64→2 readout only",
    "renorm + readout": "renorm, then fine-tune the readout",
    "full": "fine-tune every weight",
    "scratch": "fresh GRU trained on the budget only",
    "Wiener refit": "fresh Wiener filter fit on the budget only",
}


def run_ablation() -> list[dict]:
    """Rows of {"variant", "what", "<budget>s": "mean ± std"} from results/recalibration.json."""
    rec = json.loads((RESULTS / "recalibration.json").read_text())
    summary, budgets = rec["summary"], rec["budgets_s"]
    rows = []
    for m in ORDER:
        if m not in summary:
            continue
        row = {"variant": m, "what": WHAT[m]}
        for b in budgets:
            cell = summary[m].get(str(b)) or summary[m].get("0.0")  # zero-shot rows don't depend on budget
            row[f"{b:g}s"] = f"{cell['mean']:+.2f} ± {cell['std']:.2f}" if cell["seeds"] > 1 else f"{cell['mean']:+.2f}"
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    head = "| " + " | ".join(c if c not in ("variant", "what") else c.capitalize() for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main() -> str:
    rows = run_ablation()
    md = ("# Ablation: What Recalibration Updates\n\n"
          "R² on each held-out day's held-back half, mean over the 4 days; ± is the std over GRU seeds. "
          "Columns are labelled seconds from the new day.\n\n" + to_markdown(rows) + "\n")
    (RESULTS / "Ablation.md").write_text(md)
    print(md)
    return md


if __name__ == "__main__":
    main()
