# Ablation: What Recalibration Updates

R² on each held-out day's held-back half, mean over the 4 days; ± is the std over GRU seeds. Columns are labelled seconds from the new day.

| Variant | What | 2.5s | 5s | 10s | 20s |
|---|---|---|---|---|---|
| zero-shot | pretrained GRU, nothing changes | +0.18 ± 0.02 | +0.18 ± 0.02 | +0.18 ± 0.02 | +0.18 ± 0.02 |
| Wiener zero-shot | held-in Wiener filter, nothing changes | +0.01 | +0.01 | +0.01 | +0.01 |
| renorm | re-z-score inputs with the new day's spikes (no labels) | +0.28 ± 0.01 | +0.27 ± 0.02 | +0.29 ± 0.02 | +0.31 ± 0.01 |
| renorm + readout | renorm, then fine-tune the readout | +0.22 ± 0.01 | +0.20 ± 0.02 | +0.29 ± 0.01 | +0.31 ± 0.01 |
| input layer | fine-tune the 96→32 input layer only | +0.14 ± 0.02 | +0.22 ± 0.02 | +0.23 ± 0.02 | +0.25 ± 0.02 |
| readout | fine-tune the 64→2 readout only | +0.14 ± 0.04 | +0.20 ± 0.03 | +0.23 ± 0.02 | +0.26 ± 0.02 |
| full | fine-tune every weight | +0.10 ± 0.04 | +0.22 ± 0.02 | +0.27 ± 0.02 | +0.23 ± 0.02 |
| scratch | fresh GRU trained on the budget only | -0.21 ± 0.01 | -0.10 ± 0.03 | -0.04 ± 0.04 | +0.05 ± 0.03 |
| Wiener refit | fresh Wiener filter fit on the budget only | -0.13 | -0.02 | -0.06 | +0.13 |
