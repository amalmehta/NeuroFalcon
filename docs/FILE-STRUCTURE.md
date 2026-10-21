# What's Where

```
Neuro_falcon/
├── README.md                 one-glance overview, chart, links
├── pyproject.toml            dependencies (FALCON harness pinned to a commit)
├── neurofalcon/
│   ├── download.py           DANDI → slim NWB files in data/ (drops raw voltage)
│   ├── sessions.py           find and load runs via the harness's load_nwb
│   ├── wiener.py             Wiener filter (fit + streaming harness decoder)
│   ├── scoring.py            harness-faithful R²: held-out calib + official minival
│   ├── baseline.py           day 1 entry point: fit, score, write results/
│   ├── gru.py                day 2: GRU model, training, streaming decoder, save/load
│   ├── recalibration.py      day 2: recalibration curve (budgets × adaptation methods)
│   ├── ablation.py           day 3: ablation table from the recalibration results
│   └── submission.py         builds submission/decoder.pt (one network per held-out day)
├── tests/
│   ├── test_wiener.py        lag matrix, causality, streaming == batch, recovers a known map
│   ├── test_gru.py           streaming == batch, reset, per-day network choice, save/load
│   ├── test_recalibration.py adaptation/test split is disjoint, budgets, day grouping
│   └── test_data.py          slim files present, self-contained, load correctly
├── models/
│   └── GRU held-in seed {0..4}.pt    pretrained GRUs (93 KB each)
├── submission/
│   ├── Dockerfile            EvalAI image (follows the harness template)
│   ├── decode.py             runfile the image executes
│   └── decoder.pt            built by neurofalcon.submission
├── results/
│   ├── baseline.json         day 1: Wiener filter numbers
│   ├── Zero-Shot Wiener by Day.png
│   ├── gru.json              day 2: GRU numbers, per seed
│   ├── recalibration.json    day 2: every (seed, day, method, budget) R²
│   ├── Recalibration Curve.png       README chart
│   └── Ablation.md           day 3: ablation table
│   └── harness/              harness payload pickles (git-ignored)
├── data/                     downloaded NWB files (git-ignored)
│   └── 000953/sub-MonkeyN-{held-in-calib,held-out-calib,held-in-minival}/
└── docs/
    ├── Writeup.md            one-page writeup of the study
    ├── INSTRUCTIONS.md       setup, run, use
    ├── SYSTEM-DESIGN.md      architecture, flows, decisions, limits
    └── FILE-STRUCTURE.md     this file
```
