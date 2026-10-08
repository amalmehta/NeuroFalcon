# What's Where

```
Neuro_falcon/
├── README.md                 one-glance overview, chart, links
├── Neuro_falcon.md           the project brief, deliverables, open questions, changelog
├── pyproject.toml            dependencies (FALCON harness pinned to a commit)
├── neurofalcon/
│   ├── download.py           DANDI → slim NWB files in data/ (drops raw voltage)
│   ├── sessions.py           find and load runs via the harness's load_nwb
│   ├── wiener.py             Wiener filter (fit + streaming harness decoder)
│   ├── scoring.py            harness-faithful R²: held-out calib + official minival
│   ├── baseline.py           day 1 entry point: fit, score, write results/
│   ├── gru.py                day 2, yours: GRU decoder contract (stub)
│   ├── recalibration.py      day 2, yours: recalibration curve contract (stub)
│   └── ablation.py           day 3, yours: ablation contract (stub)
├── tests/
│   ├── test_wiener.py        lag matrix, causality, streaming == batch, recovers a known map
│   ├── test_data.py          slim files present, self-contained, load correctly
│   └── test_stubs.py         your parts still raise NotImplementedError
├── results/
│   ├── baseline.json         the day 1 numbers
│   ├── Zero-Shot Wiener by Day.png   README chart
│   └── harness/              harness payload pickles (git-ignored)
├── data/                     downloaded NWB files (git-ignored)
│   └── 000953/sub-MonkeyN-{held-in-calib,held-out-calib,held-in-minival}/
└── docs/
    ├── INSTRUCTIONS.md       setup, run, use
    ├── SYSTEM-DESIGN.md      architecture, flows, decisions, limits
    └── FILE-STRUCTURE.md     this file
```
