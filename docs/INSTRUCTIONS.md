# Setup, Run and Use

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11. Python 3.11 is pinned because torch 2.2.2, the last Intel-Mac wheel the harness can use, needs numpy 1.x.

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
```

## Get the data (about 2 minutes, 21 MB)

```bash
.venv/bin/python -m neurofalcon.download
```

This streams all 20 M2 files from DANDI and keeps everything except the raw 30 kHz voltage, which the harness never reads. It skips files you already have, so it's safe to re-run. Use `--splits held-out-calib` to fetch one split only.

## Run the day-1 baseline (about 2 minutes)

```bash
.venv/bin/python -m neurofalcon.baseline
```

It fits the Wiener filter on the 7 held-in runs and scores it twice:

- **minival**: the harness's official local evaluation (`FalconEvaluator.evaluate`). Held-in days only, and each file is about 3 s, so the number is noisy.
- **zero-shot held-out**: the 6 labelled held-out calibration runs, streamed through the harness and scored with its R² metric. This is the local stand-in for the leaderboard's held-out number.

The output goes to `results/baseline.json` and `results/Zero-Shot Wiener by Day.png`. Use `--history N` to change the lag window.

## Test

```bash
.venv/bin/python -m pytest -q
```

The data tests skip until the download has run.

## Day 2: GRU decoder (about 5 minutes)

```bash
.venv/bin/python -m neurofalcon.gru
```

This pretrains 3 seeds on the held-in runs, saves them to `models/`, and scores each one the same way as the baseline. The output goes to `results/gru.json`. Saved models are reused; add `--retrain` to train again.

## Day 2: Recalibration curve (about 2 hours on a busy laptop CPU)

```bash
.venv/bin/python -m neurofalcon.recalibration
```

For each held-out day it adapts the pretrained GRUs on 2.5, 5, 10 and 20 s of that day's labelled data and scores R² on the half of each run it never adapted on. The output goes to `results/recalibration.json` and `results/Recalibration Curve.png`.

## Day 3: Ablation (seconds)

```bash
.venv/bin/python -m neurofalcon.ablation
```

This turns the recalibration results into a table of which part of the model has to change: nothing, input statistics only, input layer, readout, all weights, or no pretraining. The output goes to `results/Ablation.md`.

## Score any decoder

```python
from neurofalcon import scoring, sessions
scoring.score(my_decoder, sessions.files("held-out-calib"))   # zero-shot, harness metric
scoring.minival(my_decoder)                                     # official local evaluation
```

`my_decoder` is anything that implements the harness's `BCIDecoder` (see `wiener.WienerDecoder` and `gru.GRUDecoder`).

## Leaderboard submission

1. Build the model file. It contains one network per held-out day, recalibrated on that day's calibration data:
   ```bash
   .venv/bin/python -m neurofalcon.submission --method renorm
   ```
2. Check it locally through the harness, outside Docker:
   ```bash
   EVAL_DATA_PATH=data .venv/bin/python submission/decode.py --evaluation local --phase minival
   ```
3. Build and push the image. This needs Docker running and about 10 GB free, for the PyTorch base image:
   ```bash
   docker build -f submission/Dockerfile -t neurofalcon-m2 .
   ```
   Then follow the EvalAI challenge page's "Submit" tab (`evalai push neurofalcon-m2:latest --phase <phase-name>`), using your own account.
