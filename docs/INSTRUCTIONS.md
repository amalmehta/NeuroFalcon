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

## Your parts

`neurofalcon/gru.py`, `recalibration.py` and `ablation.py` are contracts only; their docstrings say what the harness expects. Score your work the same way the baseline is scored:

```python
from neurofalcon import scoring, sessions
scoring.score(my_decoder, sessions.files("held-out-calib"))
scoring.minival(my_decoder)
```

## Leaderboard submission

Not set up yet. EvalAI takes a Docker image that runs the decoder against hidden eval files. See the harness's `decoder_demos/*.Dockerfile` and `docker_notes.md`.
