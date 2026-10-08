"""Day 2: GRU decoder. YOURS TO WRITE. This file is only the contract.

What the plumbing gives you:
  - sessions.load_split("held-in-calib") -> list[Run] with
    spikes (T, 96) uint8, velocity (T, 2), trial_change (T,), eval_mask (T,)
  - scoring.score(decoder, paths) and scoring.minival(decoder), which stream
    your decoder through the FALCON harness one 20 ms bin at a time.

What the harness needs from GRUDecoder (falcon_challenge.interface.BCIDecoder):
  - predict(neural_observations: (batch, 96) counts) -> (batch, 2) velocity,
    called once per bin, in time order. Keep the hidden state between calls.
  - reset(dataset_tags): called when a new file starts. Zero the hidden state.
  - on_done(dones): called every bin for M2. Ignoring it is fine.
  - Must be causal: a prediction for bin t may only use bins <= t.
"""
from __future__ import annotations

from typing import Sequence

from falcon_challenge.interface import BCIDecoder

from neurofalcon.sessions import CONFIG, Run


class GRUDecoder(BCIDecoder):
    def __init__(self, model_path: str | None = None, batch_size: int = 1):
        super().__init__(task_config=CONFIG, batch_size=batch_size)
        raise NotImplementedError("GRU decoder: yours to write")

    def reset(self, dataset_tags=None):
        raise NotImplementedError

    def predict(self, neural_observations):
        raise NotImplementedError

    def on_done(self, dones):
        pass


def fit_gru(runs: Sequence[Run]) -> GRUDecoder:
    """Train on held-in runs, return a decoder ready for scoring.score."""
    raise NotImplementedError("GRU training: yours to write")
