"""Zero-shot Wiener filter baseline.

A Wiener filter here is linear regression from the last `history` bins of
spike counts to finger velocity:

    v_t = b + sum_{k=0}^{history-1} W_k s_{t-k}

Counts are z-scored with held-in statistics, lags are zero-padded at the
start of each run (zero = the held-in mean), and the weights are fit with
ridge regression on the bins the harness scores (eval_mask). "Zero-shot"
means nothing is refit or renormalised on held-out days.

Defaults: history=10 bins (200 ms). On a within-run 80/20 split of held-in
data, R² was 0.10 / 0.26 / 0.30 / 0.32 for 1 / 5 / 10 / 20 bins, and ridge
strength from 1 to 1e4 changed it by under 0.002, so alpha is fixed rather
than cross-validated.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from falcon_challenge.interface import BCIDecoder

from neurofalcon.sessions import CONFIG, Run



def lagged(z: np.ndarray, history: int) -> np.ndarray:
    """(T, C) -> (T, history*C); column block k holds z_{t-k}, zeros before t=0."""
    T, C = z.shape
    out = np.zeros((T, history * C), dtype=z.dtype)
    for k in range(history):
        out[k:, k * C:(k + 1) * C] = z[:T - k]
    return out


class WienerFilter:
    def __init__(self, history: int = 10, alpha: float = 100.0):
        self.history = history
        self.alpha = alpha

    def fit(self, runs: Sequence[Run]) -> "WienerFilter":
        counts = np.concatenate([r.spikes for r in runs]).astype(np.float64)
        self.mean = counts.mean(0)
        self.std = counts.std(0)
        self.std[self.std == 0] = 1.0
        X = np.concatenate([lagged(self._z(r.spikes), self.history)[r.eval_mask] for r in runs])
        y = np.concatenate([r.velocity[r.eval_mask] for r in runs])
        # Closed-form ridge on centred data, so the intercept isn't penalised.
        x_bar, y_bar = X.mean(0), y.mean(0)
        Xc = X - x_bar
        self.coef = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_bar))  # (history*C, 2)
        self.intercept = y_bar - x_bar @ self.coef  # (2,)
        return self

    def _z(self, spikes: np.ndarray) -> np.ndarray:
        return (spikes - self.mean) / self.std

    def predict(self, spikes: np.ndarray) -> np.ndarray:
        """Whole-run prediction, (T, 96) -> (T, 2). Same output as the streaming decoder."""
        return lagged(self._z(spikes), self.history) @ self.coef + self.intercept


class WienerDecoder(BCIDecoder):
    """Streams one bin at a time through the FALCON harness."""

    def __init__(self, wf: WienerFilter, batch_size: int = 1):
        super().__init__(task_config=CONFIG, batch_size=batch_size)
        self.wf = wf
        self.reset()

    def reset(self, dataset_tags=None):
        # buffer[b, k] holds z_{t-k} for batch element b
        self.buffer = np.zeros((self.batch_size, self.wf.history, CONFIG.n_channels))

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        n = neural_observations.shape[0]
        self.buffer = np.roll(self.buffer, 1, axis=1)
        self.buffer[:n, 0] = self.wf._z(neural_observations)
        self.buffer[n:, 0] = 0.0
        x = self.buffer[:n].reshape(n, -1)
        return x @ self.wf.coef + self.wf.intercept

    def on_done(self, dones: np.ndarray):
        pass  # M2 is scored continuously; trial ends don't reset state
