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
than cross-validated. With little data (recalibration budgets of a few
seconds) that no longer holds, so `alpha=None` picks it from ALPHAS on a
time-ordered split: fit on the first 80% of the bins, score the last 20%.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from falcon_challenge.interface import BCIDecoder

from neurofalcon.sessions import CONFIG, Run

ALPHAS = np.logspace(0, 6, 13)



def lagged(z: np.ndarray, history: int) -> np.ndarray:
    """(T, C) -> (T, history*C); column block k holds z_{t-k}, zeros before t=0."""
    T, C = z.shape
    out = np.zeros((T, history * C), dtype=z.dtype)
    for k in range(history):
        out[k:, k * C:(k + 1) * C] = z[:T - k]
    return out


class WienerFilter:
    def __init__(self, history: int = 10, alpha: float | None = 100.0):
        self.history = history
        self.alpha = alpha

    def fit(self, runs: Sequence[Run]) -> "WienerFilter":
        counts = np.concatenate([r.spikes for r in runs]).astype(np.float64)
        self.mean = counts.mean(0)
        self.std = counts.std(0)
        self.std[self.std == 0] = 1.0
        X = np.concatenate([lagged(self._z(r.spikes), self.history)[r.eval_mask] for r in runs])
        y = np.concatenate([r.velocity[r.eval_mask] for r in runs])
        if self.alpha is None:
            self.alpha = self._pick_alpha(X, y)
        self.coef, self.intercept = self._ridge(X, y, self.alpha)  # (history*C, 2), (2,)
        return self

    @staticmethod
    def _ridge(X, y, alpha):
        """Closed-form ridge on centred data, so the intercept isn't penalised."""
        x_bar, y_bar = X.mean(0), y.mean(0)
        Xc = X - x_bar
        coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - y_bar))
        return coef, y_bar - x_bar @ coef

    def _pick_alpha(self, X, y) -> float:
        n = int(0.8 * len(X))
        def r2(a):
            coef, b = self._ridge(X[:n], y[:n], a)
            err = ((X[n:] @ coef + b - y[n:]) ** 2).sum()
            return 1 - err / ((y[n:] - y[n:].mean(0)) ** 2).sum()
        return float(max(ALPHAS, key=r2))

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
