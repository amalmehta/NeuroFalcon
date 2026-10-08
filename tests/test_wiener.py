import numpy as np
import pytest

from neurofalcon.sessions import Run
from neurofalcon.wiener import WienerDecoder, WienerFilter, lagged


def synthetic_run(T=3000, seed=0, history=3):
    rng = np.random.default_rng(seed)
    spikes = rng.poisson(2.0, size=(T, 96)).astype(np.uint8)
    W = rng.normal(size=(history, 96, 2)) * 0.05
    z = (spikes - 2.0) / np.sqrt(2.0)
    vel = sum(np.vstack([np.zeros((k, 96)), z[: T - k]]) @ W[k] for k in range(history))
    return Run(path=None, key="Run1_20201019", spikes=spikes, velocity=vel.astype(np.float32),
               trial_change=np.zeros(T, bool), eval_mask=np.ones(T, bool))


def test_lagged_is_causal_and_zero_padded():
    z = np.arange(12, dtype=float).reshape(6, 2)
    X = lagged(z, 3)
    assert np.array_equal(X[0], [0, 1, 0, 0, 0, 0])
    assert np.array_equal(X[4], [8, 9, 6, 7, 4, 5])


def test_recovers_a_linear_map():
    run = synthetic_run()
    wf = WienerFilter(history=3, alpha=1e-3).fit([run])
    pred = wf.predict(run.spikes)
    r2 = 1 - ((pred - run.velocity) ** 2).sum() / ((run.velocity - run.velocity.mean(0)) ** 2).sum()
    assert r2 > 0.99


def test_streaming_matches_batch():
    run = synthetic_run(T=200)
    wf = WienerFilter(history=4).fit([run])
    dec = WienerDecoder(wf, batch_size=1)
    dec.reset()
    stream = np.stack([dec.predict(s[None])[0] for s in run.spikes])
    assert np.allclose(stream, wf.predict(run.spikes))


def test_prediction_ignores_future_bins():
    run = synthetic_run(T=200)
    wf = WienerFilter(history=4).fit([run])
    changed = run.spikes.copy()
    changed[100:] = 0
    assert np.allclose(wf.predict(run.spikes)[:100], wf.predict(changed)[:100])
