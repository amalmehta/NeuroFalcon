from pathlib import Path

import numpy as np
import torch

from neurofalcon import gru
from tests.test_wiener import synthetic_run


def small_net(run):
    torch.manual_seed(0)
    net = gru.GRUNet()
    net.set_input_stats(run.spikes)
    net.set_output_stats(run.velocity)
    return net


def test_streaming_matches_batch():
    run = synthetic_run(T=150)
    net = small_net(run)
    dec = gru.GRUDecoder(net)
    dec.reset()
    stream = np.stack([dec.predict(s[None])[0] for s in run.spikes])
    assert np.allclose(stream, net.predict(run.spikes), atol=1e-5)


def test_reset_clears_state():
    run = synthetic_run(T=50)
    dec = gru.GRUDecoder(small_net(run))
    first = dec.predict(run.spikes[:1])
    for s in run.spikes[1:]:
        dec.predict(s[None])
    dec.reset()
    assert np.allclose(dec.predict(run.spikes[:1]), first)


def test_picks_the_network_for_each_day():
    run = synthetic_run(T=20)
    a, b = small_net(run), small_net(run)
    with torch.no_grad():
        b.out.bias += 1.0
    dec = gru.GRUDecoder({"default": a, "20201030": b})
    held_out = Path("sub-MonkeyN-held-out-calib_ses-2020-10-30-Run1_behavior+ecephys.nwb")
    held_in = Path("sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb")
    dec.reset([held_out])
    assert dec.active[0] is b
    dec.reset([held_in])
    assert dec.active[0] is a


def test_training_reduces_error():
    run = synthetic_run(T=600)
    net = small_net(run)
    before = ((net.predict(run.spikes) - run.velocity) ** 2).mean()
    gru.train(net, [run], steps=60, window=50, burn_in=5)
    after = ((net.predict(run.spikes) - run.velocity) ** 2).mean()
    assert after < 0.7 * before


def test_save_load_roundtrip(tmp_path):
    run = synthetic_run(T=30)
    net = small_net(run)
    gru.save({"default": net}, tmp_path / "m.pt")
    again = gru.load(tmp_path / "m.pt")["default"]
    assert np.allclose(again.predict(run.spikes), net.predict(run.spikes))
