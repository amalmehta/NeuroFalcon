"""Checks on the downloaded slim files. Skipped until the data is downloaded."""
import h5py
import pytest

from neurofalcon import sessions

try:
    ALL = {s: sessions.files(s) for s in ("held-in-calib", "held-out-calib", "held-in-minival")}
except FileNotFoundError:
    ALL = None

needs_data = pytest.mark.skipif(ALL is None, reason="run python -m neurofalcon.download first")


@needs_data
def test_every_file_is_present():
    assert [len(ALL[s]) for s in ALL] == [7, 6, 7]


@needs_data
def test_slim_files_are_self_contained():
    for paths in ALL.values():
        for p in paths:
            with h5py.File(p) as f:
                assert "fullband" not in f["acquisition"]
                ext = []
                f.visititems_links(lambda n, l: ext.append(n) if isinstance(l, h5py.ExternalLink) else None)
                assert not ext, f"{p.name} links out: {ext[:3]}"


@needs_data
def test_runs_load_with_expected_shapes():
    run = sessions.load(ALL["held-out-calib"][0])
    T = run.spikes.shape[0]
    assert run.spikes.shape == (T, 96)
    assert run.velocity.shape == (T, 2)
    assert run.eval_mask.shape == run.trial_change.shape == (T,)
    assert run.key == "Run1_20201030"
