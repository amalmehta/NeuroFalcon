"""Your parts are stubs until you write them; this pins that they stay unwritten by the plumbing."""
import pytest

from neurofalcon import ablation, gru, recalibration


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        gru.fit_gru([])
    with pytest.raises(NotImplementedError):
        recalibration.recalibration_curve(None, [], [], None)
    with pytest.raises(NotImplementedError):
        ablation.run_ablation()
