from dataclasses import replace

import numpy as np
import pytest

from neurofalcon import recalibration as rc
from tests.test_wiener import synthetic_run


def two_run_day():
    a = replace(synthetic_run(T=1000, seed=1), key="Run1_20201030")
    b = replace(synthetic_run(T=800, seed=2), key="Run2_20201030")
    return [a, b]


def test_budget_comes_only_from_first_halves():
    day = two_run_day()
    pool = rc.pool(day)
    assert [len(r.spikes) for r in pool] == [500, 400]
    data = rc.take(pool, 12.0)  # 600 bins: all of run 1's half + 100 of run 2's
    assert [len(r.spikes) for r in data] == [500, 100]
    assert np.array_equal(data[1].spikes, day[1].spikes[:100])


def test_budget_larger_than_pool_is_an_error():
    with pytest.raises(ValueError):
        rc.take(rc.pool(two_run_day()), 20.0)


def test_day_r2_scores_only_second_halves():
    day = two_run_day()

    def predict(spikes):
        out = np.zeros((len(spikes), 2))  # garbage in the first half
        return out

    def perfect_second_half(spikes):
        r = next(r for r in day if len(r.spikes) == len(spikes))
        out = np.zeros((len(spikes), 2))
        out[len(spikes) // 2:] = r.velocity[len(spikes) // 2:]
        return out

    assert rc.day_r2(perfect_second_half, day) == pytest.approx(1.0)
    assert rc.day_r2(predict, day) < 0.1


def test_by_day_groups_runs():
    day = two_run_day()
    other = replace(synthetic_run(T=100), key="Run1_20201118")
    assert list(rc.by_day([other, *day])) == ["20201030", "20201118"]
