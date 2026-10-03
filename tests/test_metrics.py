import numpy as np
import pytest

from energyfc import metrics


def test_mae_rmse_ignore_nans():
    y = [1, 2, np.nan, 4]
    yhat = [1, 4, 3, 4]
    assert metrics.mae(y, yhat) == pytest.approx(2 / 3)
    assert metrics.rmse(y, yhat) == pytest.approx(np.sqrt(4 / 3))


def test_smape_bounds_and_zero_handling():
    assert metrics.smape([0, 0], [0, 0]) == 0
    assert metrics.smape([1], [0]) == pytest.approx(200)
    assert metrics.smape([100], [110]) == pytest.approx(100 * 20 / 210)


def test_mase_seasonal_naive_is_one_on_history():
    rng = np.random.default_rng(0)
    week = np.tile(np.sin(np.arange(168) / 10), 6) + rng.normal(0, 0.1, 168 * 6)
    scale = metrics.seasonal_naive_scale(week, 168)
    assert metrics.mase(week[168:], week[:-168], scale) == pytest.approx(1.0)
