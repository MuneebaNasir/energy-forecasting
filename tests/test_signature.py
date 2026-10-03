import numpy as np
import pytest

from energyfc.signature import fit_heating_changepoint


def test_recovers_balance_point_and_slope():
    rng = np.random.default_rng(3)
    t = rng.uniform(-5, 28, 400)
    e = 1000 + 40 * np.maximum(0, 15 - t) + rng.normal(0, 10, 400)
    r = fit_heating_changepoint(t, e)
    assert r["balance_temp_c"] == pytest.approx(15, abs=0.5)
    assert r["heating_kwh_per_degday"] == pytest.approx(40, rel=0.05)
    assert r["r2"] > 0.95
