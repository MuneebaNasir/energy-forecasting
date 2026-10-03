import numpy as np

from energyfc.stats_tests import benjamini_hochberg, diebold_mariano


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    loss_b = rng.gamma(2, 1, 300)
    loss_a = loss_b * 0.7
    r = diebold_mariano(loss_a, loss_b)
    assert r["dm_stat"] < 0 and r["p_value"] < 1e-6


def test_dm_size_under_null_is_close_to_nominal():
    rng = np.random.default_rng(2)
    rejections = 0
    for _ in range(400):
        # AR(1) loss differential with zero mean: autocorrelated, as daily losses are
        e = rng.normal(size=365)
        d = np.zeros(365)
        for t in range(1, 365):
            d[t] = 0.5 * d[t - 1] + e[t]
        base = rng.gamma(2, 1, 365) + 10
        rejections += diebold_mariano(base + d, base)["p_value"] < 0.05
    assert 0.02 < rejections / 400 < 0.10


def test_bh_controls_rejections():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    r = benjamini_hochberg(p, 0.05)
    assert r.tolist() == [True, True, False, False, False, False, False, False, False]
