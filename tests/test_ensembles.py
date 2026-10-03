import numpy as np
import pandas as pd
import pytest

from energyfc import ensembles, metrics


def _pred():
    ts = pd.date_range("2017-01-01", "2017-03-31 23:00", freq="h")
    rows = []
    for b, good in [("A", "lgbm"), ("B", "gru")]:
        y = np.full(len(ts), 10.0)
        for m in ["lgbm", "gru"]:
            err = 1.0 if m == good else 3.0
            rows.append(pd.DataFrame({
                "building_id": b, "site_id": "S", "timestamp": ts, "date": ts.normalize(),
                "kwh": y, "y": y, "scale": 1.0, "is_imputed": False, "fold": ts.strftime("%Y-%m"),
                "model": m, "yhat": y + err, "kwh_hat": y + err, "scored": True,
            }))
    return pd.concat(rows, ignore_index=True)


def test_mean_ensemble_averages_members():
    out = ensembles.mean_ensemble(_pred())
    e = out[out.model == "ens_mean"]
    assert len(e) == len(out[out.model == "lgbm"])
    assert np.allclose(e.yhat, 12.0)


def test_online_selection_uses_only_past_folds():
    out, picks = ensembles.online_selection(_pred())
    first = picks[picks.fold == "2017-01"].set_index("building_id").chosen
    later = picks[picks.fold == "2017-03"].set_index("building_id").chosen
    assert (first == "lgbm").all()                    # no history yet -> default
    assert later["A"] == "lgbm" and later["B"] == "gru"
    sel = out[(out.model == "ens_select") & (out.fold == "2017-03")]
    assert np.allclose(sel.yhat, 11.0)


def test_pinball_and_coverage():
    y = np.array([0.0, 10.0])
    assert metrics.pinball(y, [5.0, 5.0], 0.5) == pytest.approx(2.5)
    assert metrics.pinball([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2)
    assert metrics.pinball([10.0], [12.0], 0.9) == pytest.approx(0.1 * 2)
    assert metrics.interval_coverage([1, 5, 9], [0, 0, 0], [6, 6, 6]) == pytest.approx(2 / 3)


def test_conformal_margin_restores_coverage():
    from energyfc.backtest import conformal_margin

    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full_like(y, -0.8), np.full_like(y, 0.8)   # too narrow: ~58% coverage
    m = conformal_margin(y, lo, hi, 0.8)
    y_new = rng.normal(0, 1, 5000)
    assert m > 0
    assert metrics.interval_coverage(y_new, lo - m, hi + m) == pytest.approx(0.8, abs=0.02)
