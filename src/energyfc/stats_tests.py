"""Statistical comparison of forecasts.

Hourly errors from a day-ahead model are not independent: all 24 hours of a day
come from the same forecast run and share the same weather error. So the
Diebold-Mariano test is run on *daily* mean loss differentials (one value per
forecast origin), with a Newey-West HAC variance for the remaining day-to-day
autocorrelation and the Harvey-Leybourne-Newbold small-sample correction.
"""
import numpy as np
from scipy import stats


def newey_west_variance(d, lags):
    """Long-run variance of the mean of d (Bartlett kernel)."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    gamma0 = np.dot(d, d) / n
    lr = gamma0
    for k in range(1, lags + 1):
        gamma_k = np.dot(d[k:], d[:-k]) / n
        lr += 2 * (1 - k / (lags + 1)) * gamma_k
    return lr / n


def diebold_mariano(loss_a, loss_b, lags=None):
    """Test H0: equal expected loss. Returns dict with mean diff, statistic, p-value.

    loss_a, loss_b: per-origin losses (e.g. daily MAE) for two models on the same
    origins. Negative statistic -> model A has lower loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "mean_diff": np.nan, "dm_stat": np.nan, "p_value": np.nan}
    if lags is None:
        lags = int(np.floor(n ** (1 / 3)))
    var = newey_west_variance(d, lags)
    if var <= 0:
        return {"n": n, "mean_diff": float(d.mean()), "dm_stat": np.nan, "p_value": np.nan}
    dm = d.mean() / np.sqrt(var)
    # HLN correction for horizon h=1 at the daily level
    h = 1
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(dm), df=n - 1)
    return {"n": n, "mean_diff": float(d.mean()), "dm_stat": float(dm), "p_value": float(p)}


def benjamini_hochberg(p_values, alpha=0.05):
    """Return boolean array of rejections controlling the false discovery rate.

    Used because one DM test is run per meter: at 80+ meters, a 5% per-test
    threshold would produce ~4 false "wins" by chance alone.
    """
    p = np.asarray(p_values, float)
    reject = np.zeros(len(p), bool)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return reject
    order = np.argsort(pv)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = pv[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    r = np.zeros(m, bool)
    r[order[:k]] = True
    reject[valid] = r
    return reject
