"""Forecast accuracy metrics. Inputs are 1-D arrays; NaNs are dropped pairwise."""
import numpy as np


def _clean(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = ~(np.isnan(y) | np.isnan(yhat))
    return y[mask], yhat[mask]


def mae(y, yhat):
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    y, yhat = _clean(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y, yhat):
    """Symmetric MAPE in %, bounded [0, 200]. Points where both are 0 count as perfect."""
    y, yhat = _clean(y, yhat)
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(2 * np.abs(y - yhat), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(100 * np.mean(ratio))


def mase(y, yhat, insample_scale):
    """MAE scaled by the in-sample MAE of the seasonal naive forecast.

    insample_scale is computed once per meter on the history period (see
    seasonal_naive_scale), so MASE < 1 means "beats last week's profile".
    """
    return mae(y, yhat) / insample_scale


def seasonal_naive_scale(history, season=168):
    history = np.asarray(history, float)
    diffs = np.abs(history[season:] - history[:-season])
    return float(np.nanmean(diffs))


def pinball(y, q_pred, alpha):
    """Quantile (pinball) loss for quantile level alpha; lower is better."""
    y, q_pred = _clean(y, q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def interval_coverage(y, lo, hi):
    """Share of observations inside [lo, hi]. A calibrated P10-P90 band should give ~0.80."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    mask = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))
