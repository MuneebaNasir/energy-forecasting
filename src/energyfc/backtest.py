"""Rolling-origin backtest of day-ahead models on the gold table (pandas).

Each fold is one calendar month of the test year. Models are refit on all data
before the fold start (expanding window), mirroring a monthly retraining schedule.
Only genuinely observed targets are scored (interpolated hours are excluded).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from energyfc import metrics
from energyfc.config import CATEGORICAL, FEATURES

WEATHER_FEATURES = ["air_temp", "temp_mean_24h", "hdd", "cdd", "dew_temp", "wind_speed"]

LGBM_PARAMS = dict(
    objective="l2", learning_rate=0.05, n_estimators=600, num_leaves=63,
    min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    verbose=-1,
)


def _scorable(df):
    return df["y"].notna() & ~df["is_imputed"].astype(bool)


def _prep(df, features):
    X = df[features].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def fit_lgbm(train, features, seed=0, **overrides):
    model = lgb.LGBMRegressor(random_state=seed, **{**LGBM_PARAMS, **overrides})
    model.fit(_prep(train, features), train["y"])
    return model


NO_WEATHER = [f for f in FEATURES if f not in WEATHER_FEATURES]
GATED_FEATURES = NO_WEATHER + [f"gated_{f}" for f in WEATHER_FEATURES]

MODELS = {
    "naive_24": None,         # same hour yesterday
    "snaive_168": None,       # same hour last week
    "lgbm_no_weather": NO_WEATHER,
    "lgbm": FEATURES,
}


def add_gated_weather(gold: pd.DataFrame, sensitive_meters):
    """Weather features kept only for meters whose energy signature says weather matters.

    For the others they are set to NaN, which LightGBM routes as "unknown", so the
    trees cannot fit noise on temperature for buildings that don't respond to it.
    The sensitive set must come from history only (see signature on 2016).
    """
    gold = gold.copy()
    keep = gold.building_id.isin(set(sensitive_meters))
    for f in WEATHER_FEATURES:
        gold[f"gated_{f}"] = gold[f].where(keep)
    return gold


def rolling_origin_backtest(gold: pd.DataFrame, test_start="2017-01-01", n_folds=12,
                            models=MODELS, log=print):
    gold = gold.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    gold["primary_use"] = gold["primary_use"].astype("category")
    fold_starts = pd.date_range(test_start, periods=n_folds + 1, freq="MS")
    out, importances = [], []

    for i in range(n_folds):
        lo, hi = fold_starts[i], fold_starts[i + 1]
        train = gold[(gold.timestamp < lo) & _scorable(gold)]
        test = gold[(gold.timestamp >= lo) & (gold.timestamp < hi)]
        preds = test[["building_id", "site_id", "timestamp", "date", "kwh", "y", "scale",
                      "is_imputed"]].copy()
        preds["fold"] = lo.strftime("%Y-%m")

        for name, feats in models.items():
            if name == "naive_24":
                yhat = test["y_lag_24"].to_numpy()
            elif name == "snaive_168":
                yhat = test["y_lag_168"].to_numpy()
            else:
                model = fit_lgbm(train, feats)
                yhat = model.predict(_prep(test, feats))
                if name == "lgbm":
                    importances.append(pd.DataFrame({
                        "fold": preds["fold"].iloc[0], "feature": feats,
                        "gain": model.booster_.feature_importance("gain"),
                    }))
            p = preds.copy()
            p["model"] = name
            p["yhat"] = yhat
            out.append(p)
        log(f"fold {lo:%Y-%m}: train={len(train):,} test={len(test):,}")

    pred = pd.concat(out, ignore_index=True)
    pred["kwh_hat"] = pred["yhat"] * pred["scale"]
    pred["scored"] = pred["y"].notna() & ~pred["is_imputed"].astype(bool)
    return pred, pd.concat(importances, ignore_index=True)


QUANTILES = (0.1, 0.5, 0.9)
CALIBRATION_DAYS = 28
MIN_CAL_POINTS = 200


def conformal_margin(y, lo, hi, coverage=0.8):
    """Split-conformal (CQR) margin: how much to widen [lo, hi] so that, on the
    calibration window, the band would have covered `coverage` of the observations."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def quantile_backtest(gold: pd.DataFrame, test_start="2017-01-01", n_folds=12,
                      features=FEATURES, quantiles=QUANTILES, log=print):
    """P10/P50/P90 day-ahead forecasts with quantile LightGBM, plus a conformally
    calibrated band (q10_cal/q90_cal).

    Per fold, models are fit on data up to CALIBRATION_DAYS before the fold; the held-out
    weeks just before the fold calibrate the band; the same models then forecast the fold.
    Margins are computed per meter in normalised units (y).
    """
    gold = gold.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    gold["primary_use"] = gold["primary_use"].astype("category")
    fold_starts = pd.date_range(test_start, periods=n_folds + 1, freq="MS")
    out = []
    for i in range(n_folds):
        lo, hi = fold_starts[i], fold_starts[i + 1]
        cal_start = lo - pd.Timedelta(days=CALIBRATION_DAYS)
        train = gold[(gold.timestamp < cal_start) & _scorable(gold)]
        cal = gold[(gold.timestamp >= cal_start) & (gold.timestamp < lo) & _scorable(gold)]
        test = gold[(gold.timestamp >= lo) & (gold.timestamp < hi)]
        models = [fit_lgbm(train, features, objective="quantile", alpha=a) for a in quantiles]

        def predict(df, models=models):
            return np.sort(np.column_stack([m.predict(_prep(df, features)) for m in models]), axis=1)

        qc, qt = predict(cal), predict(test)
        # Mondrian conformal: one margin per meter (errors differ a lot between buildings),
        # falling back to the pooled margin for meters with too little calibration data.
        c = pd.DataFrame({"b": cal.building_id.to_numpy(), "y": cal["y"].to_numpy(),
                          "lo": qc[:, 0], "hi": qc[:, -1]})
        pooled = conformal_margin(c.y.to_numpy(), c.lo.to_numpy(), c.hi.to_numpy())
        per_meter = {b: conformal_margin(g.y.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy())
                     for b, g in c.groupby("b") if len(g) >= MIN_CAL_POINTS}
        margin = test.building_id.map(per_meter).fillna(pooled).to_numpy()
        scale = test["scale"].to_numpy()
        p = test[["building_id", "site_id", "timestamp", "kwh"]].copy()
        p["fold"] = lo.strftime("%Y-%m")
        p["scored"] = _scorable(test).to_numpy()
        for j, a in enumerate(quantiles):
            p[f"q{round(a * 100)}"] = qt[:, j] * scale
        p["q10_cal"] = np.maximum(qt[:, 0] - margin, 0) * scale
        p["q90_cal"] = (qt[:, -1] + margin) * scale
        p["cal_margin"] = margin
        out.append(p)
        log(f"quantile fold {lo:%Y-%m}: median conformal margin {np.median(margin):+.3f} (normalised)")
    return pd.concat(out, ignore_index=True)


def score_intervals(qpred: pd.DataFrame):
    """Calibration (coverage of the P10-P90 band) and sharpness (band width), per site and fold."""
    q = qpred[qpred.scored]
    rows = []
    for (site, fold), g in q.groupby(["site_id", "fold"]):
        rows.append({
            "site_id": site, "fold": fold,
            "coverage_raw": metrics.interval_coverage(g.kwh, g.q10, g.q90),
            "coverage_calibrated": metrics.interval_coverage(g.kwh, g.q10_cal, g.q90_cal),
            "rel_width_raw_pct": float(100 * ((g.q90 - g.q10) / g.q50.clip(lower=1e-6)).median()),
            "rel_width_calibrated_pct": float(100 * ((g.q90_cal - g.q10_cal) / g.q50.clip(lower=1e-6)).median()),
            "pinball_p10": metrics.pinball(g.kwh, g.q10, 0.1),
            "pinball_p50": metrics.pinball(g.kwh, g.q50, 0.5),
            "pinball_p90": metrics.pinball(g.kwh, g.q90, 0.9),
        })
    return pd.DataFrame(rows)


def mase_scales(gold: pd.DataFrame, test_start="2017-01-01"):
    hist = gold[gold.timestamp < test_start].sort_values(["building_id", "timestamp"])
    return hist.groupby("building_id")["kwh"].apply(
        lambda s: metrics.seasonal_naive_scale(s.to_numpy())
    ).rename("mase_scale")


def score(pred: pd.DataFrame, scales: pd.Series):
    """Per-meter and overall accuracy, in kWh, on scored hours only."""
    p = pred[pred.scored].merge(scales, left_on="building_id", right_index=True)
    rows = []
    for (b, m), g in p.groupby(["building_id", "model"]):
        rows.append({
            "building_id": b, "site_id": g.site_id.iloc[0], "model": m,
            "mae_kwh": metrics.mae(g.kwh, g.kwh_hat),
            "rmse_kwh": metrics.rmse(g.kwh, g.kwh_hat),
            "smape_pct": metrics.smape(g.kwh, g.kwh_hat),
            "mase": metrics.mase(g.kwh, g.kwh_hat, g.mase_scale.iloc[0]),
            "n_hours": len(g),
        })
    per_meter = pd.DataFrame(rows)
    overall = per_meter.groupby("model").agg(
        median_mase=("mase", "median"), mean_mase=("mase", "mean"),
        median_smape_pct=("smape_pct", "median"), total_mae_kwh=("mae_kwh", "sum"),
    ).sort_values("median_mase")
    return per_meter, overall


def daily_losses(pred: pd.DataFrame):
    """One loss per (meter, forecast origin = day, model): daily MAE in kWh."""
    p = pred[pred.scored].copy()
    p["abs_err"] = (p.kwh - p.kwh_hat).abs()
    return p.groupby(["building_id", "date", "model"])["abs_err"].mean().unstack("model")
