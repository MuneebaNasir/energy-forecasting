"""Backtest, statistical comparison, anomaly detection, energy signatures and figures
on the gold table produced by run_pipeline_local.py (or exported from Databricks).

    python scripts/run_evaluation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from energyfc import anomalies, backtest, ensembles, plots, signature  # noqa: E402
from energyfc.stats_tests import benjamini_hochberg, diebold_mariano  # noqa: E402

DATA, OUT, FIG = ROOT / "data/local", ROOT / "data/local/outputs", ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

gold = pd.read_parquet(DATA / "gold_features")
silver = pd.read_parquet(DATA / "silver_meters")
print(f"gold: {len(gold):,} rows, {gold.building_id.nunique()} meters")

# 1. Energy signatures (weather sensitivity per building)
sig, daily = signature.energy_signatures(gold)
sig.to_csv(OUT / "energy_signatures.csv", index=False)

# 2. Rolling-origin backtest, 12 monthly folds over 2017.
# "lgbm_gated" keeps weather features only for meters whose 2016 energy signature is
# weather-driven (decided on history only, so no test information leaks in).
sig_2016, _ = signature.energy_signatures(gold[gold.timestamp < "2017-01-01"])
sensitive = sig_2016[sig_2016.r2 >= 0.5].building_id.tolist()
print(f"weather-sensitive meters (2016 signature): {len(sensitive)}")
gold_gated = backtest.add_gated_weather(gold, sensitive)
pred, imp = backtest.rolling_origin_backtest(
    gold_gated, models={**backtest.MODELS, "lgbm_gated": backtest.GATED_FEATURES})
# Deep learning challenger (cached: ~30 min on a laptop CPU). Skip with --no-gru.
if "--no-gru" not in sys.argv:
    from energyfc import deep

    gru_path = OUT / "gru_predictions.parquet"
    if gru_path.exists() and "--refit-gru" not in sys.argv:
        gru = pd.read_parquet(gru_path)
    else:
        gru = deep.rolling_origin_backtest_gru(gold)
        gru.to_parquet(gru_path, index=False)
    pred = deep.add_to_predictions(pred, gru)
    pred = ensembles.mean_ensemble(pred, ("lgbm", "gru"))
    pred, picks = ensembles.online_selection(pred, ("lgbm", "gru"))
    picks.to_csv(OUT / "champion_per_meter.csv", index=False)
pred.to_parquet(OUT / "predictions.parquet", index=False)
imp.to_csv(OUT / "feature_importance.csv", index=False)
scales = backtest.mase_scales(gold)
per_meter, overall = backtest.score(pred, scales)
per_meter.to_csv(OUT / "metrics_per_meter.csv", index=False)
overall.to_csv(OUT / "metrics_overall.csv")
print(overall.round(3).to_string())

# 3. Diebold-Mariano per meter, Benjamini-Hochberg across meters
losses = backtest.daily_losses(pred)
comparisons = [("lgbm", "snaive_168"), ("lgbm", "naive_24"), ("lgbm", "lgbm_no_weather")]
comparisons += [("lgbm_gated", "lgbm_no_weather"), ("lgbm_gated", "lgbm")]
if "gru" in losses.columns:
    comparisons += [("gru", "lgbm"), ("gru", "snaive_168"),
                    ("ens_mean", "lgbm"), ("ens_mean", "gru"), ("ens_select", "lgbm")]
dm_rows = []
for a, b in comparisons:
    rows = []
    for meter, g in losses.groupby(level="building_id"):
        r = diebold_mariano(g[a].to_numpy(), g[b].to_numpy())
        rows.append({"building_id": meter, "model_a": a, "model_b": b, **r})
    df = pd.DataFrame(rows)
    df["significant_fdr5"] = benjamini_hochberg(df.p_value.to_numpy(), 0.05)
    dm_rows.append(df)
dm = pd.concat(dm_rows, ignore_index=True)
dm.to_csv(OUT / "diebold_mariano.csv", index=False)
dm_summary = dm.groupby(["model_a", "model_b"]).apply(lambda d: pd.Series({
    "meters": len(d),
    "a_better_signif": int((d.significant_fdr5 & (d.dm_stat < 0)).sum()),
    "b_better_signif": int((d.significant_fdr5 & (d.dm_stat > 0)).sum()),
    "no_signif_diff": int((~d.significant_fdr5).sum()),
}), include_groups=False).reset_index()
dm_summary.to_csv(OUT / "diebold_mariano_summary.csv", index=False)
print(dm_summary.to_string())

# 3b. Probabilistic forecasts: P10/P50/P90, raw vs conformally calibrated
qpred = backtest.quantile_backtest(gold)
qpred.to_parquet(OUT / "quantile_predictions.parquet", index=False)
intervals = backtest.score_intervals(qpred)
intervals.to_csv(OUT / "interval_scores.csv", index=False)
qs = qpred[qpred.scored]
from energyfc import metrics  # noqa: E402

interval_summary = {
    "coverage_raw": metrics.interval_coverage(qs.kwh, qs.q10, qs.q90),
    "coverage_calibrated": metrics.interval_coverage(qs.kwh, qs.q10_cal, qs.q90_cal),
    **{f"coverage_calibrated_{site}": metrics.interval_coverage(g.kwh, g.q10_cal, g.q90_cal)
       for site, g in qs.groupby("site_id")},
}
print({k: round(v, 3) for k, v in interval_summary.items()})

# 4. Anomalies: consumption (residual-based) vs meter faults (silver flags)
events, scored = anomalies.consumption_anomalies(pred)
events.to_csv(OUT / "consumption_anomalies.csv", index=False)
faults = anomalies.meter_faults(silver)
faults.to_csv(OUT / "meter_faults.csv", index=False)
print(events.groupby(["scope", "priority"]).size().rename("events").to_string())
print(f"meter fault events: {len(faults)}")

# 5. Figures
kept = set(gold.building_id)
worst_fault = faults[faults.building_id.isin(kept)].building_id.value_counts().index[0]
plots.quality_example(silver[silver.building_id == worst_fault], FIG / "01_meter_faults.png")
heating = sig[(sig.heating_kwh_per_degday > 0) & (sig.r2 > 0.6)]
typical = sig.iloc[(sig.r2 - sig.r2.median()).abs().argsort()[:1]]
top_sig = heating.head(1).building_id.tolist() + typical.building_id.tolist()
plots.signatures(daily, sig, top_sig, FIG / "02_energy_signature.png")
best = per_meter[per_meter.model == "lgbm"].sort_values("mase").building_id.iloc[len(kept) // 2]
plots.forecast_week(pred, best, "2017-03-06", FIG / "03_forecast_week.png")
plots.mase_by_model(per_meter, FIG / "04_mase_by_model.png")
plots.importance(imp, FIG / "05_feature_importance.png")
plots.anomaly_example(scored, events[events.priority].iloc[0], FIG / "06_anomaly_example.png")
plots.prediction_interval(qpred, best, "2017-03-06", FIG / "07_prediction_interval.png")

# 6. Headline numbers for the README / French summary
ov = overall.reset_index().set_index("model")


def mae_cut_pct(model, reference):
    """% reduction in total MAE of `model` vs `reference`."""
    return round(100 * (1 - ov.loc[model, "total_mae_kwh"] / ov.loc[reference, "total_mae_kwh"]), 1)


w = dm_summary.set_index(["model_a", "model_b"])
headline = {
    "meters_modelled": len(kept),
    "meters_total": silver.building_id.nunique(),
    "share_flagged_pct": round(100 * (silver.quality_flag != "ok").mean(), 1),
    "median_mase_lgbm": round(ov.loc["lgbm", "median_mase"], 3),
    "median_mase_snaive": round(ov.loc["snaive_168", "median_mase"], 3),
    "median_smape_lgbm": round(ov.loc["lgbm", "median_smape_pct"], 1),
    "median_smape_snaive": round(ov.loc["snaive_168", "median_smape_pct"], 1),
    "mae_reduction_vs_snaive_pct": mae_cut_pct("lgbm", "snaive_168"),
    "mae_reduction_from_weather_pct": mae_cut_pct("lgbm", "lgbm_no_weather"),
    "dm_lgbm_beats_snaive": int(w.loc[("lgbm", "snaive_168"), "a_better_signif"]),
    "dm_lgbm_beats_noweather": int(w.loc[("lgbm", "lgbm_no_weather"), "a_better_signif"]),
    "dm_noweather_beats_lgbm": int(w.loc[("lgbm", "lgbm_no_weather"), "b_better_signif"]),
    **({"median_mase_gru": round(ov.loc["gru", "median_mase"], 3),
        "mae_reduction_gru_vs_lgbm_pct": mae_cut_pct("gru", "lgbm"),
        "dm_gru_beats_lgbm": int(w.loc[("gru", "lgbm"), "a_better_signif"]),
        "dm_lgbm_beats_gru": int(w.loc[("gru", "lgbm"), "b_better_signif"])} if "gru" in ov.index else {}),
    **({"median_mase_ens_mean": round(ov.loc["ens_mean", "median_mase"], 3),
        "median_mase_ens_select": round(ov.loc["ens_select", "median_mase"], 3),
        "mae_reduction_ens_mean_vs_lgbm_pct": mae_cut_pct("ens_mean", "lgbm"),
        "dm_ens_mean_beats_lgbm": int(w.loc[("ens_mean", "lgbm"), "a_better_signif"]),
        "dm_lgbm_beats_ens_mean": int(w.loc[("ens_mean", "lgbm"), "b_better_signif"]),
        "dm_ens_mean_beats_gru": int(w.loc[("ens_mean", "gru"), "a_better_signif"]),
        "dm_gru_beats_ens_mean": int(w.loc[("ens_mean", "gru"), "b_better_signif"])} if "ens_mean" in ov.index else {}),
    "n_weather_sensitive_2016": len(sensitive),
    "median_mase_lgbm_gated": round(ov.loc["lgbm_gated", "median_mase"], 3),
    "dm_gated_beats_noweather": int(w.loc[("lgbm_gated", "lgbm_no_weather"), "a_better_signif"]),
    "dm_noweather_beats_gated": int(w.loc[("lgbm_gated", "lgbm_no_weather"), "b_better_signif"]),
    **{k: round(v, 3) for k, v in interval_summary.items()},
    "n_anomaly_events": len(events),
    "n_site_wide_events": int((events.scope == "site_wide").sum()),
    "n_priority_events": int(events.priority.sum()),
    "n_meters_with_priority": int(events[events.priority].building_id.nunique()),
    "n_fault_events": len(faults),
    "median_balance_temp_c": round(sig[(sig.r2 > 0.5) & (sig.heating_kwh_per_degday > 0)].balance_temp_c.median(), 1),
    "n_weather_sensitive": int((sig.r2 > 0.5).sum()),
    "n_heating_driven": int(((sig.r2 > 0.5) & (sig.heating_kwh_per_degday > 0)).sum()),
    "median_r2_signature": round(sig.r2.median(), 2),
}
pd.Series(headline).to_json(OUT / "headline.json", indent=2)
print(pd.Series(headline).to_string())
