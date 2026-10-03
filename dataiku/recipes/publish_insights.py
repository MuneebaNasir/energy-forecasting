# Dataiku Python recipe: publish_insights
#   inputs : predictions_all, metrics_all, quantile_predictions, consumption_anomalies,
#            gold_energy_signatures, gold_features
#   output : insights_log (one row per published chart)
# Renders the project's charts with energyfc.plots and publishes them as static insights,
# which the "Energy forecasting" dashboard displays.

import dataiku.insights
import matplotlib.pyplot as plt
import pandas as pd

import dataiku
from energyfc import plots, signature


def df(name, **kw):
    d = dataiku.Dataset(name).get_dataframe(**kw)
    for c in ["timestamp", "date", "start", "end"]:
        if c in d:
            d[c] = pd.to_datetime(d[c])
    for c in ["scored", "is_imputed", "priority"]:
        if c in d:
            d[c] = d[c].astype(str).str.lower().isin(["true", "1"])
    return d


def publish(insight_id, label, fig):
    dataiku.insights.save_figure(insight_id, fig, label=label)
    plt.close(fig)
    return {"insight_id": insight_id, "label": label}


pred = df("predictions_all")
per_meter = df("metrics_all")
qpred = df("quantile_predictions")
events = df("consumption_anomalies")
gold = df("gold_features", columns=["building_id", "site_id", "date", "kwh", "air_temp",
                                    "is_weekend", "is_holiday"])
sig, daily = signature.energy_signatures(gold)
heating = sig[(sig.heating_kwh_per_degday > 0) & (sig.r2 > 0.6)].building_id.head(1).tolist()
typical = sig.iloc[(sig.r2 - sig.r2.median()).abs().argsort()[:1]].building_id.tolist()
lgbm = per_meter[per_meter.model == "lgbm"].sort_values("mase")
example = lgbm.building_id.iloc[len(lgbm) // 2]
scored = pred[(pred.model == "lgbm") & pred.scored]

log = [
    publish("mase_by_model", "Forecast error by model (MASE per meter)",
            plots.mase_by_model(per_meter)),
    publish("prediction_interval", "Calibrated P10-P90 forecast, one week",
            plots.prediction_interval(qpred, example, "2017-03-06")),
    publish("forecast_week", "Day-ahead forecast vs actual, one week",
            plots.forecast_week(pred, example, "2017-03-06")),
    publish("energy_signature", "Energy signature: consumption vs temperature",
            plots.signatures(daily, sig, heating + typical)),
    publish("top_anomaly", "Largest building-specific anomaly",
            plots.anomaly_example(scored, events[events.priority].iloc[0])),
]
dataiku.Dataset("insights_log").write_with_schema(pd.DataFrame(log))
