# Dataiku Python recipe: compute_ensemble
#   inputs : predictions, gru_predictions, gold_features
#   outputs: predictions_all, metrics_all, champion_per_meter
# Adds the GRU, the LightGBM+GRU mean ensemble and the online per-meter champion to the
# point forecasts, and re-scores every model on the same hours.
import pandas as pd

import dataiku
from energyfc import backtest, deep, ensembles

pred = dataiku.Dataset("predictions").get_dataframe()
pred["timestamp"] = pd.to_datetime(pred["timestamp"])
pred["date"] = pd.to_datetime(pred["date"])
for c in ["is_imputed", "scored"]:
    pred[c] = pred[c].astype(str).str.lower().isin(["true", "1"])
gru = dataiku.Dataset("gru_predictions").get_dataframe()
gru["timestamp"] = pd.to_datetime(gru["timestamp"])
gold = dataiku.Dataset("gold_features").get_dataframe(columns=["building_id", "timestamp", "kwh"])
gold["timestamp"] = pd.to_datetime(gold["timestamp"])

pred = deep.add_to_predictions(pred, gru)
pred = ensembles.mean_ensemble(pred, ("lgbm", "gru"))
pred, picks = ensembles.online_selection(pred, ("lgbm", "gru"))
per_meter, _ = backtest.score(pred, backtest.mase_scales(gold))

dataiku.Dataset("predictions_all").write_with_schema(pred)
dataiku.Dataset("metrics_all").write_with_schema(per_meter)
dataiku.Dataset("champion_per_meter").write_with_schema(picks)
