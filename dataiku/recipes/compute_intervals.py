# Dataiku Python recipe: compute_intervals
#   input : gold_features
#   output: quantile_predictions, interval_scores
# P10/P50/P90 with quantile LightGBM, then a per-meter conformal correction calibrated on
# the 4 weeks before each month, so the P10-P90 band really contains ~80% of outcomes.
import pandas as pd

import dataiku
from energyfc import backtest

vars_ = dataiku.get_custom_variables()
gold = dataiku.Dataset("gold_features").get_dataframe()
gold["timestamp"] = pd.to_datetime(gold["timestamp"])
gold["is_imputed"] = gold["is_imputed"].astype(str).str.lower().isin(["true", "1"])

qpred = backtest.quantile_backtest(gold, test_start=vars_.get("test_start", "2017-01-01"),
                                   n_folds=int(vars_.get("n_folds", 12)))
dataiku.Dataset("quantile_predictions").write_with_schema(qpred)
dataiku.Dataset("interval_scores").write_with_schema(backtest.score_intervals(qpred))
