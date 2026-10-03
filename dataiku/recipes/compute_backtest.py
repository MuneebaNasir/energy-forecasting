# Dataiku Python recipe: compute_backtest
#   input : gold_features
#   output: predictions, metrics_per_meter, metrics_overall, feature_importance
# Code env: Python 3.10+ with lightgbm, scipy, pandas (see README "Dataiku setup").
# Library: copy src/energyfc into the project library (Libraries > python/energyfc).
import pandas as pd

import dataiku
from energyfc import backtest

vars_ = dataiku.get_custom_variables()
test_start = vars_.get("test_start", "2017-01-01")
n_folds = int(vars_.get("n_folds", 12))

gold = dataiku.Dataset("gold_features").get_dataframe()
gold["timestamp"] = pd.to_datetime(gold["timestamp"])
gold["date"] = pd.to_datetime(gold["date"])
gold["is_imputed"] = gold["is_imputed"].astype(str).str.lower().isin(["true", "1"])

pred, imp = backtest.rolling_origin_backtest(gold, test_start=test_start, n_folds=n_folds)
per_meter, overall = backtest.score(pred, backtest.mase_scales(gold, test_start))

dataiku.Dataset("predictions").write_with_schema(pred)
dataiku.Dataset("metrics_per_meter").write_with_schema(per_meter)
dataiku.Dataset("metrics_overall").write_with_schema(overall.reset_index())
dataiku.Dataset("feature_importance").write_with_schema(imp)
