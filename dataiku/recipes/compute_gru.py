# Dataiku Python recipe: compute_gru
#   input : gold_features
#   output: gru_predictions
# GRU sequence-to-24h challenger, same monthly rolling-origin protocol as compute_backtest.
# Needs a code env with PyTorch (~30 min on a laptop CPU for 12 folds).
import pandas as pd

import dataiku
from energyfc import deep

vars_ = dataiku.get_custom_variables()
gold = dataiku.Dataset("gold_features").get_dataframe()
gold["timestamp"] = pd.to_datetime(gold["timestamp"])
gold["is_imputed"] = gold["is_imputed"].astype(str).str.lower().isin(["true", "1"])

gru = deep.rolling_origin_backtest_gru(gold, test_start=vars_.get("test_start", "2017-01-01"),
                                       n_folds=int(vars_.get("n_folds", 12)))
dataiku.Dataset("gru_predictions").write_with_schema(gru)
