# Dataiku Python recipe: compute_dm_tests
#   input : predictions_all
#   output: dm_tests
# Is LightGBM *significantly* better than each baseline, meter by meter?
# Diebold-Mariano on daily losses + Benjamini-Hochberg across meters.
import pandas as pd

import dataiku
from energyfc import backtest
from energyfc.stats_tests import benjamini_hochberg, diebold_mariano

COMPARISONS = [("lgbm", "snaive_168"), ("lgbm", "naive_24"), ("lgbm", "lgbm_no_weather"),
               ("gru", "lgbm"), ("ens_mean", "lgbm"), ("ens_mean", "gru"), ("ens_select", "lgbm")]

pred = dataiku.Dataset("predictions_all").get_dataframe()
pred["scored"] = pred["scored"].astype(str).str.lower().isin(["true", "1"])
losses = backtest.daily_losses(pred)

out = []
for a, b in COMPARISONS:
    rows = [{"building_id": m, "model_a": a, "model_b": b, **diebold_mariano(g[a].to_numpy(), g[b].to_numpy())}
            for m, g in losses.groupby(level="building_id")]
    df = pd.DataFrame(rows)
    df["significant_fdr5"] = benjamini_hochberg(df.p_value.to_numpy(), 0.05)
    df["verdict"] = df.apply(lambda r: ("A better" if r.dm_stat < 0 else "B better")
                             if r.significant_fdr5 else "no significant difference", axis=1)
    out.append(df)

dataiku.Dataset("dm_tests").write_with_schema(pd.concat(out, ignore_index=True))
