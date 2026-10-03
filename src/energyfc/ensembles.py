"""Combining forecasters without peeking at the test month.

- mean ensemble: plain average of the members (no fitted weights, so nothing to leak);
- online per-meter selection: for test month k, each meter uses whichever candidate had
  the lower MAE on that meter over months < k. Month 1 falls back to the default model.
  This is the champion/challenger rule a monthly scenario would apply in production.
"""
import pandas as pd

KEY = ["building_id", "timestamp"]


def _template(pred, model):
    return pred[pred.model == model].drop(columns=["yhat", "kwh_hat"]).set_index(KEY)


def _append(pred, rows, name):
    rows = rows.reset_index()
    rows["model"] = name
    rows["kwh_hat"] = rows["yhat"] * rows["scale"]
    return pd.concat([pred, rows[pred.columns]], ignore_index=True)


def mean_ensemble(pred: pd.DataFrame, members=("lgbm", "gru"), name="ens_mean"):
    yhat = pd.concat([pred[pred.model == m].set_index(KEY).yhat.rename(m) for m in members], axis=1)
    rows = _template(pred, members[0])
    rows["yhat"] = yhat.mean(axis=1, skipna=False)
    return _append(pred, rows, name)


def online_selection(pred: pd.DataFrame, candidates=("lgbm", "gru"), default="lgbm",
                     name="ens_select"):
    p = pred[pred.model.isin(candidates) & pred.scored]
    fold_mae = (p.assign(err=(p.kwh - p.kwh_hat).abs())
                .groupby(["building_id", "fold", "model"]).err.agg(["sum", "count"]))
    folds = sorted(pred.fold.unique())
    picks = []
    for i, fold in enumerate(folds):
        past = fold_mae[fold_mae.index.get_level_values("fold").isin(folds[:i])]
        if past.empty:
            chosen = pd.Series(default, index=pred.building_id.unique())
        else:
            agg = past.groupby(["building_id", "model"]).sum()
            mae = (agg["sum"] / agg["count"]).unstack("model")
            chosen = mae.idxmin(axis=1).reindex(pred.building_id.unique()).fillna(default)
        picks.append(pd.DataFrame({"building_id": chosen.index, "fold": fold, "chosen": chosen.values}))
    picks = pd.concat(picks, ignore_index=True)

    cand = pred[pred.model.isin(candidates)].merge(picks, on=["building_id", "fold"])
    rows = cand[cand.model == cand.chosen].drop(columns=["model", "chosen", "kwh_hat"]).set_index(KEY)
    return _append(pred, rows, name), picks
