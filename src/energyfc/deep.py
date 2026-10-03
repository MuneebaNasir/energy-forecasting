"""Deep learning challenger: a GRU sequence-to-24h model (PyTorch).

Same day-ahead protocol as the LightGBM backtest. For each (meter, day D) it sees
the meter's last 7 days of hourly load (+ a missing-value channel) and the context
known for day D (hourly temperature, day of week, holiday, season, which meter),
and outputs all 24 hours of day D in one shot. One global model, retrained per fold,
early-stopped on the last 4 weeks before the fold.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

HISTORY_DAYS = 7
HIDDEN = 64
EMB = 8
BATCH = 256
MAX_EPOCHS = 40
PATIENCE = 5
VAL_DAYS = 28


class DayAheadGRU(nn.Module):
    def __init__(self, n_meters, ctx_dim):
        super().__init__()
        self.meter = nn.Embedding(n_meters, EMB)
        self.encoder = nn.GRU(input_size=2, hidden_size=HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + EMB + ctx_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 24),
        )

    def forward(self, seq, ctx, meter, last_day):
        _, h = self.encoder(seq)
        z = torch.cat([h[-1], self.meter(meter), ctx], dim=1)
        # predict the change from the last observed day's profile (residual learning)
        return last_day + self.head(z)


def build_day_tensors(gold: pd.DataFrame):
    """Reshape the (complete-grid) gold table into per-meter-day arrays."""
    g = gold.sort_values(["building_id", "timestamp"])
    meters = g.building_id.unique()
    ts = pd.DatetimeIndex(g.timestamp.iloc[: len(g) // len(meters)])
    if len(g) != len(meters) * len(ts) or ts[0].hour != 0 or len(ts) % 24:
        raise ValueError("gold must be a complete hourly grid of whole days for every meter")
    n_days = len(ts) // 24

    def grid(col, dtype=float):
        return g[col].to_numpy(dtype).reshape(len(meters), n_days, 24)

    y = grid("y")
    scored = ~np.isnan(y) & ~grid("is_imputed", bool)
    temp = np.nan_to_num((grid("air_temp") - 10.0) / 8.0)
    days = ts[::24]
    dow = np.eye(7)[days.dayofweek]
    season = np.column_stack([np.sin(2 * np.pi * days.dayofyear / 365.25),
                              np.cos(2 * np.pi * days.dayofyear / 365.25)])
    holiday = grid("is_holiday")[:, :, 0:1]
    return dict(meters=meters, days=days, y=y, scored=scored, temp=temp,
                cal=np.concatenate([dow, season], axis=1), holiday=holiday)


def _samples(t, day_idx):
    """Stack inputs for all meters x the given target days (each needs 7 days of history)."""
    m_idx, d_idx = np.meshgrid(np.arange(len(t["meters"])), day_idx, indexing="ij")
    m_idx, d_idx = m_idx.ravel(), d_idx.ravel()
    hist = np.stack([t["y"][m_idx, d_idx - k] for k in range(HISTORY_DAYS, 0, -1)], axis=1)
    hist = hist.reshape(len(m_idx), HISTORY_DAYS * 24)
    missing = np.isnan(hist)
    seq = np.stack([np.nan_to_num(hist), missing.astype(float)], axis=2)
    last_day = t["y"][m_idx, d_idx - 1]
    # fall back to the same day last week when yesterday is missing, else the 7-day mean
    week = t["y"][m_idx, d_idx - 7]
    last_day = np.where(np.isnan(last_day), week, last_day)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing week -> NaN, handled below
        fallback = np.nanmean(np.where(missing, np.nan, hist), axis=1, keepdims=True)
    last_day = np.where(np.isnan(last_day), np.nan_to_num(fallback, nan=1.0), last_day)
    ctx = np.concatenate([t["temp"][m_idx, d_idx], t["cal"][d_idx], t["holiday"][m_idx, d_idx]], axis=1)
    target = t["y"][m_idx, d_idx]
    mask = t["scored"][m_idx, d_idx]
    f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    return dict(seq=f(seq), ctx=f(ctx), meter=torch.as_tensor(m_idx), last_day=f(last_day),
                target=f(np.nan_to_num(target)), mask=f(mask), m_idx=m_idx, d_idx=d_idx)


def _masked_l1(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)


def _predict(model, s):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(s["meter"]), 4096):
            sl = slice(i, i + 4096)
            out.append(model(s["seq"][sl], s["ctx"][sl], s["meter"][sl], s["last_day"][sl]))
    return torch.cat(out).numpy()


def fit_gru(t, train_days, val_days, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = _samples(t, train_days), _samples(t, val_days)
    keep = tr["mask"].sum(1) > 0
    tr = {k: v[keep] for k, v in tr.items()}
    model = DayAheadGRU(len(t["meters"]), tr["ctx"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, bad = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(tr["meter"]))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            loss = _masked_l1(model(tr["seq"][b], tr["ctx"][b], tr["meter"][b], tr["last_day"][b]),
                              tr["target"][b], tr["mask"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = _masked_l1(torch.as_tensor(_predict(model, va)), va["target"], va["mask"]).item()
        if val < best - 1e-4:
            best, best_state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def rolling_origin_backtest_gru(gold: pd.DataFrame, test_start="2017-01-01", n_folds=12, log=print):
    """Returns (building_id, timestamp, yhat) for every test hour, model refit per month."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t = build_day_tensors(gold)
    days = t["days"]
    fold_starts = pd.date_range(test_start, periods=n_folds + 1, freq="MS")
    out = []
    for i in range(n_folds):
        lo, hi = fold_starts[i], fold_starts[i + 1]
        first_test = int(np.searchsorted(days, lo))
        last_test = int(np.searchsorted(days, hi))
        train_days = np.arange(HISTORY_DAYS, first_test - VAL_DAYS)
        val_days = np.arange(first_test - VAL_DAYS, first_test)
        model = fit_gru(t, train_days, val_days, seed=i)
        s = _samples(t, np.arange(first_test, last_test))
        yhat = _predict(model, s)
        stamps = days[s["d_idx"]].to_numpy()[:, None] + np.arange(24) * np.timedelta64(1, "h")
        out.append(pd.DataFrame({
            "building_id": np.repeat(t["meters"][s["m_idx"]], 24),
            "timestamp": stamps.ravel(),
            "yhat": yhat.ravel(),
        }))
        log(f"gru fold {lo:%Y-%m}: train days={len(train_days)}")
    return pd.concat(out, ignore_index=True)


def add_to_predictions(pred: pd.DataFrame, gru: pd.DataFrame, name="gru"):
    """Append GRU forecasts in the same long format as backtest.rolling_origin_backtest."""
    base = pred[pred.model == "lgbm"].drop(columns=["yhat", "kwh_hat"])
    g = base.merge(gru, on=["building_id", "timestamp"], how="left")
    g["model"] = name
    g["kwh_hat"] = g["yhat"] * g["scale"]
    return pd.concat([pred, g[pred.columns]], ignore_index=True)
