"""Energy signature: how much a building's daily consumption responds to temperature.

Change-point model fit on working days (removes occupancy effects):
    E_day = base + slope * max(0, T_balance - T_day)
with T_balance chosen by grid search. Output per meter: baseload, heating slope,
balance-point temperature and R². This is the standard way an energy manager
reads "how weather-sensitive is this site" and what a cold snap will cost.
"""
import numpy as np
import pandas as pd

BALANCE_GRID = np.arange(8.0, 20.5, 0.5)


def daily_frame(gold: pd.DataFrame, min_hours=20):
    g = gold[gold.kwh.notna() & (gold.is_weekend == 0) & (gold.is_holiday == 0)]
    d = g.groupby(["building_id", "site_id", "date"]).agg(
        kwh_day=("kwh", "sum"), n=("kwh", "size"), temp=("air_temp", "mean"),
    ).reset_index()
    d = d[d.n >= min_hours]
    d["kwh_day"] = d.kwh_day * 24 / d.n
    return d


def fit_heating_changepoint(temp, energy):
    temp, energy = np.asarray(temp, float), np.asarray(energy, float)
    best = None
    sst = np.sum((energy - energy.mean()) ** 2)
    for tb in BALANCE_GRID:
        x = np.maximum(0.0, tb - temp)
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, energy, rcond=None)
        sse = np.sum((energy - X @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, tb, coef)
    sse, tb, (base, slope) = best
    return {"balance_temp_c": tb, "base_kwh_day": base, "heating_kwh_per_degday": slope,
            "heating_pct_per_c": 100 * slope / base if base > 0 else np.nan,
            "r2": 1 - sse / sst if sst > 0 else np.nan}


def energy_signatures(gold: pd.DataFrame):
    d = daily_frame(gold)
    rows = []
    for (b, s), g in d.groupby(["building_id", "site_id"]):
        if len(g) < 100:
            continue
        rows.append({"building_id": b, "site_id": s, "n_days": len(g),
                     **fit_heating_changepoint(g.temp, g.kwh_day)})
    return pd.DataFrame(rows).sort_values("heating_pct_per_c", ascending=False), d
