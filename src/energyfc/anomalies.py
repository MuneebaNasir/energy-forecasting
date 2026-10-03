"""Two kinds of events an energy manager cares about, kept separate on purpose:

- meter faults: the *measurement* is wrong (stuck register, zero dropout, spike),
  detected in silver from the raw readings themselves;
- consumption anomalies: the reading is plausible but the building used far more
  or less than the model expected given weather and calendar (e.g. heating left
  on over a holiday), detected from forecast residuals.
"""
import numpy as np
import pandas as pd

RESIDUAL_Z = 5.0
MIN_EVENT_HOURS = 3
SITE_WIDE_SHARE = 0.2     # event day shared by >= 20% of a site's meters -> site-wide cause
PRIORITY_PCT_OF_DAY = 20  # building-specific events worth >= 20% of a normal day's energy


def _runs(df, flag_col, key="building_id"):
    """Group consecutive flagged hours per meter into (start, end, hours) events."""
    df = df.sort_values([key, "timestamp"])
    new_run = (df[flag_col] != df.groupby(key)[flag_col].shift()) | (
        df.timestamp - df.groupby(key).timestamp.shift() != pd.Timedelta("1h")
    )
    df = df.assign(_run=new_run.cumsum())
    return df[df[flag_col]].groupby([key, "_run"])


def consumption_anomalies(pred: pd.DataFrame, model="lgbm"):
    p = pred[(pred.model == model) & pred.scored].copy()
    p["resid_kwh"] = p.kwh - p.kwh_hat
    med = p.groupby("building_id").resid_kwh.transform("median")
    mad = p.groupby("building_id").resid_kwh.transform(lambda r: (r - r.median()).abs().median())
    p["resid_z"] = (p.resid_kwh - med) / (1.4826 * mad + 1e-9)
    p["flag"] = p.resid_z.abs() > RESIDUAL_Z

    events = _runs(p, "flag").agg(
        start=("timestamp", "min"), end=("timestamp", "max"), hours=("timestamp", "size"),
        excess_kwh=("resid_kwh", "sum"), peak_z=("resid_z", lambda z: z.iloc[np.argmax(z.abs().to_numpy())]),
        site_id=("site_id", "first"), scale=("scale", "first"),
    ).reset_index().drop(columns="_run")
    events = events[events.hours >= MIN_EVENT_HOURS].copy()
    events["event_type"] = np.where(events.excess_kwh > 0, "over_consumption", "under_consumption")
    events["excess_pct_of_day"] = 100 * events.excess_kwh / (24 * events.scale)

    # Many meters deviating on the same day points to a shared cause (snow, term
    # calendar, site event), not a building problem: report those separately.
    events["date"] = events.start.dt.normalize()
    meters_per_site = p.groupby("site_id").building_id.nunique()
    affected = events.groupby(["site_id", "date"]).building_id.nunique().rename("meters_same_day")
    events = events.join(affected, on=["site_id", "date"])
    share = events.meters_same_day / events.site_id.map(meters_per_site)
    events["scope"] = np.where(share >= SITE_WIDE_SHARE, "site_wide", "building_specific")
    events["priority"] = (events.scope == "building_specific") & (
        events.excess_pct_of_day.abs() >= PRIORITY_PCT_OF_DAY)
    events = events.drop(columns=["scale"])
    return events.sort_values("excess_pct_of_day", key=np.abs, ascending=False), p


def meter_faults(silver: pd.DataFrame):
    s = silver.copy()
    s["flag"] = s.quality_flag.isin(["flatline", "zero", "spike", "negative"])
    events = _runs(s, "flag").agg(
        start=("timestamp", "min"), end=("timestamp", "max"), hours=("timestamp", "size"),
        fault=("quality_flag", lambda f: f.mode().iloc[0]), site_id=("site_id", "first"),
    ).reset_index().drop(columns="_run")
    return events.sort_values("hours", ascending=False)
