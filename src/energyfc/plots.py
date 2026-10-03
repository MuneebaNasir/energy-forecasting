"""Matplotlib figures for the README / stakeholder summary (static, light mode)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
MUTED = "#b4b2ab"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
MODEL_LABELS = {"ens_mean": "Mean of LightGBM + GRU", "ens_select": "Per-meter champion",
                "lgbm_gated": "LightGBM, weather where it matters",
                "gru": "GRU (deep learning)", "lgbm": "LightGBM + weather", "lgbm_no_weather": "LightGBM, no weather",
                "snaive_168": "Same hour last week", "naive_24": "Same hour yesterday"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 130,
})


def _save(fig, path):
    """Write to `path` and close, or return the open figure when path is None (Dataiku insights)."""
    fig.tight_layout()
    if path is None:
        return fig
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return None


def quality_example(silver_meter: pd.DataFrame, path=None):
    s = silver_meter.sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(s.timestamp, s.kwh_raw, color=MUTED, lw=0.6, label="Raw reading")
    bad = s.kwh_raw.where(s.quality_flag.isin(["flatline", "zero", "spike"]))
    ax.plot(s.timestamp, bad, color=SERIES[1], lw=1.8, label="Flagged (stuck / zero / spike)")
    ax.set_title(f"Meter faults detected in silver: {s.building_id.iloc[0]}")
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, path)


def signatures(daily: pd.DataFrame, sig: pd.DataFrame, buildings, path=None):
    fig, axes = plt.subplots(1, len(buildings), figsize=(4.2 * len(buildings), 3.4), sharey=False)
    for ax, b in zip(np.atleast_1d(axes), buildings, strict=True):
        d = daily[daily.building_id == b]
        r = sig.set_index("building_id").loc[b]
        ax.scatter(d.temp, d.kwh_day, s=9, color=SERIES[0], alpha=0.45, linewidths=0)
        t = np.linspace(d.temp.min(), d.temp.max(), 100)
        fit = r.base_kwh_day + r.heating_kwh_per_degday * np.maximum(0, r.balance_temp_c - t)
        if r.r2 >= 0.5:
            ax.plot(t, fit, color=TEXT, lw=1.6)
            ax.axvline(r.balance_temp_c, color=TEXT_2, lw=0.8)
        label = (f"balance {r.balance_temp_c:.1f} °C\n+{r.heating_pct_per_c:.1f}% per °C colder\nR² {r.r2:.2f}"
                 if r.r2 >= 0.5 else f"no clear temperature effect\nR² {r.r2:.2f}")
        ax.annotate(label,
                    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", color=TEXT_2, fontsize=9)
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("Daily mean temperature (°C)")
    np.atleast_1d(axes)[0].set_ylabel("kWh per working day")
    fig.suptitle("Energy signature: consumption vs temperature", x=0.01, ha="left",
                 fontweight="bold", color=TEXT)
    return _save(fig, path)


def forecast_week(pred: pd.DataFrame, building, start, path=None):
    p = pred[(pred.building_id == building) & (pred.timestamp >= start)
             & (pred.timestamp < pd.Timestamp(start) + pd.Timedelta("7D"))]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    actual = p[p.model == "lgbm"].sort_values("timestamp")
    ax.plot(actual.timestamp, actual.kwh, color=TEXT, lw=1.8, label="Actual")
    for i, m in enumerate(["lgbm", "snaive_168"]):
        q = p[p.model == m].sort_values("timestamp")
        ax.plot(q.timestamp, q.kwh_hat, color=SERIES[i], lw=1.6, label=MODEL_LABELS[m])
    ax.set_title(f"Day-ahead forecast, one week: {building}")
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper left", ncol=3)
    return _save(fig, path)


def mase_by_model(per_meter: pd.DataFrame, path=None):
    order = per_meter.groupby("model").mase.median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.0))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = per_meter[per_meter.model == m].mase
        color = {"lgbm": SERIES[0], "gru": SERIES[1], "ens_mean": SERIES[2]}.get(m, MUTED)
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10, color=color, alpha=0.6, linewidths=0)
        med = v.median()
        ax.plot([med, med], [i - 0.3, i + 0.3], color=TEXT, lw=2)
        ax.annotate(f"{med:.2f}", (med, i + 0.32), ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.axvline(1.0, color=TEXT_2, lw=0.8)
    ax.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order])
    ax.set_xlabel("MASE per meter (lower is better; 1 = last week's profile in 2016)")
    ax.set_title("Forecast error across 2017, one dot per meter (bar = median)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def importance(imp: pd.DataFrame, path=None, top=12):
    g = imp.groupby("feature").gain.mean()
    g = (g / g.sum() * 100).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(g.index, g.values, color=SERIES[0], height=0.7)
    for y, v in enumerate(g.values):
        ax.annotate(f"{v:.1f}%", (v, y), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT_2)
    ax.set_xlabel("Share of total gain, mean over 12 monthly refits (%)")
    ax.set_title("What drives the LightGBM forecast")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def anomaly_example(scored: pd.DataFrame, event: pd.Series, path=None):
    lo, hi = event.start - pd.Timedelta("3D"), event.end + pd.Timedelta("3D")
    p = scored[(scored.building_id == event.building_id) & scored.timestamp.between(lo, hi)]
    p = p.sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axvspan(event.start, event.end + pd.Timedelta("1h"), color=SERIES[1], alpha=0.15, lw=0)
    ax.plot(p.timestamp, p.kwh, color=TEXT, lw=1.6, label="Actual")
    ax.plot(p.timestamp, p.kwh_hat, color=SERIES[0], lw=1.6, label="Expected (forecast)")
    word = "more" if event.excess_kwh > 0 else "less"
    ax.set_title(f"Consumption anomaly: {event.building_id}, {event.hours}h, "
                 f"{abs(event.excess_kwh):,.0f} kWh {word} than expected")
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, path)


def prediction_interval(qpred: pd.DataFrame, building, start, path=None):
    q = qpred[(qpred.building_id == building) & (qpred.timestamp >= start)
              & (qpred.timestamp < pd.Timestamp(start) + pd.Timedelta("7D"))].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(q.timestamp, q.q10_cal, q.q90_cal, color=SERIES[0], alpha=0.18, lw=0,
                    label="P10-P90, calibrated")
    ax.plot(q.timestamp, q.q50, color=SERIES[0], lw=1.6, label="P50 forecast")
    ax.plot(q.timestamp, q.kwh, color=TEXT, lw=1.6, label="Actual")
    inside = ((q.kwh >= q.q10_cal) & (q.kwh <= q.q90_cal)).mean()
    ax.set_title(f"Probabilistic day-ahead forecast: {building} ({inside:.0%} of hours inside the band)")
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper left", ncol=3)
    return _save(fig, path)
