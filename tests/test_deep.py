import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from energyfc import deep  # noqa: E402


def _gold(n_meters=2, n_days=10):
    ts = pd.date_range("2016-01-01", periods=24 * n_days, freq="h")
    rows = []
    for m in range(n_meters):
        rows.append(pd.DataFrame({
            "building_id": f"B{m}", "timestamp": ts,
            "y": np.arange(len(ts), dtype=float) + 1000 * m,  # value encodes (meter, hour)
            "is_imputed": False, "air_temp": 10.0, "is_holiday": 0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_history_window_ends_the_hour_before_the_target_day():
    t = deep.build_day_tensors(_gold())
    s = deep._samples(t, np.array([8]))
    for i in range(2):
        seq = s["seq"][i, :, 0].numpy()
        target = s["target"][i].numpy()
        assert seq[-1] == target[0] - 1          # last input = 23:00 the day before
        assert seq[0] == target[0] - 7 * 24      # first input = 00:00 seven days before
        assert s["last_day"][i, 0].item() == target[0] - 24


def test_rejects_incomplete_grid():
    with pytest.raises(ValueError):
        deep.build_day_tensors(_gold().iloc[:-1])
