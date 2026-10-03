from datetime import datetime, timedelta

import pytest

from energyfc import cleaning


def _frame(spark, values, building="Robin_office_X"):
    t0 = datetime(2016, 1, 1)
    rows = [(t0 + timedelta(hours=i), building, "Robin", v) for i, v in enumerate(values)]
    return spark.createDataFrame(rows, "timestamp timestamp, building_id string, site_id string, kwh double")


def _clean(spark, values):
    end = (datetime(2016, 1, 1) + timedelta(hours=len(values) - 1)).strftime("%Y-%m-%d %H:%M:%S")
    out = cleaning.clean_meters(_frame(spark, values), "2016-01-01 00:00:00", end)
    return out.orderBy("timestamp").toPandas()


def test_short_gap_is_interpolated_long_gap_is_not(spark):
    vals = [10.0 + (i % 5) for i in range(40)]
    vals[5:7] = [None, None]          # 2h gap -> fill
    vals[20:25] = [None] * 5          # 5h gap -> keep missing
    df = _clean(spark, vals)
    assert df.is_imputed[5:7].all()
    assert df.kwh[5] == pytest.approx(vals[4] + (vals[7] - vals[4]) / 3)
    assert df.kwh[20:25].isna().all()
    assert not df.is_imputed[20:25].any()


def test_flatline_zero_and_spike_flags(spark):
    vals = [10.0 + (i % 7) for i in range(120)]
    vals[30:60] = [12.0] * 30         # 30h stuck register
    vals[80] = 0.0                    # dropout
    vals[100] = 5000.0                # spike
    df = _clean(spark, vals)
    assert (df.quality_flag[30:60] == "flatline").all()
    assert df.quality_flag[80] == "zero"
    assert df.quality_flag[100] == "spike"
    assert df.kwh[30:60].isna().all()          # long fault -> left missing
    assert df.is_imputed[80] and df.is_imputed[100]  # 1h faults -> interpolated


def test_missing_timestamps_become_rows(spark):
    vals = [1.0, 2.0, 3.0, 4.0]
    sdf = _frame(spark, vals).where("hour(timestamp) != 2")
    out = cleaning.clean_meters(sdf, "2016-01-01 00:00:00", "2016-01-01 03:00:00").toPandas()
    assert len(out) == 4


def test_recurring_peaks_are_not_spikes(spark):
    # school-like profile: low nights, daily 10x peak lasting several hours
    vals = [100.0 if 9 <= (i % 24) <= 15 else 10.0 + (i % 3) for i in range(24 * 7)]
    df = _clean(spark, vals)
    assert (df.quality_flag != "spike").all()


def test_coarse_integer_meter_ramps_are_not_spikes(spark):
    # integer register with long constant stretches: typical change is 0, ramps are normal
    day = [2.0] * 7 + [3.0, 9.0, 6.0, 11.0, 9.0, 8.0, 9.0, 7.0, 4.0] + [2.0] * 8
    vals = day * 14
    vals[100] = 400.0
    df = _clean(spark, vals)
    assert set(df.index[df.quality_flag == "spike"]) == {100}
