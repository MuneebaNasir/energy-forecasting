"""Shared constants for the pipeline. Notebooks and recipes override via widgets/variables."""

SITES = ["Robin", "Wolf"]  # London and Dublin campuses (local time, heating-dominated)

TRAIN_END = "2017-01-01"  # 2016 = history; 2017 = 12 monthly rolling-origin test folds

# Cleaning thresholds
FLATLINE_MIN_HOURS = 24      # identical consecutive readings for >= 24h -> stuck meter
MAX_INTERP_GAP_HOURS = 3     # gaps up to 3h are linearly interpolated, longer stay missing
SPIKE_ROBUST_Z = 10.0        # isolated jump vs both neighbours, in robust std-devs of hourly changes
MIN_VALID_SHARE = 0.85       # meters with fewer valid hours are excluded from modeling
WEATHER_MAX_INTERP_HOURS = 6

# Degree-day base temperatures (°C), standard UK/IE values
HDD_BASE = 15.5
CDD_BASE = 22.0

# Model inputs (built in features.build_gold). Load features are all lagged >= 24h.
FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_holiday",
    "air_temp", "temp_mean_24h", "hdd", "cdd", "dew_temp", "wind_speed",
    "y_lag_24", "y_lag_48", "y_lag_168", "y_lag_336",
    "y_mean_24_lag24", "y_mean_168_lag24",
    "primary_use", "sqm",
]
CATEGORICAL = ["primary_use"]

# Public holidays for the two sites (England & Wales for Robin, Ireland for Wolf)
HOLIDAYS = {
    "Robin": [
        "2016-01-01", "2016-03-25", "2016-03-28", "2016-05-02", "2016-05-30", "2016-08-29",
        "2016-12-26", "2016-12-27", "2017-01-02", "2017-04-14", "2017-04-17", "2017-05-01",
        "2017-05-29", "2017-08-28", "2017-12-25", "2017-12-26",
    ],
    "Wolf": [
        "2016-01-01", "2016-03-17", "2016-03-28", "2016-05-02", "2016-06-06", "2016-08-01",
        "2016-10-31", "2016-12-26", "2016-12-27", "2017-01-02", "2017-03-17", "2017-04-17",
        "2017-05-01", "2017-06-05", "2017-08-07", "2017-10-30", "2017-12-25", "2017-12-26",
    ],
}
