"""Bronze -> silver transforms (PySpark). Pure functions: DataFrame in, DataFrame out.

The same code runs in the Databricks notebooks (Delta tables) and locally (parquet)
via scripts/run_pipeline_local.py.

BDG2 timestamps are local wall-clock time. The Spark session must run with
spark.sql.session.timeZone=UTC so they are treated as naive: in a DST-aware zone
the hourly grid would gain/lose an hour twice a year and break lag features.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from energyfc import config


def melt_meters(wide: DataFrame, sites=None) -> DataFrame:
    """BDG2 ships one column per meter; reshape to (building_id, site_id, timestamp, kwh)."""
    meter_cols = [c for c in wide.columns if c != "timestamp"]
    if sites:
        meter_cols = [c for c in meter_cols if c.split("_")[0] in sites]
    long = wide.select("timestamp", *meter_cols).unpivot(
        ["timestamp"], meter_cols, "building_id", "kwh"
    )
    return long.select(
        F.to_timestamp("timestamp").alias("timestamp"),
        "building_id",
        F.split("building_id", "_")[0].alias("site_id"),
        F.col("kwh").cast("double").alias("kwh"),
    )


def complete_hourly_grid(df: DataFrame, key: str, start: str, end: str) -> DataFrame:
    """Left-join onto a full hourly grid so every gap becomes an explicit null row.

    Lag features downstream rely on this: lag(24) is only "same hour yesterday"
    if no hour is missing from the partition.
    """
    hours = df.sparkSession.sql(
        f"SELECT explode(sequence(timestamp'{start}', timestamp'{end}', interval 1 hour)) AS timestamp"
    )
    keys = df.select(*([key, "site_id"] if key != "site_id" else [key])).distinct()
    grid = keys.crossJoin(hours)
    deduped = df.dropDuplicates([key, "timestamp"])
    join_cols = [key, "timestamp"] + (["site_id"] if key != "site_id" else [])
    return grid.join(deduped, join_cols, "left")


def interpolate_short_gaps(df: DataFrame, key: str, col: str, max_gap_hours: int) -> DataFrame:
    """Linearly interpolate runs of nulls no longer than max_gap_hours; longer gaps stay null.

    Adds `<col>_imputed` (bool). Assumes a complete hourly grid.
    """
    w = Window.partitionBy(key).orderBy("timestamp")
    t = F.col("timestamp").cast("long")
    point = F.when(F.col(col).isNotNull(), F.struct(t.alias("t"), F.col(col).alias("v")))
    prev = F.last(point, ignorenulls=True).over(w.rowsBetween(Window.unboundedPreceding, -1))
    nxt = F.first(point, ignorenulls=True).over(w.rowsBetween(1, Window.unboundedFollowing))

    df = df.withColumn("_prev", prev).withColumn("_next", nxt)
    gap_hours = (F.col("_next.t") - F.col("_prev.t")) / 3600 - 1
    frac = (t - F.col("_prev.t")) / (F.col("_next.t") - F.col("_prev.t"))
    interp = F.col("_prev.v") + frac * (F.col("_next.v") - F.col("_prev.v"))
    fillable = F.col(col).isNull() & (gap_hours <= max_gap_hours)

    return (
        df.withColumn(f"{col}_imputed", F.coalesce(fillable, F.lit(False)))
        .withColumn(col, F.when(fillable, interp).otherwise(F.col(col)))
        .drop("_prev", "_next")
    )


def flag_meter_quality(df: DataFrame) -> DataFrame:
    """Label each reading: ok / missing / negative / flatline / zero / spike.

    - flatline: same reading repeated >= FLATLINE_MIN_HOURS (stuck register or
      a gap the provider back-filled with a constant)
    - zero: reading of 0 on a meter whose median is > 0 (comms dropout, not real load)
    - spike: an isolated reading that jumps away from *both* neighbours by more
      than SPIKE_ROBUST_Z robust std-devs of the meter's (non-zero) hour-to-hour
      changes, and lands outside the meter's own 1-99 percentile range.
      A global outlier rule is deliberately avoided: peaky buildings (e.g. a
      school at 10x its night load) would have their real peaks flagged.
    """
    w = Window.partitionBy("building_id").orderBy("timestamp")
    changed = (
        F.col("kwh").isNull()
        | F.lag("kwh").over(w).isNull()
        | (F.col("kwh") != F.lag("kwh").over(w))
    )
    df = df.withColumn("_run_id", F.sum(changed.cast("int")).over(w))
    df = df.withColumn(
        "_run_len", F.count("*").over(Window.partitionBy("building_id", "_run_id"))
    )

    d_prev = F.col("kwh") - F.lag("kwh").over(w)
    d_next = F.col("kwh") - F.lead("kwh").over(w)
    df = df.withColumn("_d_prev", d_prev).withColumn("_d_next", d_next)
    nonzero_change = F.when(F.col("_d_prev") != 0, F.abs("_d_prev"))
    stats = df.groupBy("building_id").agg(
        F.percentile_approx("kwh", 0.5).alias("_median"),
        F.percentile_approx("kwh", 0.01).alias("_p_lo"),
        F.percentile_approx("kwh", 0.99).alias("_p_hi"),
        F.percentile_approx(nonzero_change, 0.5).alias("_mad_diff"),
    )
    df = df.join(stats, "building_id")
    is_spike = (
        (F.signum("_d_prev") == F.signum("_d_next"))
        & (F.least(F.abs("_d_prev"), F.abs("_d_next"))
           > config.SPIKE_ROBUST_Z * 1.4826 * F.col("_mad_diff"))
        & ((F.col("kwh") > F.col("_p_hi")) | (F.col("kwh") < F.col("_p_lo")))
    )

    flag = (
        F.when(F.col("kwh").isNull(), "missing")
        .when(F.col("kwh") < 0, "negative")
        .when(F.col("_run_len") >= config.FLATLINE_MIN_HOURS, "flatline")
        .when((F.col("kwh") == 0) & (F.col("_median") > 0), "zero")
        .when(is_spike, "spike")
        .otherwise("ok")
    )
    return df.withColumn("quality_flag", flag).drop(
        "_run_id", "_run_len", "_median", "_p_lo", "_p_hi", "_mad_diff", "_d_prev", "_d_next"
    )


def clean_meters(long: DataFrame, start: str, end: str) -> DataFrame:
    """Bronze long table -> silver: complete grid, quality flags, short-gap interpolation."""
    df = complete_hourly_grid(long, "building_id", start, end)
    df = flag_meter_quality(df)
    df = df.withColumn("kwh_raw", F.col("kwh")).withColumn(
        "kwh", F.when(F.col("quality_flag") == "ok", F.col("kwh"))
    )
    df = interpolate_short_gaps(df, "building_id", "kwh", config.MAX_INTERP_GAP_HOURS)
    return df.withColumnRenamed("kwh_imputed", "is_imputed").select(
        "building_id", "site_id", "timestamp", "kwh_raw", "kwh", "quality_flag", "is_imputed"
    )


def meter_quality_report(silver: DataFrame) -> DataFrame:
    """One row per meter: share of each flag and whether it is kept for modeling."""
    flags = ["ok", "missing", "negative", "flatline", "zero", "spike"]
    aggs = [F.avg((F.col("quality_flag") == f).cast("double")).alias(f"share_{f}") for f in flags]
    aggs.append(F.avg(F.col("kwh").isNotNull().cast("double")).alias("share_usable"))
    report = silver.groupBy("building_id", "site_id").agg(*aggs)
    return report.withColumn("keep", F.col("share_ok") >= config.MIN_VALID_SHARE)


def clean_weather(weather: DataFrame, sites, start: str, end: str) -> DataFrame:
    df = weather.where(F.col("site_id").isin(sites)).select(
        F.to_timestamp("timestamp").alias("timestamp"),
        "site_id",
        F.col("airTemperature").cast("double").alias("air_temp"),
        F.col("dewTemperature").cast("double").alias("dew_temp"),
        F.col("windSpeed").cast("double").alias("wind_speed"),
    )
    df = complete_hourly_grid(df, "site_id", start, end)
    for c in ["air_temp", "dew_temp", "wind_speed"]:
        df = interpolate_short_gaps(df, "site_id", c, config.WEATHER_MAX_INTERP_HOURS)
    return df.drop("dew_temp_imputed", "wind_speed_imputed").withColumnRenamed(
        "air_temp_imputed", "temp_imputed"
    )
