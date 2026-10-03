"""Silver -> gold feature table (PySpark).

Forecast setup: day-ahead, hourly. Forecasts for all 24 hours of day D are issued
at midnight, so every load feature must be lagged by >= 24h. Weather uses observed
values as a stand-in for a day-ahead weather forecast (perfect-forecast assumption,
documented in the README).
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from energyfc import config

LOAD_LAGS = [24, 48, 168, 336]
FEATURES, CATEGORICAL = config.FEATURES, config.CATEGORICAL


def holidays_frame(spark) -> DataFrame:
    rows = [(site, d) for site, days in config.HOLIDAYS.items() for d in days]
    return spark.createDataFrame(rows, ["site_id", "date"]).select(
        "site_id", F.to_date("date").alias("date"), F.lit(1).alias("is_holiday")
    )


def build_gold(silver: DataFrame, quality: DataFrame, weather: DataFrame,
               metadata: DataFrame, train_end: str = config.TRAIN_END) -> DataFrame:
    kept = quality.where("keep").select("building_id")
    df = silver.join(kept, "building_id")

    # Scale each meter by its mean over the history period only (no test leakage),
    # so one global model can learn shapes across meters of very different size.
    scale = (
        df.where(F.col("timestamp") < F.lit(train_end).cast("timestamp"))
        .groupBy("building_id").agg(F.avg("kwh").alias("scale"))
    )
    df = df.join(scale, "building_id").withColumn("y", F.col("kwh") / F.col("scale"))

    w = Window.partitionBy("building_id").orderBy("timestamp")
    for lag in LOAD_LAGS:
        df = df.withColumn(f"y_lag_{lag}", F.lag("y", lag).over(w))
    df = df.withColumn("y_mean_24_lag24", F.avg("y").over(w.rowsBetween(-47, -24)))
    df = df.withColumn("y_mean_168_lag24", F.avg("y").over(w.rowsBetween(-191, -24)))

    ws = Window.partitionBy("site_id").orderBy("timestamp").rowsBetween(-23, 0)
    weather = weather.withColumn("temp_mean_24h", F.avg("air_temp").over(ws))
    df = df.join(weather, ["site_id", "timestamp"], "left")

    meta = metadata.select(
        "building_id",
        F.col("primaryspaceusage").alias("primary_use"),
        F.col("sqm").cast("double").alias("sqm"),
    )
    df = df.join(meta, "building_id", "left")

    df = (
        df.withColumn("date", F.to_date("timestamp"))
        .join(holidays_frame(df.sparkSession), ["site_id", "date"], "left")
        .withColumn("is_holiday", F.coalesce("is_holiday", F.lit(0)))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("dow", F.dayofweek("timestamp"))  # 1 = Sunday
        .withColumn("month", F.month("timestamp"))
        .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
        .withColumn("hdd", F.greatest(F.lit(0.0), config.HDD_BASE - F.col("temp_mean_24h")))
        .withColumn("cdd", F.greatest(F.lit(0.0), F.col("temp_mean_24h") - config.CDD_BASE))
    )
    return df.select(
        "building_id", "site_id", "timestamp", "date", "kwh", "y", "scale",
        "is_imputed", "quality_flag", *FEATURES,
    )
