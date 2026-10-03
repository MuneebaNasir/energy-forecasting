# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: data quality
# MAGIC For every meter-hour:
# MAGIC - **complete hourly grid**: missing timestamps become explicit rows, so lag features stay correct
# MAGIC - **quality flag**: `missing`, `negative`, `flatline` (same value ≥ 24h: stuck register),
# MAGIC   `zero` (0 on a meter whose median is > 0: comms dropout), `spike` (isolated jump vs both neighbours)
# MAGIC - flagged values are removed; gaps ≤ 3h are linearly interpolated (`is_imputed`), longer gaps stay missing
# MAGIC - meters with < 85% valid hours are excluded from modelling (`silver_meter_quality.keep`)

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))

from pyspark.sql import functions as F
from energyfc import cleaning, config

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "energy")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.conf.set("spark.sql.session.timeZone", "UTC")
START, END = "2016-01-01 00:00:00", "2017-12-31 23:00:00"
t = lambda name: f"{catalog}.{schema}.{name}"

# COMMAND ----------

bronze = spark.table(t("bronze_meters")).drop("_ingested_at")
silver = cleaning.clean_meters(bronze, START, END)
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t("silver_meters"))

quality = cleaning.meter_quality_report(spark.table(t("silver_meters")))
quality.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t("silver_meter_quality"))

sites = [r.site_id for r in bronze.select("site_id").distinct().collect()]
weather = cleaning.clean_weather(spark.table(t("bronze_weather")), sites, START, END)
weather.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t("silver_weather"))

# COMMAND ----------

# MAGIC %md ### Data quality summary

# COMMAND ----------

display(spark.table(t("silver_meters")).groupBy("site_id", "quality_flag").count().orderBy("site_id", F.desc("count")))

# COMMAND ----------

display(spark.table(t("silver_meter_quality")).orderBy("share_ok"))

# COMMAND ----------

# Data quality expectations: fail the job loudly instead of feeding bad data downstream
q = spark.table(t("silver_meter_quality"))
kept = q.where("keep").count()
assert kept >= 0.5 * q.count(), f"only {kept}/{q.count()} meters pass quality: check the source"
w = spark.table(t("silver_weather"))
assert w.where("air_temp IS NULL").count() / w.count() < 0.01, "weather has > 1% unrecoverable gaps"
