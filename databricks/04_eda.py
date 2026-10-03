# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Exploratory analysis
# MAGIC Trends, seasonality, weather sensitivity (energy signature) and correlations,
# MAGIC computed on the gold table. Use the chart editor on each `display()` to switch to a line/scatter view.

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))

from pyspark.sql import functions as F
from energyfc import signature

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "energy")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.conf.set("spark.sql.session.timeZone", "UTC")
gold = spark.table(f"{catalog}.{schema}.gold_features")

# COMMAND ----------

# MAGIC %md ### Trend and seasonality: daily site load vs temperature

# COMMAND ----------

display(gold.groupBy("site_id", "date").agg(
    F.sum("kwh").alias("site_kwh"), F.avg("air_temp").alias("temp_c")).orderBy("date"))

# COMMAND ----------

# MAGIC %md ### Weekly profile: hour × day-of-week (normalised load)

# COMMAND ----------

display(gold.groupBy("site_id", "dow", "hour").agg(F.avg("y").alias("mean_norm_load")).orderBy("site_id", "dow", "hour"))

# COMMAND ----------

# MAGIC %md ### Correlation of normalised load with drivers, per building use

# COMMAND ----------

drivers = ["air_temp", "hdd", "dew_temp", "wind_speed", "y_lag_24", "y_lag_168"]
display(gold.groupBy("primary_use").agg(*[F.corr("y", c).alias(f"corr_{c}") for c in drivers]))

# COMMAND ----------

# MAGIC %md ### Energy signature per building (change-point regression, working days)

# COMMAND ----------

pdf = gold.select("building_id", "site_id", "date", "kwh", "air_temp", "is_weekend", "is_holiday").toPandas()
sig, daily = signature.energy_signatures(pdf)
spark.createDataFrame(sig).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.gold_energy_signatures")
display(sig)
