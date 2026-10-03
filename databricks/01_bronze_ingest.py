# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze: ingest raw meter readings
# MAGIC BDG2 ships one column per meter (1,578 columns). Bronze reshapes it to a long
# MAGIC table `(building_id, site_id, timestamp, kwh)` and stores it as-is otherwise:
# MAGIC no cleaning, so silver can always be rebuilt from here.

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))

from pyspark.sql import functions as F
from energyfc import cleaning, config

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "energy")
dbutils.widgets.text("sites", ",".join(config.SITES))
dbutils.widgets.text("raw_dir", "")  # empty -> the Unity Catalog volume created in 00_setup
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
sites = [s.strip() for s in dbutils.widgets.get("sites").split(",") if s.strip()]
raw_dir = dbutils.widgets.get("raw_dir") or f"/Volumes/{catalog}/{schema}/raw"

# Timestamps are local wall-clock time: keep Spark in UTC so DST doesn't add/drop hours.
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

wide = spark.read.csv(f"{raw_dir}/electricity.csv", header=True)
bronze = cleaning.melt_meters(wide, sites).withColumn("_ingested_at", F.current_timestamp())
bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.bronze_meters")

for name in ["weather", "metadata"]:
    (spark.read.csv(f"{raw_dir}/{name}.csv", header=True)
     .write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{catalog}.{schema}.bronze_{name}"))

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.bronze_meters")
        .groupBy("site_id").agg(F.countDistinct("building_id").alias("meters"), F.count("*").alias("rows"),
                                F.min("timestamp").alias("first"), F.max("timestamp").alias("last")))
