# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the schema and a Unity Catalog volume for raw files, then downloads the
# MAGIC [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2)
# MAGIC electricity, weather and metadata files into it.
# MAGIC
# MAGIC If the cluster has no outbound internet, upload the three CSVs to the volume via
# MAGIC **Catalog → energy → raw → Upload to this volume** instead.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "energy")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw")
raw_dir = f"/Volumes/{catalog}/{schema}/raw"

# COMMAND ----------

import os
import urllib.request

BASE = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
FILES = {
    "electricity.csv": f"{BASE}/meters/raw/electricity.csv",
    "weather.csv": f"{BASE}/weather/weather.csv",
    "metadata.csv": f"{BASE}/metadata/metadata.csv",
}
for name, url in FILES.items():
    target = f"{raw_dir}/{name}"
    if not os.path.exists(target):
        urllib.request.urlretrieve(url, target)
    print(name, f"{os.path.getsize(target) / 1e6:.1f} MB")
