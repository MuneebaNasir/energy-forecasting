# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Hand-off to Dataiku
# MAGIC **Preferred:** in Dataiku, add a Databricks connection and create a dataset directly on
# MAGIC `gold_features` (no copy). **Fallback** (e.g. Dataiku Free Edition without the Databricks
# MAGIC connector): export the tables below as files and upload them as Dataiku datasets.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "energy")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.exports")
out = f"/Volumes/{catalog}/{schema}/exports"

for table in ["gold_features", "silver_meters", "silver_meter_quality", "gold_energy_signatures"]:
    (spark.table(f"{catalog}.{schema}.{table}").coalesce(1)
     .write.mode("overwrite").option("header", True).option("compression", "gzip")
     .csv(f"{out}/{table}"))
    print(table, "->", f"{out}/{table}")
