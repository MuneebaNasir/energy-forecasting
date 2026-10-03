# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: forecasting features
# MAGIC Day-ahead setup: forecasts for all 24 hours of day D are issued at midnight, so all
# MAGIC load features are lagged ≥ 24h. Each meter is scaled by its 2016 mean (`y = kwh / scale`)
# MAGIC so a single global model learns load shapes across buildings of very different size.
# MAGIC Weather enters as hourly temperature, its 24h mean, heating/cooling degree-hours, dew point and wind.

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))

from energyfc import features

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "energy")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.conf.set("spark.sql.session.timeZone", "UTC")
t = lambda name: f"{catalog}.{schema}.{name}"

# COMMAND ----------

gold = features.build_gold(
    spark.table(t("silver_meters")),
    spark.table(t("silver_meter_quality")),
    spark.table(t("silver_weather")),
    spark.table(t("bronze_metadata")),
)
(gold.write.mode("overwrite").option("overwriteSchema", "true")
 .partitionBy("site_id").saveAsTable(t("gold_features")))

# COMMAND ----------

display(spark.sql(f"""
  SELECT site_id, COUNT(DISTINCT building_id) AS meters, COUNT(*) AS rows,
         ROUND(AVG(CASE WHEN y IS NULL THEN 1 ELSE 0 END), 3) AS share_missing_target
  FROM {t('gold_features')} GROUP BY site_id
"""))
