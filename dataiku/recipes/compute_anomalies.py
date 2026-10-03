# Dataiku Python recipe: compute_anomalies
#   inputs : predictions, silver_meters
#   outputs: consumption_anomalies, meter_faults
import pandas as pd

import dataiku
from energyfc import anomalies

pred = dataiku.Dataset("predictions").get_dataframe()
pred["timestamp"] = pd.to_datetime(pred["timestamp"])
pred["scored"] = pred["scored"].astype(str).str.lower().isin(["true", "1"])
silver = dataiku.Dataset("silver_meters").get_dataframe()
silver["timestamp"] = pd.to_datetime(silver["timestamp"])

events, _ = anomalies.consumption_anomalies(pred, model="lgbm")
dataiku.Dataset("consumption_anomalies").write_with_schema(events)
dataiku.Dataset("meter_faults").write_with_schema(anomalies.meter_faults(silver))
