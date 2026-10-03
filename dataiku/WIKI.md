# Energy load forecasting: project handover

*Paste as the Dataiku wiki home page. Written so that someone picking the project up
can run, change and defend it without the original author.*

## What this project does
Day-ahead, hourly electricity forecasts per meter (76 meters, two campuses: London and
Dublin) plus two monitoring outputs: **meter faults** (the measurement is wrong) and
**consumption anomalies** (the building used far more/less than expected for the weather and day).

## Where things live
| Layer | Where | Owner of the logic |
|---|---|---|
| Raw → bronze → silver → gold | Databricks, `workspace.energy.*`, job `energy-forecasting-pipeline` | `src/energyfc/cleaning.py`, `features.py` |
| Model training & comparison | This project, zone *Modeling* | Visual ML `load_forecast` + recipe `compute_backtest` |
| Significance tests | recipe `compute_dm_tests` | `src/energyfc/stats_tests.py` |
| Faults & anomalies | zone *Monitoring* | `src/energyfc/anomalies.py` |
| Code shared by both platforms | Git repo, synced into *Libraries → python/energyfc* | |

## Decisions and why
- **Forecast issued at midnight for the next 24h** → every load feature is lagged ≥ 24h.
  Do not add `lag_1`…`lag_23`: they would not be available in production.
- **Observed weather is used in place of a weather forecast** (perfect-forecast assumption).
  Real accuracy will be a bit worse; swap `air_temp` for Météo-France / ECMWF forecasts in production.
- **Time-based train/test split only.** A random split leaks the future through lag features.
- **One global model for all meters**, target scaled by each meter's 2016 mean. Lets small,
  noisy meters borrow strength from similar buildings; scale is computed on 2016 only (no leakage).
- **Interpolated hours (gaps ≤ 3h) are used as model inputs but never scored.**
- **Spikes are defined locally** (jump vs both neighbours), not as global outliers: peaky
  buildings such as schools would otherwise lose their real peaks.
- **Significance is tested per meter on daily losses** (Diebold-Mariano, HAC variance), with
  Benjamini-Hochberg across meters. Hourly errors within a day are not independent.
- **Production model = mean of LightGBM and GRU** (best on 63/76 meters, worse on none). Per-meter
  "pick the winner" was tested and did worse: monthly winners are noisy.
- **P10–P90 bands are conformally calibrated per meter** on the 4 weeks before each month
  (73.7 % → 79.3 % coverage). Check `interval_scores.coverage_calibrated` stays near 0.80.

## How to run
- Monthly: scenario **Monthly retrain** (1st of month, 03:00). It fails, and e-mails, if LightGBM
  stops beating the seasonal-naive baseline by at least 5% (median MASE).
- Manual: build `dm_tests` and `consumption_anomalies` recursively.

## Known limitations / next steps
1. Weather forecasts instead of observations (see above).
2. Serve the LightGBM+GRU ensemble from Dataiku (needs a PyTorch code env).
3. Meter faults are rule-based; confirm thresholds with the metering team.
4. Two campuses only; the Databricks job takes a `sites` parameter to scale to all 19 BDG2 sites.
