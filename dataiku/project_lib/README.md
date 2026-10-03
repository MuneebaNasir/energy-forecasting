Copy `src/energyfc/` here, then in Dataiku: **Libraries → python → upload/sync folder `energyfc`**.
Only the pandas modules are needed in Dataiku: `config.py`, `metrics.py`, `stats_tests.py`,
`backtest.py`, `anomalies.py`, `signature.py`, `__init__.py`.

    python scripts/package_dataiku_lib.py   # builds dataiku/project_lib/energyfc.zip
