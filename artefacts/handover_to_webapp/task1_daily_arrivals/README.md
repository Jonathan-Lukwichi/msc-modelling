# Task 1 — Daily Total ED Arrivals

This sub-folder ships everything needed to deploy the **Task 1** product in the web app.

## What Task 1 forecasts

A single time-series: the **total number of patients walking into the ED on any given day**. One number per day.

## Models available (6 — all 🟢 Operational badge)

| Alias | Family | val MAPE | val RMSE | Pickle |
|---|---|---:|---:|---|
| **Hybrid 1** | Hybrid | **12.04%** ⭐ | **8.88** ⭐ | `models/hybrid1.pkl` |
| ML 1 | Machine Learning | 11.96% | 9.39 | `models/ml1.pkl` |
| Hybrid 2 | Hybrid | 12.19% | 9.05 | `models/hybrid2.pkl` |
| ML 2 | Machine Learning | 12.32% | 9.24 | `models/ml2.pkl` |
| Stat 2 | Statistical | 12.34% | 8.98 | `models/stat2.pkl` |
| Stat 1 | Statistical | 13.33% | 10.20 | `models/stat1.pkl` |

All 6 sit at or near the noise floor for daily count forecasting. Hybrid 1 wins on RMSE; ML 1 wins on MAPE; **either is a defensible default for the web app.**

## Folder layout

```
task1_daily_arrivals/
├── README.md                       ← this file
├── models/                         6 deployable pickles
│   ├── stat1.pkl                   ARIMA
│   ├── stat2.pkl                   SARIMAX
│   ├── ml1.pkl                     XGBoost
│   ├── ml2.pkl                     ANN
│   ├── hybrid1.pkl                 SARIMAX + XGBoost
│   └── hybrid2.pkl                 SARIMAX + LSTM
├── metrics/
│   ├── headline.json               headline metrics per model
│   └── per_horizon.json            daily/weekly/monthly/yearly errors per model
├── cards/                          one card.json per model (full metadata)
│   ├── stat1.json
│   ├── stat2.json
│   ├── ml1.json
│   ├── ml2.json
│   ├── hybrid1.json
│   └── hybrid2.json
└── inference/                      ready-to-use Python helpers
    ├── load.py                     load_model(alias) → bundle
    ├── forecast.py                 forecast(bundle, horizon, start_date)
    └── example_demo.py             end-to-end smoke test
```

## API endpoints this powers

- `GET  /task1/models` → returns headline.json
- `GET  /task1/metrics` → returns per_horizon.json
- `POST /task1/forecast` → calls `inference/forecast.py`

See `../api_spec.yaml` for full schemas.

## Horizon support

All 6 models are trained on daily resolution. Aggregated horizons are computed at request time by summing daily predictions:

| Horizon | Implementation |
|---|---|
| 1d  | `model.predict(n_periods=1, X=exog_future[:1])` |
| 7d  | `model.predict(n_periods=7, X=exog_future[:7])` |
| Monthly | `sum(model.predict(n_periods=30, X=exog_future[:30]))` |
| Yearly  | `sum(model.predict(n_periods=365, X=exog_future[:365]))` |

For SARIMAX-based models (Stat 2, Hybrid 1, Hybrid 2) the `X=exog_future` matrix must be the §5.2.5 raw-10 block (6 day-of-week dummies + 2 scaled continuous + 7 calendar binaries). The bundled `feature_scaler` is used to z-score the continuous columns.

## Retraining cadence

| Trigger | Action |
|---|---|
| Time | Monthly retrain on rolling 24-month window |
| KS distance > 0.20 (train vs prediction-month distribution) | Soft alarm |
| KS distance > 0.35 | Hard retrain on most recent 18-month window |
| 3 consecutive months with >18% daily MAPE | Drift correction subroutine |
