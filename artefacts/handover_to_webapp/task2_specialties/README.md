# Task 2 — Per-Specialty ED Arrivals

This sub-folder ships everything needed to deploy the **Task 2** product in the web app — forecasting arrivals **per clinical specialty**, not just the hospital total.

## Specialties supported

| Specialty | Resolution | Daily mean | Models deployed | Best badge |
|---|---|---:|---|:-:|
| **Medicine**     | daily | ~57 patients/day  | Stat 1, ML 1, ML 2 | 🟡 |
| **Orthopaedics** | daily | ~13 patients/day  | Stat 1 only        | 🔴 |
| **Surgery**      | daily | ~3 patients/day   | ML 1, ML 2         | 🔴 |
| **Gynaecology**  | daily | ~2.5 patients/day | Stat 1, ML 1, ML 2 | 🔴 |
| **Paediatrics**  | daily | ~2.5 patients/day | Stat 1, ML 2       | 🔴 |
| **Maternity**    | **weekly** | ~0.15 patients/day  | Stat 2 (weekly)    | 🔴 |
| **Psychiatry**   | **weekly** | ~0.10 patients/day  | Stat 2 (weekly)    | 🔴 |

> ⚠️ **Maternity and Psychiatry are forecast at WEEKLY resolution only** — the daily count is too sparse (≤2 patients/day average). When a user selects either of these in the web app, the horizon selector must switch from daily mode (1d/7d/monthly/yearly) to weekly mode (1week/4weeks/yearly).

## Folder layout

```
task2_specialties/
├── README.md                       ← this file
├── catalogue.json                  ← specialty → available aliases
├── headline_all.json               ← flattened per-(specialty, alias) metrics
│
├── medicine/                       (Stat 1 + ML 1 + ML 2)
│   ├── models/
│   │   ├── stat1.pkl
│   │   ├── ml1.pkl
│   │   └── ml2.pkl
│   ├── cards/
│   └── metrics/headline.json
│
├── orthopaedics/                   (Stat 1 only — others give >148% MAPE)
│   ├── models/stat1.pkl
│   ├── cards/
│   └── metrics/headline.json
│
├── surgery/                        (ML 1 + ML 2)
├── gynaecology/                    (Stat 1 + ML 1 + ML 2)
├── paediatrics/                    (Stat 1 + ML 2)
├── maternity_weekly/               (Stat 2 weekly only)
├── psychiatry_weekly/              (Stat 2 weekly only)
│
└── inference/
    ├── load.py                     load_model(specialty, alias)
    ├── forecast.py                 forecast(bundle, horizon, start_date)
    └── example_demo.py             smoke test
```

## Filtering rule the app must enforce

When the user picks a specialty, the model dropdown is filtered to only the aliases that have a trained pickle for that specialty. **Showing an alias that isn't trained for a specialty is a deployment bug.**

The filter list lives in `catalogue.json`:

```json
[
  {"specialty": "Medicine",   "resolution": "daily",  "available_models": ["Stat 1", "ML 1", "ML 2"]},
  {"specialty": "Orthopaedics","resolution": "daily",  "available_models": ["Stat 1"]},
  ...
  {"specialty": "Maternity",  "resolution": "weekly", "available_models": ["Stat 2"]},
  {"specialty": "Psychiatry", "resolution": "weekly", "available_models": ["Stat 2"]}
]
```

## API endpoints this powers

- `GET  /task2/specialties` → returns catalogue.json
- `GET  /task2/metrics` → returns headline_all.json
- `POST /task2/forecast` → calls `inference/forecast.py` with (specialty, alias)

See `../api_spec.yaml` for full schemas.

## Honest accuracy caveat

**Only Medicine has commercially-grade accuracy (~21% MAPE)** in Task 2. All other specialties land in the 47–85% MAPE range — the thesis itself flagged Task 2 as a harder problem due to share-of-target volatility and per-specialty low counts.

The traffic-light badges (🟢🟡🔴) ensure the user is never misled. Always show the badge alongside the forecast.

## Per-specialty headline metrics (val block)

| Specialty | Alias | Scientific | val MAPE | Badge |
|---|---|---|---:|:-:|
| Medicine     | Stat 1 | ARIMA   | 21.4% | 🟡 |
| Medicine     | ML 1   | XGBoost | 21.6% | 🟡 |
| Medicine     | ML 2   | ANN     | 22.5% | 🟡 |
| Orthopaedics | Stat 1 | ARIMA   | 84.6% | 🔴 |
| Surgery      | ML 1   | XGBoost | 54.2% | 🔴 |
| Surgery      | ML 2   | ANN     | 53.6% | 🔴 |
| Gynaecology  | Stat 1 | ARIMA   | 47.3% | 🔴 |
| Gynaecology  | ML 1   | XGBoost | 47.0% | 🔴 |
| Gynaecology  | ML 2   | ANN     | 46.3% | 🔴 |
| Paediatrics  | Stat 1 | ARIMA   | 54.9% | 🔴 |
| Paediatrics  | ML 2   | ANN     | 55.0% | 🔴 |
| Maternity    | Stat 2 | SARIMAX-weekly | 54.0% | 🔴 |
| Psychiatry   | Stat 2 | SARIMAX-weekly | 77.5% | 🔴 |
