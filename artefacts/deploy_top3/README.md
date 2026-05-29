# Deployment artefacts — top-3 RMSE-minimising models

**Hospital**: Steve Biko Academic Hospital, Pretoria, South Africa
**Target**: `total_daily_arrivals` (daily ED arrival count)
**Fit window**: train + val (2022-03-01 → 2024-12-31, 1,037 days)
**Selection criterion**: minimum val_RMSE on held-out 184-day validation block
**Source**: Chapter 6 modelling decisions

## Manifest

| Rank | Model | Family | val_MAPE | val_RMSE | Pickle size |
|---:|---|---|---:|---:|---:|
| **1** | `sarimax_xgb` | Hybrid (SARIMAX base + XGBoost residual refiner) | **12.04%** | **8.88** | 38.1 MB |
| **2** | `sarimax` | SARIMAX(1,1,1)(0,1,1,7), auto_arima AIC-selected | 12.34% | 8.98 | 37.8 MB |
| **3** | `sarimax_lstm` | Hybrid (SARIMAX base + LSTM residual refiner) | 12.19% | 9.05 | 37.9 MB |

## Why these three

- **Rank 1 — SARIMAX+XGB**: best val_RMSE overall. SARIMAX produces the level forecast; XGBoost refines its residual. Picked by Optuna refiner-only HPO (10 trials × 5 folds).
- **Rank 2 — SARIMAX (auto_arima)**: classical baseline at Chapter 5 §5.2.2 mandated `(p,1,q)(P,1,Q)_7` with AIC selection. Ships as a fallback if XGBoost/torch dependencies are unavailable.
- **Rank 3 — SARIMAX+LSTM**: LSTM residual refiner. Pre-existing RMSE-tuned artefact. Ships for ensemble averaging.

## Loading a model (Python)

```python
import joblib

bundle = joblib.load("artefacts/deploy_top3/01_sarimax_xgb.pkl")

# Typical sub-components inside the hybrid bundle:
#   bundle['base']           -> fitted SARIMAX (pmdarima.ARIMA)
#   bundle['refiner']        -> fitted XGBRegressor on SARIMAX residuals
#   bundle['exog_scaler']    -> StandardScaler for the raw-10 exog block
#   bundle['feature_names']  -> list of refiner feature names
```

## Operational expectations

| Horizon | Aggregation | Expected MAPE | Use case |
|---|---|---:|---|
| Daily | none | ~12% (±7 patients on a 60-patient day) | Next-day staff planning (with safety buffer) |
| Weekly | sum-7 | ~5% | Week-ahead nurse roster |
| Monthly | sum-30 | ~2% | Monthly budget allocation |
| Yearly | sum-365 | ~1% | Annual capacity planning, DoH submissions |

## Web-app integration checklist

- [ ] `pip install joblib pmdarima xgboost torch pandas pyyaml fastapi uvicorn`
- [ ] Mount `artefacts/deploy_top3/` into the app container
- [ ] For each forecast day, build the raw-10 exog row from G1 (6 dow dummies + 2 z-scored continuous weather + 7 calendar binaries)
- [ ] Call `bundle['base'].predict(n_periods=H, X=exog_future)` → level forecast
- [ ] Build feature row for refiner from engineered + consensus pipeline
- [ ] Final forecast = `level + refiner.predict(features_future)`
- [ ] Wrap with 95% conformal prediction interval (use existing MAPIE artefacts in `artefacts/uq/`)
- [ ] Log `(date, actual, predicted, lower_95, upper_95)` for drift monitoring

## Re-training cadence

| Trigger | Action |
|---|---|
| **Time** | Monthly retrain on rolling 24-month window |
| **KS distance > 0.20** between training month and prediction month | Soft alarm — investigate, manual review |
| **KS distance > 0.35** (Chapter 5 §5.5.2 drift band) | Hard alarm — force retrain on most recent 18-month window |
| **3 consecutive months above 18% MAPE** | Drift correction subroutine — apply offset bias correction (per Chapter 6) |

## Audit trail

- Trained: 2026-05-29
- Phase 2 HPO traces: `artefacts/phase2_hpo/trace_*.csv`
- Validation predictions: `artefacts/phase2_hpo/val_preds_*.csv`
- Phase 1 default-baseline comparison: `artefacts/phase1_defaults/summary_phase1.csv`
- Literature corpus reference: `artefacts/lit_review/pattern_report.md`
- Source thesis chapter: Chapter_6_Model_Development_and_Evaluation.pdf

## Liability disclaimer

These models are research-grade artefacts produced for academic evaluation. **Deployment in a clinical decision context requires**:
1. Multi-site prospective validation (minimum 3 hospitals × 12 months)
2. Ethical approval at deployment site
3. POPIA compliance review (South Africa)
4. Clinician training and on-site validation
5. Continuous drift monitoring + manual override pathway

No warranty is provided for production use without these steps.
