# Master Model Comparison Report

This report consolidates Phases 1, 2, and 3 of the Task-1 (daily ED arrivals) forecasting experiment.

## Phase 1 — Chapter 5 defaults (no HPO)

| model       |   cv_avg_RMSE |   cv_avg_MAPE |   val_MAPE |   val_MAE |   val_RMSE |   val_R2 |   weekly_avg_pct_error |   monthly_avg_pct_error |   yearly_avg_pct_error |
|:------------|--------------:|--------------:|-----------:|----------:|-----------:|---------:|-----------------------:|------------------------:|-----------------------:|
| arima       |         8.927 |        12.439 |     13.324 |     7.798 |     10.195 |   -0.058 |                  6.064 |                   2.254 |                  0.762 |
| sarimax     |         8.382 |        12.339 |     12.582 |     7.211 |      9.222 |    0.135 |                  6.067 |                   2.415 |                  1.467 |
| stl_xgb     |        13.871 |        18.718 |     15.405 |     9.018 |     11.540 |   -0.355 |                  8.515 |                   2.867 |                  0.648 |
| stl_ann     |        12.900 |        17.671 |     15.703 |     9.136 |     11.955 |   -0.454 |                  8.152 |                   2.777 |                  0.241 |
| sarimax_xgb |        10.668 |        15.338 |     12.753 |     7.383 |      9.492 |    0.083 |                  6.398 |                   2.469 |                  0.381 |

## Phase 2 — HPO targeting minimum RMSE

| model   | algo       |   n_trials |   cv_RMSE |   cv_MAPE |   cv_MAE |   val_MAPE |   val_MAE |   val_RMSE |   val_R2 | winner_params                                |
|:--------|:-----------|-----------:|----------:|----------:|---------:|-----------:|----------:|-----------:|---------:|:---------------------------------------------|
| arima   | auto_arima |          1 |     7.791 |    10.035 |    6.330 |     13.326 |     7.798 |     10.198 |   -0.058 | {"order": [0, 1, 2], "seasonal_order": null} |

## Master comparison — defaults vs HPO winner

| model       |   defaults_cv_RMSE |   defaults_cv_MAPE |   defaults_val_MAPE |   defaults_val_RMSE |   defaults_val_R2 |   defaults_yearly_pct | hpo_algo   |   hpo_cv_RMSE |   hpo_cv_MAPE |   hpo_val_MAPE |   hpo_val_RMSE |   hpo_val_R2 |   delta_MAPE_pp |   delta_RMSE |
|:------------|-------------------:|-------------------:|--------------------:|--------------------:|------------------:|----------------------:|:-----------|--------------:|--------------:|---------------:|---------------:|-------------:|----------------:|-------------:|
| arima       |              8.927 |             12.439 |              13.324 |              10.195 |            -0.058 |                 0.762 | auto_arima |         7.791 |        10.035 |         13.326 |         10.198 |       -0.058 |           0.002 |        0.002 |
| sarimax     |              8.382 |             12.339 |              12.582 |               9.222 |             0.135 |                 1.467 | nan        |       nan     |       nan     |        nan     |        nan     |      nan     |         nan     |      nan     |
| stl_xgb     |             13.871 |             18.718 |              15.405 |              11.540 |            -0.355 |                 0.648 | nan        |       nan     |       nan     |        nan     |        nan     |      nan     |         nan     |      nan     |
| stl_ann     |             12.900 |             17.671 |              15.703 |              11.955 |            -0.454 |                 0.241 | nan        |       nan     |       nan     |        nan     |        nan     |      nan     |         nan     |      nan     |
| sarimax_xgb |             10.668 |             15.338 |              12.753 |               9.492 |             0.083 |                 0.381 | nan        |       nan     |       nan     |        nan     |        nan     |      nan     |         nan     |      nan     |

## Phase 3 — Feature ablation (top-2 models)

_Phase 3 in progress._

## Figures

- ![Defaults vs HPO](figures/defaults_vs_hpo.png)
- ![Horizons](figures/horizons.png)
- ![Best model predicted vs actual](figures/best_model_pred_actual.png)
- ![Ablation summary](figures/ablation_summary.png)
