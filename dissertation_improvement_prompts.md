# MSc Dissertation Improvement Prompts

**Project:** Optimising Hospital Supply Chain Demand Forecasting Using Machine Learning — Steve Biko Academic Hospital ED case study
**Author:** Jonathan Lukwichi (22872966), University of Pretoria, MEng Industrial Engineering
**Supervisor:** Dr W.L. Bean
**Repos:**
- `Jonathan-Lukwichi/msc-modelling` (Python pipeline)
- `Jonathan-Lukwichi/latex-disertation` (thesis source)

**Working branch (both repos):** `claude/review-dissertation-repos-UQtqT`

---

## Verified literature foundation

These citations have been triangulated against arxiv, journals, and public repos. Use exactly as written.

| Topic | Citation |
|---|---|
| Original ARIMA+ANN hybrid | Zhang (2003) *Neurocomputing* 50:159–175 |
| Critique of in-sample residual hybrids | Khashei & Bijari (2011) *Applied Soft Computing* 11(2):2664–2675 |
| Modern hybrid evaluation critique | Hewamalage, Bergmeir & Bandara (2021) *IJF* 37(1):388–427 |
| Correctly engineered hybrid (M4 winner) | Smyl (2020) *IJF* 36(1):75–85 |
| Multi-step strategies (Direct vs Recursive) | Bontempi, Ben Taieb & Le Borgne (2013) Springer LNBIP 138:62–77 |
| Empirical multi-step review | Ben Taieb, Bontempi, Atiya & Sorjamaa (2012) *ESwA* 39(8):7067–7083 |
| Adaptive Conformal Inference | Gibbs & Candès (2021) NeurIPS, arXiv:2106.00170 |
| ACI for time series | Zaffran, Féron, Goude, Josse & Dieuleveut (2022) ICML |
| Kernel Mean Matching | Gretton, Smola, Huang, Schmittfull, Borgwardt & Schölkopf (2009) in *Dataset Shift in Machine Learning*, MIT Press |
| Importance-Weighted CV | Sugiyama, Krauledat & Müller (2007) *JMLR* 8:985–1005 |
| RuLSIF density ratio | Yamada, Suzuki, Kanamori, Hachiya & Sugiyama (2013) *Neural Computation* 25(5):1324–1370 |
| Test-Time Training | Sun, Wang, Liu, Miller, Efros & Hardt (2020) ICML, arXiv:1909.13231 |
| DeepAR | Salinas, Flunkert, Gasthaus & Januschowski (2020) *IJF* 36(3):1181–1191 |
| Temporal Fusion Transformer | Lim, Arik, Loeff & Pfister (2021) *IJF* 37(4):1748–1764 |
| MinT reconciliation | Wickramasuriya, Athanasopoulos & Hyndman (2019) *JASA* 114(526):804–819 |
| Temporal hierarchies | Athanasopoulos, Hyndman, Kourentzes & Petropoulos (2017) *EJOR* |
| MASE | Hyndman & Koehler (2006) *IJF* 22(4):679–688 |
| Damped trend | Gardner & McKenzie (1985) *Management Science* 31(10):1237–1246 |
| STL decomposition | Cleveland, Cleveland, McRae & Terpenning (1990) *J. Official Statistics* 6:3–73 |
| MAPE interpretation (practitioner) | Lewis (1982) *Industrial and Business Forecasting Methods*, Butterworth Scientific |
| ED forecasting MAPE benchmark | Silva et al. (2023) *Int. J. Health Plann. Mgmt* 38(4):904–917 — 7 studies all <10% |
| ED forecasting MAPE benchmark | Jiang et al. (2023) *Quant. Imaging Med. Surg.* — 34 of 37 datasets <10% |
| Pandemic concept drift in ED | Susnjak & Maddigan (2023) *EPJ Data Science* 12:11 |
| Continuous training ED | Peláez-Rodríguez et al. (2024) *Comput. Methods Programs Biomed.* |

---

## How to use these prompts

1. Open `msc-modelling` in VS Code, run `claude` in the terminal.
2. Paste **Prompt 0** first (creates branch, scaffolding, baseline snapshot).
3. Then paste Prompts 1 → 15 in order. After each, review the diff, run tests, and approve the commit.
4. If you only have 2 weeks, do **Prompts 0, 1, 2, 4, 7, 8, 13** — biggest defensible improvements.

---

## Prompt 0 — Setup & baseline snapshot

```
Context: msc-modelling repo at /path/to/msc-modelling. Branch: claude/review-dissertation-repos-UQtqT (use this, do NOT push to main).

Task:
1. Verify we are on the correct branch; create it if missing.
2. Run all existing pytest tests and save a transcript to `artefacts/baseline_test_log.txt`. Do not fix failures yet — just snapshot.
3. Add to requirements.txt (do not remove existing entries): hydra-core>=1.3, dvc[s3]>=3.0, mlflow>=2.9, mapie>=0.8, mlforecast>=0.13, hierarchicalforecast>=0.4, statsforecast>=1.7, gluonts[torch]>=0.14, densratio>=0.3, optuna>=3.5, pyarrow>=15.
4. Create empty directories: `src/forecasting/drift/`, `src/forecasting/hybrids/oof/`, `src/forecasting/uq/`, `src/forecasting/disagg/` (move existing __init__.py here if present), `pipeline_configs/` (Hydra root), `notebooks/audit/`.
5. Create `artefacts/leaderboard_canonical.parquet` as an empty pyarrow table with schema: model, family, criterion, seed, val_mape, val_rmse, val_mase, test_mape, test_rmse, test_mase, test_winkler_80, test_coverage_80, h1_mape, h3_mape, h7_mape, source_csv, timestamp.
6. Commit with message "chore: scaffolding for Ch6 refactor (rolling, hybrids, drift, UQ, leaderboard)".

Acceptance: branch exists, requirements installed, directories created, parquet schema verified, one commit pushed.
```

---

## Prompt 1 — Consolidate the five duplicated rolling-forecast functions

```
Context: src/forecasting/ currently has rolling_forecast() reimplemented in five places: models/sarimax.py, models/lstm.py, models/xgboost_m.py, models/ann.py, models/negbin.py — plus standalone copies in scripts/17_final_test.py and scripts/21_uncertainty_quantification.py.

Task:
1. Create src/forecasting/rolling.py with a single class:

   class RollingForecaster:
       def __init__(self, model_factory: Callable[[pd.DataFrame, pd.Series], FittedModel],
                    step_days: int = 7, horizon_days: int = 7,
                    min_train_days: int = 365, window_days: int | None = None,
                    refit_every: int = 1, sample_weight_fn: Callable | None = None):
           ...
       def fit_predict(self, X: pd.DataFrame, y: pd.Series,
                       eval_index: pd.DatetimeIndex) -> pd.DataFrame:
           # returns columns: yhat, lower_80, upper_80, lower_95, upper_95, fold_id

   window_days=None ⇒ expanding window (current behavior).
   window_days=N ⇒ sliding window keeping last N days.
   refit_every=k ⇒ refit base every k weeks instead of every week.

2. Rewrite each of sarimax.py, lstm.py, xgboost_m.py, ann.py, negbin.py so that rolling_forecast is a 5-10 line wrapper that constructs a model_factory closure and delegates to RollingForecaster.

3. Delete the duplicate rolling_arima / rolling_sarimax / rolling_nbglm / rolling_xgboost in scripts/17_final_test.py and have that script call the wrappers instead.

4. Add tests/test_rolling.py covering:
   - expanding vs sliding window train index lengths
   - h=7, step=7 produces exactly 57 folds across the test block (Jan 2025–Jan 2026)
   - no train index ever overlaps fold's eval index
   - sample_weight_fn is applied when provided

5. Run pytest. All existing tests must still pass.

6. Commit with message "refactor: unify rolling-origin into single RollingForecaster class".

Acceptance: net LOC reduced by 200+ lines, all tests green, leaderboard CSVs from re-running 02_arima.py and 06_xgboost.py byte-identical to baseline (deterministic seed=42).
```

---

## Prompt 2 — Add MASE, per-horizon metrics, and canonical leaderboard

```
Context: src/forecasting/metrics.py currently exposes MAPE/MAE/RMSE/R² only. Hyndman & Koehler (2006, IJF 22(4):679–688) recommend MASE as the scale-free alternative when targets contain zeros (17 in our data) and under asymmetric loss.

Task:
1. Extend metrics.py:
   def mase(y_true, y_pred, y_train, seasonality: int = 7) -> float
   def winkler_score(y_true, lower, upper, alpha: float) -> float
   def coverage(y_true, lower, upper) -> float
   def per_horizon_metrics(df_with_horizon_col) -> pd.DataFrame  # rows=h=1..7

2. Add a unified results writer:
   src/forecasting/leaderboard.py
   - append_row(parquet_path, model, family, criterion, seed, val_metrics, test_metrics, per_horizon, source_csv)
   - load_leaderboard(parquet_path) → DataFrame sorted by test_mape ascending
   - to_latex(df, columns=...) → LaTeX longtable string for chap6.tex

3. Modify scripts/10_master_leaderboard.py to write to artefacts/leaderboard_canonical.parquet via append_row, replacing the hard-coded BASE_MODELS dict. Read all *_metrics.csv and *_rmse_metrics.csv files in artefacts/metrics/ and reconcile into the canonical schema.

4. Add an OOD-honesty table writer:
   - per_quarter_table(model_name) using existing test_per_quarter.csv if present, else compute from predictions
   - column "drift_sensitivity" = max(quarterly_mape) − min(quarterly_mape), sorted descending

5. Tests in tests/test_metrics.py: MASE on perfect prediction = 0; MASE on naive seasonal = ~1; Winkler on uncovered point penalises; coverage matches np.mean((y >= lower) & (y <= upper)).

6. Commit with message "feat: MASE + Winkler + canonical leaderboard parquet + OOD honesty table".

Acceptance: leaderboard_canonical.parquet has one row per (model, criterion); to_latex output renders cleanly when included in chap6.tex; MASE numbers added to leaderboard.
```

---

## Prompt 3 — Hydra config-driven runner

```
Task:
1. Create pipeline_configs/ Hydra structure:
   pipeline_configs/
     config.yaml                 # defaults list
     paths.yaml                  # mirror configs/paths.yaml
     splits.yaml                 # mirror configs/split.yaml
     cv/
       expanding.yaml            # window_days: null
       sliding_365.yaml          # window_days: 365
       sliding_450.yaml          # window_days: 450
     model/
       arima.yaml sarimax.yaml negbin.yaml xgboost.yaml ann.yaml lstm.yaml
       hybrid_sarimax_xgb.yaml hybrid_sarimax_lstm.yaml hybrid_lstm_xgb.yaml
       hybrid_stl_xgb.yaml hybrid_stl_ann.yaml hybrid_stl_lstm.yaml
       deepar.yaml direct_xgb.yaml
     hpo/grid.yaml random.yaml optuna.yaml
     stage/hpo.yaml val.yaml test.yaml

2. Create scripts/run.py:
   @hydra.main(config_path="../pipeline_configs", config_name="config")
   def main(cfg):
       # dispatch on cfg.stage and cfg.model.name
       # uses RollingForecaster + leaderboard.append_row
       # writes hydra outputs/ logs + mlflow run
   if __name__ == "__main__":
       main()

3. Keep scripts/01..21 working — they call scripts/run.py with hard-coded overrides under the hood (deprecation shims with a warning).

4. Add MLflow autologging:
   src/forecasting/tracking.py:start_run(model_name, cfg) using mlflow.start_run; log params, metrics, artifacts.

5. Add tests/test_runner.py: smoke test for `python scripts/run.py model=arima stage=val cv=sliding_365` exits 0 and appends to leaderboard.

6. Commit "feat: Hydra runner + MLflow tracking, scripts/01..21 become shims".

Acceptance: `python scripts/run.py model=xgboost stage=val` reproduces previous XGBoost val MAPE 12.02 ± 0.05 with deterministic seed.
```

---

## Prompt 4 — Rebuild residual hybrids with out-of-fold residuals (CRITICAL)

```
Context: src/forecasting/hybrids/residual.py currently fits SARIMAX on the entire 848-day train fold then computes IN-SAMPLE residuals — this is the Zhang (2003) original recipe but it has selection bias documented by Khashei & Bijari (2011, Applied Soft Computing 11(2):2664–2675) and methodologically critiqued by Hewamalage, Bergmeir & Bandara (2021, IJF 37(1):388–427). Smyl (2020, IJF 36(1):75–85) is the canonical correctly-engineered hybrid (ES-RNN M4 winner) — joint estimation, not residual stacking. Our empirical evidence in artefacts/RESULTS.md §4sexies shows LSTM+XGB val MAPE 13.50 > LSTM-alone 12.74 — exactly the bias signature.

Task: build statistically honest residual hybrids.

1. Create src/forecasting/hybrids/oof_residuals.py:

   def build_oof_residuals(base_model_factory, X_train, y_train,
                           n_folds: int = 69, horizon: int = 7) -> pd.DataFrame:
       # Use RollingForecaster with min_train_days=365, step=horizon=7
       # Returns columns: date, y_true, yhat_oof, residual (= y_true - yhat_oof), fold_id
       # These residuals are honest out-of-sample, variance-correct for downstream refiner.

2. Rewrite src/forecasting/hybrids/residual.py:

   class OOFResidualHybrid:
       def __init__(self, base_factory, refiner_factory,
                    standardize_residuals: bool = True,
                    nested_hpo: bool = True,
                    refit_refiner_every: int = 4):   # in weeks
           ...
       def fit(self, X_train, y_train):
           # 1. Build OOF residuals
           # 2. If nested_hpo: 5-fold rolling-origin inside OOF for refiner HPO via Optuna
           # 3. Fit refiner on (X_oof, r_oof) with optimal params
       def predict(self, X_eval, eval_dates):
           # base.predict + refiner.predict, with PI propagation
       def predict_interval(self, X_eval, alpha=0.05):
           # combine base PI variance + refiner residual variance via convolution

3. Both XGB-refiner and LSTM-refiner MUST z-score residuals before fit (eliminate the standardisation inconsistency on lines 110 vs 130 of current residual.py).

4. Rewrite scripts/09_hybrids.py and scripts/11_lstm_xgb_hybrid.py to use OOFResidualHybrid. Compare before/after val MAPE and write to artefacts/metrics/hybrid_oof_comparison.csv with columns: hybrid, base_only_val, in_sample_hybrid_val (OLD), oof_hybrid_val (NEW), Δ.

5. Add tests/test_oof_hybrid.py:
   - assert OOF residual dates do not overlap base training dates for any fold
   - assert OOF residual variance ≥ in-sample residual variance for SARIMAX base (selection-bias smoke test)
   - assert refiner_params are tuned independently of base_params (no inheritance bug from §4sexies)

6. Update artefacts/RESULTS.md §4sexies with the corrected numbers.

7. Commit "fix: residual hybrids use out-of-fold residuals + nested refiner HPO (closes Zhang-2003 selection bias, fixes §4sexies HPO-inheritance bug)".

Acceptance: SARIMAX+LSTM val MAPE improves OR remains within ±0.3pp of 12.10 (the win); LSTM+XGB val MAPE moves from 13.50 closer to LSTM-alone 12.74 (bias removed); all OOF tests pass.
```

---

## Prompt 5 — STL hybrids with weekly re-decomposition

```
Context: src/forecasting/hybrids/stl_hybrid.py decomposes y_train ONCE over 848 days, then uses Gardner-McKenzie damped linear (phi=0.9) to extrapolate trend over 184 val / 396 test days. This is why STL hybrid test MAPE is 20-23% (RESULTS.md §4sexies). The fix is weekly re-decomposition using statsmodels STLForecast (https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.stl.STLForecast.html).

Task:
1. Rewrite src/forecasting/hybrids/stl_hybrid.py:

   class RollingSTLHybrid:
       def __init__(self, refiner_factory,
                    period: int = 7, robust: bool = True,
                    seasonal: int = 13, low_pass: int = 11,
                    horizon: int = 7):
           ...
       def fit_predict_rolling(self, X, y, eval_dates):
           # For each weekly origin t:
           #   decomp = STL(y[:t], period=7, robust=True, seasonal=13, low_pass=11).fit()
           #   trend_fc = STLForecast(y[:t], ARIMA, model_kwargs={"order":(1,1,1)}).fit().forecast(7)
           #   seasonal_fc = tile last seasonal cycle
           #   residual_train = y[:t] - trend - seasonal
           #   refiner.fit(X[:t], residual_train)
           #   r_hat = refiner.predict(X[t:t+7])
           #   yhat = trend_fc + seasonal_fc + r_hat

2. Reference Cleveland, Cleveland, McRae & Terpenning (1990) J. Official Stats 6:3-73 for STL; cite robust=True for the COVID structural break.

3. Replace stl_hybrid usage in scripts/09_hybrids.py and 15_task2_ml_hybrids.py.

4. Add tests/test_stl_rolling.py: verify exactly 57 decompositions for test block; verify seasonal component reproduces lag-7 ACF ≈ 0.49 of input series.

5. Commit "fix: STL hybrids use rolling weekly re-decomposition (STLForecast)".

Acceptance: STL+XGB test MAPE drops from 22% to under 15%; STL+ANN and STL+LSTM also drop.
```

---

## Prompt 6 — Direct multi-output XGBoost (per-horizon strategy)

```
Context: scripts/06_xgboost.py uses recursive single-step. test_per_horizon.csv shows h=1 SARIMAX MAPE 14.50 collapsing by h=5 to 8.22 — exactly the recursive refit-from-Sunday bias. Bontempi et al. (2013) Springer LNBIP 138:62-77 and Ben Taieb et al. (2012) ESwA 39(8):7067-7083 show Direct multi-output dominates on noisy short series. Nixtla mlforecast supports this natively: https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/one_model_per_horizon.html

Task:
1. Create src/forecasting/models/direct_xgb.py wrapping Nixtla mlforecast.MLForecast with max_horizon=7, one model per horizon. Lag features: lag_7, lag_14, lag_21, lag_28; rolling means: 7d, 14d, 28d.

2. Add scripts/06b_direct_xgboost.py mirroring 06_xgboost.py but using DirectXGB. Output: artefacts/metrics/direct_xgb_metrics.csv with per-horizon MAPE.

3. Add a "strategy" column to leaderboard_canonical.parquet: recursive | direct.

4. Reference: cite Bontempi 2013 and Ben Taieb 2012 in the chapter outline comments.

5. Commit "feat: Direct multi-output XGBoost via mlforecast (Bontempi 2013)".

Acceptance: direct_xgb h=1 MAPE < recursive XGBoost h=1 MAPE by at least 1pp; direct_xgb mean test MAPE ≤ recursive XGBoost test MAPE.
```

---

## Prompt 7 — Sliding-window + importance-weighted CV (OOD priority #1)

```
Context: cv.py uses expanding window. Under +18.3% test drift (KS D=0.44), 2022 days dilute the gradient. References: Gretton, Smola, Huang, Schmittfull, Borgwardt & Schölkopf (2009) book chapter "Covariate Shift by Kernel Mean Matching" in Dataset Shift in Machine Learning, MIT Press; Sugiyama, Krauledat & Müller (2007) JMLR 8:985-1005 (IWCV); Yamada, Suzuki, Kanamori, Hachiya & Sugiyama (2013) Neural Computation 25(5):1324-1370 (RuLSIF).

Task:
1. Create src/forecasting/drift/importance_weights.py:
   def kmm_weights(X_train, X_recent, sigma="median", B=1.0) -> np.ndarray
   def rulsif_weights(X_train, X_recent, alpha=0.1) -> np.ndarray  # via densratio package

2. Create src/forecasting/drift/sliding_cv.py extending RollingForecaster:
   def fit_predict(...) accepts window_days (sliding) and weight_method ∈ {None, "kmm", "rulsif"}.
   For weighted models: pass sample_weight to XGBoost/sklearn; multiply MSE per-sample for LSTM/ANN PyTorch loss.

3. Add scripts/22_drift_aware_refit.py:
   For each base model in {xgboost, sarimax, ann, lstm, dow_mean}, run 3 configurations:
     - expanding (baseline)
     - sliding window_days=450
     - sliding window_days=450 + RuLSIF weights
   Output: artefacts/metrics/drift_aware_comparison.csv (model, config, val_mape, test_mape, Δtest).

4. Add pipeline_configs/cv/sliding_450_rulsif.yaml.

5. Tests: assert KMM weights sum to ~n_train, are non-negative, and concentrate mass on dates close to test distribution.

6. Commit "feat: sliding-window CV + KMM/RuLSIF importance weighting (Sugiyama 2007, Gretton 2009, Yamada 2013) — OOD priority #1".

Acceptance: dow_mean test MAPE 17.76 → drops by at least 2pp under sliding+RuLSIF; XGBoost test MAPE 12.61 → drops by at least 0.5pp; report deltas in artefacts/metrics/drift_aware_comparison.csv.
```

---

## Prompt 8 — Adaptive Conformal Inference (OOD priority #2)

```
Context: scripts/21_uncertainty_quantification.py uses static split-conformal — uq_coverage.csv shows actual coverage 89.6% vs nominal 95% (under-coverage from distribution drift). Fix via Adaptive Conformal Inference: Gibbs & Candès (2021) NeurIPS, arXiv:2106.00170, update rule α_{t+1} = α_t + γ(α − 1{y_t ∉ Ĉ_t}). Time-series extension by Zaffran, Féron, Goude, Josse & Dieuleveut (2022) ICML — this is the version MAPIE implements: https://github.com/scikit-learn-contrib/mapie.

Task:
1. Create src/forecasting/uq/aci.py wrapping mapie.regression.MapieTimeSeriesRegressor with method="aci", agg_function="mean".

2. Rewrite scripts/21_uncertainty_quantification.py:
   For each best base model (XGBoost RMSE-best, SARIMAX, OOFResidualHybrid SARIMAX+LSTM), produce:
     - Split conformal (existing, baseline)
     - ACI (Gibbs & Candès 2021)
     - ACI for time series (Zaffran 2022) — γ ∈ {0.001, 0.005, 0.01, 0.05} grid
   Output: artefacts/metrics/uq_coverage_aci.csv with columns (model, method, gamma, coverage_80, coverage_95, mean_width, winkler_80, winkler_95).

3. Generate fig_6_uq_aci.png: rolling coverage over test block for each method (target line at 95%, observed coverage trajectory).

4. Add tests/test_aci.py: synthetic drift case, assert ACI mean coverage > split-conformal coverage when drift injected mid-stream.

5. Commit "feat: Adaptive Conformal Inference via MAPIE (Gibbs-Candès 2021, Zaffran 2022) — OOD priority #2".

Acceptance: ACI test coverage@95 ≥ 93% (currently 89.6% with split-conformal); Winkler@95 not more than 15% wider than split-conformal.
```

---

## Prompt 9 — MinT hierarchical reconciliation across 7 specialties

```
Context: scripts/15_task2_ml_hybrids.py produces 7 independent specialty forecasts. artefacts/metrics/task2_sum_consistency.csv shows they don't sum to G1 total. Wickramasuriya, Athanasopoulos & Hyndman (2019) JASA 114(526):804-819 (MinT) reconcile via residual covariance. Implementation: Nixtla hierarchicalforecast.methods.MinTrace.

Task:
1. Create src/forecasting/reconciliation.py with:
   def build_hierarchy(g1_forecast, specialty_forecasts: dict) → HierarchicalData
   def reconcile_mint(forecasts, residuals_in_sample, method="mint_shrink") → reconciled DataFrame

2. Add scripts/23_reconcile_specialties.py:
   - Load per-specialty val/test forecasts from existing per_specialty outputs
   - Load in-sample residuals for covariance estimation
   - Apply MinTrace(method="mint_shrink"); also try "mint_cov", "ols", "wls_struct"
   - Output: artefacts/metrics/task2_reconciled_metrics.csv (rows = specialty + total, cols = base MAPE, MinT MAPE, Δ)

3. Cite Athanasopoulos, Hyndman, Kourentzes & Petropoulos (2017) EJOR for temporal hierarchies as a future-work pointer (daily/weekly/monthly coherence).

4. Tests: assert reconciled forecasts sum exactly to total within float tolerance; assert MinT(MAPE) ≤ base(MAPE) on at least 5 of 7 specialties.

5. Commit "feat: MinT reconciliation across specialties (Wickramasuriya 2019) via hierarchicalforecast".

Acceptance: task2_reconciled_metrics.csv shows mean specialty MAPE reduced by ≥ 0.5pp; sum-consistency residual reduced by ≥ 90%.
```

---

## Prompt 10 — DeepAR multi-task across specialties (alternative architecture)

```
Context: small/noisy specialty series (Maternity weekly MAPE 54%, Psychiatry 77%) need cross-series signal sharing. Salinas, Flunkert, Gasthaus & Januschowski (2020) IJF 36(3):1181-1191 — DeepAR with NegativeBinomialOutput matches our VMR=3.49 overdispersion. Implementation: from gluonts.torch.distributions import NegativeBinomialOutput.

Task:
1. Create src/forecasting/models/deepar.py wrapping gluonts.torch.model.deepar.DeepAREstimator with:
   - distr_output=NegativeBinomialOutput()
   - context_length=28, prediction_length=7
   - specialty_id as static categorical feature
   - calendar + weather as dynamic real features

2. Add scripts/24_deepar_multitask.py: train one DeepAR jointly on all 7 specialty series + G1 total; produce per-series val/test MAPE and per-quantile predictions for ACI integration.

3. Add to leaderboard_canonical.parquet with strategy="multitask_deepar".

4. Tests: assert NegBin output dispersion estimate ≈ empirical VMR within 30%; assert model trains under 30 min on CPU for 8 series × ~850 days.

5. Commit "feat: DeepAR multi-task with NegBin likelihood (Salinas et al. 2020)".

Acceptance: Maternity and Psychiatry weekly MAPE drop by at least 20pp absolute; G1 daily MAPE within ±1pp of standalone XGBoost.
```

---

## Prompt 11 — Test-time training for deep models (OOD bonus)

```
Context: Sun, Wang, Liu, Miller, Efros, Hardt (2020) ICML, arXiv:1909.13231 — Test-Time Training improves on distribution-shift benchmarks. We will not claim a specific recovery percentage; we will measure it.

Task:
1. Create src/forecasting/drift/test_time_training.py:
   def ttt_finetune(model, recent_X, recent_y, n_steps=5, lr=1e-4) → fine-tuned model

2. Integrate into RollingForecaster for LSTM and ANN: at each weekly origin, after the standard refit, take n_steps SGD steps on last 28 days at lr=1e-4 with main loss only (no auxiliary task — full TTT requires self-supervised pretext, defer to future work).

3. Add config flag in pipeline_configs/cv/ttt.yaml.

4. Output: artefacts/metrics/ttt_comparison.csv (LSTM, ANN with/without TTT, val/test MAPE).

5. Commit "feat: test-time training for deep models (Sun et al. 2020 simplified variant)".

Acceptance: LSTM/ANN test MAPE strictly reduced or unchanged (no regression); report measured delta in chap6.tex as empirical contribution.
```

---

## Prompt 12 — Per-model figure pack generator

```
Task: produce a standard 4-panel PNG per model for chap6.tex.

1. Create src/forecasting/plotting/model_card.py:
   def render_model_card(model_name, predictions_df, residuals_df, output_path):
       # 4-panel matplotlib figure:
       # (a) forecast vs actual time series with 80% PI band, full test block
       # (b) residual ACF + QQ-plot (2 small panels stacked) + residual vs predicted scatter
       # (c) test MAPE bar chart by horizon h=1..7
       # (d) error heatmap: rows=DoW (Mon-Sun), cols=month (Jan-Dec), cell=mean |residual|

2. Add scripts/render_all_model_cards.py iterating over every model in leaderboard_canonical.parquet, writing to artefacts/figures/model_cards/{model_name}.png.

3. Output: 14-20 PNGs at 300 DPI for direct LaTeX inclusion.

4. Commit "feat: per-model 4-panel diagnostic figure pack".

Acceptance: every PNG renders without overlap, axis labels are 11pt, files under 800KB each.
```

---

## Prompt 13 — Canonical Ch 6 LaTeX results section

```
Context: chap6.tex is currently a stub. Use leaderboard_canonical.parquet and the figure pack to write a publication-ready chapter.

Task: rewrite latex-disertation/chap6.tex on branch claude/review-dissertation-repos-UQtqT (the latex repo, separate from msc-modelling). Sections must follow this order:

6.1 Experimental Setup
  - Reference Ch 5 §5.5.2 split: 848/184/396 days, KS D=0.44 (test is OOD by design)
  - Rolling-origin expanding CV (69 folds, h=7) — reference Hyndman & Athanasopoulos (2021) Forecasting: Principles and Practice §5.10
  - HPO: Optuna TPE for LSTM (30 trials), grid for XGBoost, random for ANN — justified by hpo_comparison results
  - Reproducibility: seed=42, DVC tracking, MLflow run IDs

6.2 Parametric Baselines
  - ARIMA(1,1,1) (auto_arima); SARIMAX(1,1,1)(0,1,1)_7; NB-GLM
  - Table 6.1 from leaderboard_canonical.parquet (parametric subset)
  - Reference Box & Jenkins (1970); cite NB-GLM justification = VMR=3.49 from Ch 5

6.3 Machine Learning and Deep Learning Models
  - XGBoost, ANN (PyTorch MLP), LSTM (PyTorch)
  - Insert SHAP figure (existing fig_6_6_xgb_shap.png)
  - Insert hpo_comparison.csv as Table 6.2

6.4 Hybrid Models
  - Critique in-sample residual approach citing Khashei & Bijari (2011) Appl. Soft Comput. 11(2):2664-2675 and Hewamalage, Bergmeir & Bandara (2021) IJF 37(1):388-427
  - Describe OOFResidualHybrid implementation and rolling STL via STLForecast
  - Table 6.3: hybrid_oof_comparison.csv (in-sample vs OOF columns)
  - Reference Smyl (2020) IJF 36(1):75-85 (ES-RNN) as the canonical correctly-engineered hybrid

6.5 Direct Multi-output Strategy
  - Recursive vs Direct, cite Bontempi, Ben Taieb & Le Borgne (2013) Springer LNBIP 138:62-77 and Ben Taieb, Bontempi, Atiya & Sorjamaa (2012) Expert Systems with Applications 39(8):7067-7083
  - Table 6.4: per-horizon recursive vs direct

6.6 Ensembles
  - Top-3, inverse-MAPE, optimal convex, ridge stacking
  - Table 6.5 from ensembles.csv

6.7 Uncertainty Quantification
  - Split conformal baseline (Vovk et al. 2005)
  - Adaptive Conformal Inference: Gibbs & Candès (2021) NeurIPS, Zaffran et al. (2022) ICML
  - Table 6.6 from uq_coverage_aci.csv
  - Figure 6.UQ rolling coverage trajectory

6.8 Out-of-Distribution Generalisation
  - Frame test set as drift validation; cite Susnjak & Maddigan (2023) EPJ Data Science 12:11 (pandemic concept drift in ED flows)
  - Table 6.7 OOD honesty table (sort by drift_sensitivity)
  - Sliding-window + RuLSIF results from drift_aware_comparison.csv
  - References: Sugiyama et al. (2007) JMLR; Gretton et al. (2009) MIT Press book chapter; Yamada et al. (2013) Neural Computation

6.9 Hierarchical Reconciliation
  - Wickramasuriya, Athanasopoulos & Hyndman (2019) JASA 114(526):804-819 — MinT
  - Table 6.8 task2_reconciled_metrics.csv
  - Cite Athanasopoulos, Hyndman, Kourentzes & Petropoulos (2017) EJOR for future temporal-hierarchy work

6.10 Final Leaderboard
  - Full Table 6.9 from leaderboard_canonical.parquet sorted by test_mape ASC
  - Include MASE column; cite Hyndman & Koehler (2006) IJF 22(4):679-688 for MASE
  - Note Lewis (1982) MAPE interpretation scale (<10 excellent, 10-20 good); flag this is a practitioner textbook reference not peer-reviewed
  - Reference benchmarks: Silva et al. (2023) IJHPM 38(4):904-917 (7 ED studies, all errors <10%); Jiang et al. (2023) Quant. Imaging Med. Surg. (34/37 datasets MAPE<10%)

6.11 Concluding Remarks

After writing, run pdflatex+biber+pdflatex+pdflatex. Fix any undefined references or overfull boxes.

Commit on the latex-disertation branch with message "feat(chap6): full Model Development chapter with OOF hybrids, ACI UQ, MinT reconciliation, OOD analysis".

Acceptance: chap6.tex compiles cleanly; all citations resolve; all referenced figures/tables exist in artefacts/.
```

---

## Prompt 14 — Ch 7 Case Study Application LaTeX

```
Context: chap7.tex is a stub. Operations-research integration: per-specialty roster, inventory (s,S), bed planning.

Task: rewrite latex-disertation/chap7.tex on branch claude/review-dissertation-repos-UQtqT.

7.1 From Forecasts to Operational Decisions
  - Three-layer framing from DJANGO_BUILD_PLAN.md
  - Layer 1 demand forecasts feed Layer 2 (hourly disaggregation) feeds Layer 3 (OR optimisation)

7.2 Hourly Disaggregation (Layer 2)
  - Day-type proportion curves from Ch 5 §5.4 (Day 41 / Evening 41 / Night 18)
  - Note current msc-modelling repo has only stub disagg/__init__.py; specify the disaggregation algorithm

7.3 Inventory Planning ((s,S) Policy)
  - Cite Silver, Pyke & Peterson (1998) Inventory Management
  - Monte Carlo S calibration uses ACI 95% upper bound from §6.7
  - Worked example for 3 high-volume consumables

7.4 Staff Roster Optimisation (PuLP IP)
  - Decision variables, constraints from NDoH nurse ratios
  - 45h/week, 11h rest, weekly MAPE 5.89% from leaderboard supports weekly roster window
  - Reference Bard & Purnomo (2005) Health Care Manag Sci 8(3):173-184 for nurse scheduling IP

7.5 Bed Capacity Planning
  - Monthly MAPE 3.47% supports monthly bed-block decisions
  - Reference Gorunescu, McClean & Millard (2002) for ED bed planning baseline

7.6 Deployment Architecture
  - Reference DJANGO_BUILD_PLAN.md Django app; .pkl ModelPackage from deploy.py
  - Single-user local-first design rationale

7.7 Concluding Remarks

Commit on latex repo "feat(chap7): Case Study Application with three-layer OR integration".

Acceptance: compiles cleanly; cross-references to Ch 6 leaderboard and Ch 8 discussion resolve.
```

---

## Prompt 15 — Ch 8 Discussion LaTeX

```
Task: rewrite latex-disertation/chap8.tex on branch claude/review-dissertation-repos-UQtqT.

8.1 Position in Literature
  - Compare 12% val MAPE / 13-17% test MAPE to:
    - Silva et al. (2023) Int. J. Health Plann. Mgmt 38(4):904-917 — 7 daily-ED studies, all MAPE<10% on validation
    - Jiang et al. (2023) Quant. Imaging Med. Surg. — 34 of 37 datasets MAPE<10%
  - Explain: validation matches the literature band; test diverges due to documented +18.3% post-2024 drift (KS D=0.44)

8.2 The 10% MAPE Wall is a Noise Floor, Not a Modelling Failure
  - At CV=23% on n=848 days, theoretical irreducible noise floor 6-10% (cite Makridakis & Hibon 2000 M3 competition)
  - Our 12% val MAPE sits 2pp from the floor
  - Reference Hewamalage, Bergmeir & Bandara (2021) IJF on what is achievable

8.3 OOD Honesty
  - Cite Susnjak & Maddigan (2023) EPJ Data Science 12:11 (pandemic-induced concept drift in ED flows)
  - Reference Peláez-Rodríguez et al. (2024) Comput. Methods Programs Biomed. (continuous training framing)
  - Frame val MAPE as "stable-regime lower bound" and test MAPE as "drift upper bound"
  - dow_mean test MAPE 17.76 vs SARIMAX 13.11 quantifies the cost of NOT retraining — operational message

8.4 NHI Policy Implications
  - Tertiary hospitals (Steve Biko-class): XGBoost + ACI deployment feasible
  - Primary/secondary clinics without ML capacity: weekly-refit DoW-mean defensible floor at ~14% MAPE (post our sliding-window improvement)
  - Translate to operational metric: forecast error band feeds 95% PI for safety-stock setting

8.5 Methodological Contributions
  - OOF residual hybrid recipe (fixes Zhang 2003 selection bias documented by Khashei & Bijari 2011)
  - ACI integration for under-resourced drift settings (Gibbs & Candès 2021, Zaffran 2022)
  - Multi-task NegBin DeepAR for low-count specialty series (Salinas 2020)

8.6 Limitations
  (a) 848-day training window short relative to literature norms (≥3 years preferred)
  (b) external_signals/ collected but not integrated (Eskom, NICD)
  (c) no surveillance covariates (syndromic, vaccination uptake)
  (d) ED arrivals only — does not model admission/discharge tension
  (e) single-site case study; external generalisability unconfirmed

8.7 Future Work
  - Direct multi-output across the full model suite
  - Temporal MinT (Athanasopoulos et al. 2017) for daily/weekly/monthly coherence
  - Online learning + continual training pipeline
  - Surveillance-feature integration (NICD bulletins)
  - Multi-site validation across NHI rollout pilot hospitals

Commit on latex repo "feat(chap8): Discussion with OOD honesty, NHI policy, limitations, future work".

Acceptance: all citations resolve; argument flows from leaderboard numbers through discussion to policy.
```

---

## Execution sequence and time budget

| # | Prompt | Touches | Repo | Time |
|---|---|---|---|---|
| 0 | Scaffolding | both | msc-modelling | 30 min |
| 1 | RollingForecaster | code | msc-modelling | 2 h |
| 2 | MASE + leaderboard | code | msc-modelling | 2 h |
| 3 | Hydra runner | code | msc-modelling | 3 h |
| **4** | **OOF residual hybrids** | code | msc-modelling | **4 h — critical** |
| 5 | Rolling STL hybrids | code | msc-modelling | 2 h |
| 6 | Direct multi-output | code | msc-modelling | 2 h |
| **7** | **Sliding window + IW** | code | msc-modelling | **3 h — OOD #1** |
| **8** | **ACI via MAPIE** | code | msc-modelling | **2 h — OOD #2** |
| 9 | MinT reconciliation | code | msc-modelling | 2 h |
| 10 | DeepAR multi-task | code | msc-modelling | 3 h |
| 11 | Test-time training | code | msc-modelling | 1 h |
| 12 | Figure pack | code | msc-modelling | 1 h |
| **13** | **Ch 6 LaTeX** | text | latex-disertation | **4 h** |
| 14 | Ch 7 LaTeX | text | latex-disertation | 2 h |
| 15 | Ch 8 LaTeX | text | latex-disertation | 2 h |

**Total: ~35 hours.** Two-week minimum path: **0, 1, 2, 4, 7, 8, 13** (~17 hours) — ships statistically corrected hybrids, both OOD interventions, and a defensible Ch 6 grounded in verified literature.

---

## Public-repo references for code triangulation

| Recommendation | Library / Class | URL |
|---|---|---|
| Direct multi-output | Nixtla `mlforecast` `MLForecast(max_horizon=H)` | https://nixtlaverse.nixtla.io/mlforecast/ |
| OOF residual stacking | sklearn `StackingRegressor(cv=...)` | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html |
| Adaptive Conformal Inference | MAPIE `MapieTimeSeriesRegressor(method="aci")` | https://github.com/scikit-learn-contrib/mapie |
| MinT reconciliation | Nixtla `hierarchicalforecast.methods.MinTrace` | https://github.com/Nixtla/hierarchicalforecast |
| DeepAR + NegBin | GluonTS `DeepAREstimator(distr_output=NegativeBinomialOutput())` | https://github.com/awslabs/gluonts |
| Sliding window CV | sklearn `TimeSeriesSplit(max_train_size=N)` | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html |
| Rolling STL decomposition | `statsmodels.tsa.forecasting.stl.STLForecast` | https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.stl.STLForecast.html |
| RuLSIF density ratio | `densratio` Python package | https://pypi.org/project/densratio/ |

---

*Generated for Jonathan Lukwichi's MSc dissertation — University of Pretoria, May 2026.*
