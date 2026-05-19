# Chapter 6 — Model Development and Evaluation: Build Plan

**Project:** MSc Thesis — Optimising Hospital Supply Chain Demand Forecasting Using Machine Learning
**Author:** Jonathan Lukwichi (University of Pretoria)
**Plan date:** 2026-05-18 (Monday)
**Target delivery:** 2026-05-22 (Friday)
**Window:** 4 working days (Mon to Fri)
**Status:** Single source of truth. Supersedes `chapter6_baselines_plan.md` and `CHAPTER_6_MODELLING_BUILD_PLAN.md`.

**Scope:** Forecasting models and their evaluation. Not in scope: (s, S) inventory module, integer-programming workforce scheduler, Streamlit cloud app, Chapter 7. Those are later work.

---

## Table of contents

1. [Pre-flight](#1-pre-flight)
2. [Settled decisions inventory](#2-settled-decisions-inventory)
3. [Repository structure](#3-repository-structure)
4. [Data sources and splits](#4-data-sources-and-splits)
5. [Design language](#5-design-language)
6. [Step 1 — Reference floor (naive + DoW mean)](#6-step-1--reference-floor)
7. [Step 2 — ARIMA baseline](#7-step-2--arima-baseline)
8. [Step 3 — SARIMAX + NB GLM (parallel)](#8-step-3--sarima--nb-glm-parallel)
9. [Step 4 — Feature engineering (ML models only)](#9-step-4--feature-engineering-ml-models-only)
10. [Step 5 — Four-method consensus selection](#10-step-5--four-method-consensus-selection)
11. [Step 6 — XGBoost, ANN, LSTM standalone](#11-step-6--xgboost-ann-lstm-standalone)
12. [Step 7 — Hybrids (3 residual + 3 STL)](#12-step-7--hybrids-3-residual--3-stl)
13. [Step 8 — Task 2 per-specialty](#13-step-8--task-2-per-specialty)
14. [Step 9 — Layer 2 hourly disaggregation](#14-step-9--layer-2-hourly-disaggregation)
15. [Step 10 — Consolidated leaderboard (Table 6.1)](#15-step-10--consolidated-leaderboard-table-61)
16. [Step 11 — Pre-COVID secondary analyses](#16-step-11--pre-covid-secondary-analyses)
17. [Step 12 — OOD test pass](#17-step-12--ood-test-pass)
18. [Step 13 — Verification harness](#18-step-13--verification-harness)
19. [4-day execution schedule](#19-4-day-execution-schedule)
20. [Risk register and mitigations](#20-risk-register-and-mitigations)
21. [Defer matrix](#21-defer-matrix)
22. [Always-ship floor](#22-always-ship-floor)
23. [LaTeX outline](#23-latex-outline)
24. [Acceptance checklist](#24-acceptance-checklist)
25. [Crosswalk to chapters 3, 4, 5](#25-crosswalk-to-chapters-3-4-5)

---

## 1. Pre-flight

All paths are now known and pinned. **No open decisions.** NB likelihood handling and engineering/consensus scope were settled by reading Chapter 5 §5.7 and Chapter 3 §3.4.3 — see §2 below.

| # | Resource | Absolute path |
|---|---|---|
| 1 | Modelling repo | `c:\Users\BIBINBUSINESS\OneDrive\Desktop\msc-thesis--modelling-and-evaluation\` |
| 2 | Upstream ETL pipeline (raw → G1..G4) | `C:\Users\BIBINBUSINESS\OneDrive\Desktop\data transformation pipline\healthforecast_pipeline\healthforecast_pipeline\` |
| 3 | EDA + joined dataset outputs (Ch4 / Ch5 artefacts) | `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\` |
| 4 | LaTeX figures sync target | `c:\Users\BIBINBUSINESS\OneDrive\Desktop\latex code\figures\ch6\` |

The modelling repo reads from path 3 (joined CSVs already produced by the EDA pipeline). Path 2 is the ETL only; not invoked at modelling time. All paths overridable via `configs/paths.yaml`.

---

## 2. Settled decisions inventory

Every entry is traceable to a specific section in Chapter 3, 4, or 5. Nothing in the code may contradict this list without first revising the corresponding chapter section.

### 2.1 Models and likelihoods

| Decision | Source |
|---|---|
| 11 candidate models: ARIMA, SARIMAX, LSTM, XGBoost, ANN + 6 hybrids (LSTM+XGBoost, SARIMAX+XGBoost, SARIMAX+LSTM, STL+XGBoost, STL+ANN, STL+LSTM) | Ch3 §3.5 |
| **Two parallel parametric baselines per §5.7:** (a) SARIMAX with Gaussian likelihood as the time-series baseline; (b) Negative Binomial GLM regression with the §5.2.5 exogenous block as the headline parametric likelihood. **Not** an NB-likelihood SARIMAX — the thesis prescribes them as side-by-side models, not nested | Ch5 §5.7, §5.2.1 |
| SARIMAX template `(p, 1, q)(P, 1, Q)_7`; p, q, P, Q ∈ {0, 1, 2}; d = 1 and D = 1 both fixed; AIC-selected | Ch5 §5.2.2 |
| Normal likelihood retained as **sensitivity** check against NB (AIC gap 0.4%) | Ch5 §5.2.1 |
| No log or Box-Cox transform (λ ≈ 0.85, close to identity) | Ch5 §5.2.1 |
| Tree-based quantile loss as fallback if parametric likelihood proves inadequate | Ch5 §5.7 |
| Residual hybrid recipe: `ŷ = f_A(x) + f_B(residuals_of_A)` (Zhang 2003, Alg 6) | Ch3 §3.5.4 |
| STL hybrid recipe: STL with period s=7, robust; trend extrapolated linearly, seasonal via seasonal-naive; ML model fitted on residual `R_t` | Ch3 §3.5.4 Alg 7 |
| Hybrid selection loss: MAPE on validation block | Ch3 §3.5.4 |

### 2.2 Task 1 feature inventory (§5.2.5)

**10 features**, used directly by SARIMAX, NB GLM, and ARIMA-with-exogenous variants. No engineering/consensus expansion for these baselines.

1. Day of Week (categorical, 7 levels → 6 dummies)
2. Mean temperature (continuous, standardised on train)
3. Maximum wind speed (continuous, standardised on train)
4. Weekend
5. Long Weekend
6. School Holiday
7. Public Holiday
8. Winter Holiday
9. Festive Season
10. Near Holiday

Five §5.2.3-significant calendar effects **explicitly excluded** from the inventory (Year End, Year Start, December, Day Before Holiday, Month End Period). They are flagged only for operational interval widening in §5.5.2. Year feature explicitly excluded.

### 2.3 Feature engineering and selection (for ML models only)

| Decision | Source |
|---|---|
| Engineered feature space: 50–100 derived features from the §5.2.5 raw 10 (cyclical encodings, Fourier harmonics, lags, rolling stats, interactions) | Ch3 §3.4.2 |
| Four-method consensus selection (Algorithm 1) on the engineered space: Dummy / RF permutation / Lasso / GBM gain; vote threshold τ = 2 of 4 | Ch3 §3.4.3 |
| Engineering + consensus apply to **XGBoost, ANN, LSTM, and the 6 hybrids only.** SARIMAX / ARIMA / NB GLM use the §5.2.5 raw 10 directly | Ch3 §3.4.3 (final paragraph distinguishes raw-19 §5.2.5 triangulation from engineered-space Alg 1) |

### 2.4 Task 2 per-specialty (§5.3)

| Decision | Source |
|---|---|
| 5 daily-resolution specialties (Medicine 73.3%, Orthopaedics 17.1%, Surgery, Paediatrics, Gynaecology) | Ch5 §5.3.1 |
| 2 weekly-resolution specialties (Maternity 0.15/d, Psychiatry 0.10/d) | Ch5 §5.3.1 |
| Independent per-specialty models (max cross-correlation 0.224 rules out multi-output) | Ch5 §5.3.2 |
| Per-specialty calendar and weather coefficients **mandatory** (Surgery / Orthopaedics sign reversals rule out shared coefficients) | Ch5 §5.3.3, §5.5.1 |
| Surgery sign-reversal magnitudes: +41.0% weekend, +32.1% long weekend, +26.2% public holiday | Ch5 §5.3.3 |
| Weather features per specialty: Medicine (temp + wind), Orthopaedics (temp), Paediatrics (wind), Maternity (wind); Surgery / Gynaecology / Psychiatry weather-flat | Ch5 §5.3.3 |
| Forecast share-of-header, multiply by Task 1 forecast at delivery; sum-consistency reported **post hoc** not enforced at training | Ch4 §4.4.4, Ch3 §3.5.10 |

### 2.5 Layer 2 hourly disaggregation (§5.4)

| Decision | Source |
|---|---|
| In scope for Chapter 6 (§3.5.8 is methodology) | Ch3 §3.5.8 |
| Disaggregation rule: `λ_{h,d} = ŷ_d · p_{h,w(d)}` | Ch3 eq 3.17 |
| Two day-type curves: Weekday vs Weekend-or-Holiday | Ch5 §5.4.2 |
| Shift shares (post-COVID): Day 41.0% (07–14), Evening 40.8% (15–22), Night 18.2% (23–06) | Ch5 §5.4 |
| Per-hour refinement only for 09:00–15:00 (seven core peak-band hours, year-to-year rel spread ≤ 10%); other hours use uniform within-shift split | Ch5 §5.4.3 |

### 2.6 Data and splits

| Decision | Source |
|---|---|
| Authoritative target: `patient_count` header total (exact = arrivals_NH + arrivals_AH) | Ch4 §4.4.4 |
| Specialty sub-tallies reconcile to header on 77.2% of days; usable only for proportional analysis | Ch4 §4.4.4 |
| 17 zero-arrival days flagged `is_zero_day = 1`, MCAR, kept, excluded from non-zero subset (n = 2,423) | Ch4 §4.4.1 |
| G1 daily 2,440 × 48; G2 hourly 58,560 × 43; G3 clinical daily 2,440 × 149; G4 clinical hourly 58,560 × 152 | Ch4 §4.5 |
| Train: 2022-03-01 to 2024-06-30, 853 days, mean 58.4, SD 13.7 | Ch5 §5.5.2 |
| Validation: 2024-07-01 to 2024-12-31, 184 days, mean 60.6, SD 9.9 | Ch5 §5.5.2 |
| Test: 2025-01-01 to 2026-01-31, 396 days, mean 69.1, SD 12.0 | Ch5 §5.5.2 |
| Test out-of-distribution: KS D = 0.44 vs train (p = 7.3×10⁻⁴⁷); test mean +18.3% above train | Ch5 §5.5.2 |
| Pre-COVID (2019-05-01 to 2020-02-29, 302 d) and during-COVID (2020-03-01 to 2022-03-31, 724 d) excluded from training; retained for sensitivity, calendar stability, historical context | Ch5 §5.5.2 |

### 2.7 Validation, HPO, metrics

| Decision | Source |
|---|---|
| Rolling-origin expanding-window CV; horizon h = 7 days; refit weekly inside train block | Ch3 §3.6.1 |
| ARIMA: ADF + KPSS + AIC via `pmdarima` stepwise | Ch3 §3.5.2 |
| SARIMAX: grid (p,q,P,Q ∈ {0,1,2}) + AIC, d=D=1 fixed | Ch5 §5.2.2 |
| XGBoost: grid search (n_estimators 100–500, max_depth 3–8, LR 0.01–0.3, subsample 0.7–1.0) | Ch3 §3.5.9 |
| ANN: random search 20 iterations (1–2 hidden layers, 64–256 units, dropout 0.1–0.4, LR 0.0005–0.01, batch 16–64) | Ch3 §3.5.9 |
| LSTM: Optuna TPE 30 trials (lookback 14–30, units 64–256, dropout 0.1–0.4, LR 0.0005–0.01, batch 16–64) | Ch3 §3.5.9 |
| Metrics: MAPE primary (ranking), MAE, RMSE, R²; MAPE < 10% conventionally excellent | Ch3 §3.6.2 |
| Test block touched **exactly once** for final OOD pass; no model selection on test | Ch5 §5.5.2 |

---

## 3. Repository structure

```
msc-thesis--modelling-and-evaluation/
├─ pyproject.toml
├─ requirements.txt
├─ README.md                       # how to run, seed = 42, reproducibility note
├─ .gitignore
├─ configs/
│  ├─ paths.yaml                   # data + output locations
│  ├─ split.yaml                   # fixed train/val/test dates
│  ├─ features_task1.yaml          # §5.2.5 raw 10-feature inventory (SARIMAX/NB)
│  ├─ features_task2.yaml          # per-specialty exogenous blocks
│  ├─ engineering.yaml             # §3.4.2 expansion recipe (ML models)
│  ├─ consensus.yaml               # §3.4.3 Algorithm 1 parameters
│  ├─ hpo_ranges.yaml              # mirrors Ch3 §3.5.9 Table 3.1
│  └─ models.yaml                  # active model list + flags
├─ data/
│  ├─ raw/                         # symlinks or read-only copies of G1..G4
│  └─ processed/                   # feature matrices, splits
├─ src/forecasting/
│  ├─ __init__.py
│  ├─ io.py                        # load G1..G4, freeze splits
│  ├─ features.py                  # raw 10 (Task 1) + per-specialty (Task 2)
│  ├─ engineering.py               # §3.4.2 expand to 50–100
│  ├─ consensus.py                 # §3.4.3 Algorithm 1
│  ├─ cv.py                        # rolling-origin generator (expanding, h=7)
│  ├─ metrics.py                   # MAPE, MAE, RMSE, R², per-horizon
│  ├─ models/
│  │   ├─ naive.py                 # naive_yest, naive_seasonal, dow_mean
│  │   ├─ arima.py                 # pmdarima auto_arima
│  │   ├─ sarima.py                # statsmodels SARIMAX (Gaussian)
│  │   ├─ negbin.py                # sm.GLM Negative Binomial (headline parametric)
│  │   ├─ xgboost_m.py
│  │   ├─ ann.py                   # Keras MLP
│  │   └─ lstm.py                  # Keras LSTM
│  ├─ hybrids/
│  │   ├─ residual.py              # Zhang Alg 6: LSTM+XGB, SARIMAX+XGB, SARIMAX+LSTM
│  │   └─ stl_hybrid.py            # STL+XGB, STL+ANN, STL+LSTM
│  ├─ hpo/
│  │   ├─ selector.py              # dispatch by family
│  │   ├─ grid.py
│  │   ├─ random.py
│  │   └─ tpe.py                   # Optuna
│  └─ disagg/
│      └─ shift_share.py           # Layer 2 (§5.4)
├─ scripts/
│  ├─ 01_build_features.py
│  ├─ 02_engineer_and_select.py    # §3.4.2 + §3.4.3 (ML only)
│  ├─ 03_tune.py                   # --model X
│  ├─ 04_train_eval.py             # rolling-origin on train + val
│  ├─ 05_task2_run.py              # per-specialty loop
│  ├─ 06_disaggregate.py           # Layer 2
│  ├─ 07_final_test.py             # single OOD pass on test block
│  ├─ 08_compare.py                # leaderboard + figures
│  └─ 09_verify.py                 # verification harness (§18)
├─ artefacts/
│  ├─ models/                      # *.pkl / *.h5
│  ├─ predictions/                 # parquet, one per (model, task, fold)
│  ├─ metrics/                     # csv per run + master leaderboard.csv
│  └─ figures/                     # png/pdf for LaTeX
└─ tests/
   ├─ test_io.py
   ├─ test_features.py
   ├─ test_cv.py
   ├─ test_metrics.py
   ├─ test_consensus.py            # asserts §5.2.5 survivors are retained
   └─ test_hybrids.py              # asserts no test-data leak
```

### Python dependencies

```
numpy
pandas
pyarrow
scikit-learn
statsmodels
pmdarima
xgboost
tensorflow==2.15            # CPU build pinned for Windows
optuna
pyyaml
matplotlib
seaborn
joblib
tqdm
shap
pytest
```

---

## 4. Data sources and splits

### 4.1 Primary modelling inputs (joined G1..G4, post-COVID + pre-COVID merged into one frame each)

Located under `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\`. These are the v2.1 "pure" joined CSVs produced by the EDA pipeline and frozen for Chapter 5. **Modelling code reads these directly.**

| Role | File (absolute path) |
|---|---|
| Task 1 (daily total) | `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\01_joined_daily_demand\G1_pure_daily_demand.csv` |
| Layer 2 (hourly) | `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\02_joined_hourly_demand\G2_pure_hourly_demand.csv` |
| Task 2 (specialty daily) | `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\03_joined_clinical_daily\G3_pure_clinical_daily.csv` |
| Task 2 (specialty hourly) | `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\04_joined_clinical_hourly\G4_pure_clinical_hourly.csv` |

- **Task 1:** target column `total_daily_arrivals` (header total per §4.4.4). Filter `is_zero_day == 0`.
- **Task 2:** 7 specialty count columns. Forecast share-of-header, multiply by Task 1 forecast at delivery (§4.4.4).
- **Layer 2:** hourly counts; shift/hour proportions already computed upstream (see §4.3).

### 4.2 Upstream raw + transform sources (read-only; not invoked at modelling time)

**Raw daily registers** (xlsx, monthly files 2019-04 to 2026-01):
`C:\Users\BIBINBUSINESS\OneDrive\Desktop\data transformation pipline\healthforecast_pipeline\healthforecast_pipeline\Casualty_Daily_Register_Dataset\`

**ETL pipeline code** (extract / transform / validate; produced the ml-ready CSVs):
`C:\Users\BIBINBUSINESS\OneDrive\Desktop\data transformation pipline\healthforecast_pipeline\healthforecast_pipeline\src\`

**Hospital + external source CSVs** (consumed by the EDA join):
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\hospital\Steve_Biko_Daily_Dataset.csv`
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\hospital\Steve_Biko_Hourly_Dataset.csv`
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\hospital\clinical_reasons_daily.csv`
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\hospital\clinical_reasons_hourly.csv`
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\external\calendar_features_2019_2026.csv`
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\external\pretoria_weather_daily_2019_2026.csv`
- `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\external\pretoria_weather_hourly_2019_2026.csv`

**Pure pre-COVID stratum** (already merged into G1..G4 above; provided here for traceability):
`C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\data\hospital\PURE DATASET\`
(`pure_daily_arrival.csv`, `pure_hourly_arrival.csv`, `pure_clinical_arrival.csv`, `pure_hourly_clinical_arrival.csv`)

### 4.3 Pre-computed upstream artefacts available to load (DO NOT re-derive)

`outAnalysis/07_chapter3_alignment/` already contains several artefacts that Steps §9, §10, §14, and §12 would otherwise reproduce. **Load these instead of recomputing** unless an audit shows they need refreshing.

| Upstream artefact | Replaces / informs |
|---|---|
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\feature_engineering\G1_fully_engineered.csv` | §9 engineered matrix (cross-check column count and recipe match) |
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\feature_selection\feature_selection_results.csv` | §10 four-method consensus output (Algorithm 1) |
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\temporal_disaggregation\hourly_proportion_curves.csv` | §14 hourly proportion curves and shift shares |
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\stl_components\stl_components.csv` | §12 STL hybrid base decomposition |
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\evaluation_framework\hpo_ranges.csv` | §2.7 HPO ranges (cross-check against Ch3 §3.5.9) |
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\evaluation_framework\evaluation_metrics.csv` | §2.7 metric definitions (cross-check) |
| `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\01_joined_daily_demand\section_5_1_5_features\feature_ranking_triangulated.csv` | §5.2.5 raw 10 inventory (the SARIMAX / NB GLM input block) |

**Rule:** for every Step in §6 onwards, the first action is to attempt to load the upstream artefact. If shape, dates, or column inventory match the methodology spec exactly, use it. If not, document the divergence and regenerate.

### 4.4 Split (§5.5.2)

| Block | Window | Days | Mean | SD | Role |
|---|---|---|---|---|---|
| Pre-COVID | 2019-05-01 to 2020-02-29 | 302 | 57.1 | n/a | Sensitivity, calendar stability, narrative |
| During-COVID | 2020-03-01 to 2022-03-31 | 724 | 48.4 | n/a | Excluded |
| Train | 2022-03-01 to 2024-06-30 | 853 | 58.4 | 13.7 | Model fitting, rolling CV |
| Validation | 2024-07-01 to 2024-12-31 | 184 | 60.6 | 9.9 | HPO and order selection |
| Test | 2025-01-01 to 2026-01-31 | 396 | 69.1 | 12.0 | Final reporting only |

KS distances: train–val D = 0.14 (p = 6.3×10⁻³); val–test D = 0.37; train–test D = 0.44 (p = 7.3×10⁻⁴⁷). Test block is explicitly out-of-distribution with mean +18.3% above train.

### 4.5 Cross-validation (§3.6.1)

Rolling-origin expanding window. Initial training window covers earliest portion of train block. Forecast horizon 7 days. After evaluation, expand training window by 7 days and repeat. Final metrics averaged across windows. No random shuffling.

---

## 5. Design language

Every script obeys the same matplotlib preamble and palette so Ch6 figures sit next to Ch4 and Ch5 figures.

### 5.1 Matplotlib preamble

```python
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})
```

### 5.2 Palette

```python
NAVY      = '#1e6091'   # primary, history line, headline metric
TEAL      = '#0d9488'   # forecast line, secondary highlight
GREEN     = '#16a34a'   # within tolerance
AMBER     = '#d97706'   # warning zone
ROSE      = '#dc2626'   # alert, sign-reversal, breach
NEUTRAL   = '#475569'   # auxiliary lines, axis text
LIGHT     = '#e5e7eb'   # gridlines, fills
DARK_TEXT = '#0f172a'   # body text
```

### 5.3 Inherited conventions

- Symmetric correlation matrices use lower triangle only, diagonal masked, capped colour scale (correlations ±0.30, effect sizes ±50 pp). Cells exceeding cap carry `†`.
- Significance asterisks on coloured cells use white path-effect stroke:
  ```python
  ax.text(x, y, '*', color='black', ha='center', va='center',
          path_effects=[withStroke(linewidth=2, foreground='white')])
  ```
- Forecast charts: history in NEUTRAL, forecast in TEAL, shaded 95% PI. Last historical point repeated as first forecast point.
- Stability or threshold bars use GREEN/AMBER/ROSE banding.
- Labels that overlap data go into white-bordered boxes.
- Day labels `Mon May 4`, not ISO.
- Every figure saves at 300 dpi PNG into `artefacts/figures/` and is synced to `c:\Users\BIBINBUSINESS\OneDrive\Desktop\latex code\figures\ch6\`.

---

## 6. Step 1 — Reference floor

### 6.1 Goal

Absolute floor any later model must beat. Three zero/light-fit rules:

| Name | Definition | Purpose |
|---|---|---|
| `naive_yest` | `ŷ(t) = y(t-1)` | persistence floor |
| `naive_seasonal` | `ŷ(t) = y(t-7)` | weekly seasonality floor |
| `dow_mean` | `ŷ(t) = mean of same weekday on train` | calendar-aware floor |

Grounding: lag-1 ACF 0.538, lag-7 ACF 0.490 (§5.2.2); Kruskal-Wallis on DoW p = 3.3×10⁻³⁰.

### 6.2 Coding prompt

```text
Implement three zero/light-fit baselines for daily ED arrivals.

Inputs: G1, filter is_zero_day == 0, split per §4 of this plan.

  1. predict_naive_yest(series) = series.shift(1)
  2. predict_naive_seasonal(series, lag=7) = series.shift(7)
  3. predict_dow_mean(train_series, full_index) = mean per weekday on train,
     mapped onto every date in full_index.

For each, compute MAE, RMSE, MAPE on val and test. Write
artefacts/metrics/reference_floor.csv with one row per (block, baseline)
combination (six rows): block, baseline, MAE, RMSE, MAPE.

No parameters fit. Do not touch test during model selection.
```

### 6.3 Figure 6.1 — Reference floor

11×4 inches. Plot last 60 days of train + full val block. Three lines: truth NEUTRAL solid, `naive_seasonal` TEAL dashed, `dow_mean` NAVY dashed. MAE per baseline annotated in white-bordered box lower-right. GREEN dashed vertical line at 2024-06-30 labelled `train | val`.

### 6.4 Reporting

- Record all three MAE values on val and test.
- Report whether `naive_seasonal` beats `naive_yest` (expected given lag-7 ACF > lag-1 by amount).
- Report whether `dow_mean` is competitive with `naive_seasonal`.

No pass/fail gate. Numbers are what they are.

---

## 7. Step 2 — ARIMA baseline

### 7.1 Goal

Lower of the two statistical baselines named in §3.5.2. ARIMA fitted **without exogenous regressors** so the SARIMAX + NB GLM contribution in §8 has somewhere to come from.

### 7.2 Grounding

- §5.2.2 ADF rejects unit root (p = 0.013), KPSS rejects stationarity (p < 0.05) → trend-stationary, d = 1.
- PACF dominant lag 1 (0.538), secondary lag 2 (0.242), then flat. ACF tails geometrically. AR(1) or AR(2) candidates.

### 7.3 Order selection (§3.5.2 Algorithm 2)

Box-Jenkins with `pmdarima` stepwise fallback. Grid: p ∈ {0..3}, d = 1 fixed, q ∈ {0..3}, seasonal off. Selection by AIC. Diagnostics: Ljung-Box at lags 7, 14, 21 (α = 0.05) and Jarque-Bera.

### 7.4 Coding prompt

```text
Implement ARIMA baseline per §3.5.2.

Inputs: daily target filtered is_zero_day == 0; splits per §4.

Use pmdarima.auto_arima(seasonal=False, max_p=3, max_q=3, d=1,
trace=True, suppress_warnings=True, stepwise=True, error_action='ignore',
information_criterion='aic').

Diagnostics on residuals:
  - Ljung-Box at lags 7, 14, 21
  - Jarque-Bera

Forecast strategy:
  - Validation: one-step-ahead refit on rolling origin (weekly refit, 7-day forecast,
    advance origin by 7 days).
  - Test: same procedure, single OOD pass at end of project.

Save:
  - artefacts/predictions/arima.parquet (date, block, actual, predicted, lower_95, upper_95)
  - artefacts/metrics/arima_metrics.csv (block, MAE, RMSE, MAPE, R²)
  - artefacts/models/arima_order.txt (p,d,q)
  - artefacts/metrics/arima_diagnostics.csv (Ljung-Box, JB stats)

Seed: np.random.seed(42).
```

### 7.5 Figure 6.2 — ARIMA residual diagnostics

12×4 inches, three panels.

- **Left:** one-step-ahead forecast on val (184 days) with 95% PI band. Truth NEUTRAL, forecast TEAL, band TEAL at 0.2 alpha.
- **Middle:** residual ACF up to lag 21, blue stems, 95% CI band light grey. Significant lags marked with asterisks.
- **Right:** residual QQ plot against standard normal. Reference line NAVY. JB p-value annotated.

### 7.6 Reporting (no enforcement)

- Picked order. Report; do not cap.
- Ljung-Box at lag 14. Report; if it rejects whiteness, that is a finding worth noting in §6.3 of the chapter.
- Val MAPE vs `naive_seasonal`. Report relative position.

---

## 8. Step 3 — SARIMAX + NB GLM (parallel)

### 8.1 Goal

The two parallel parametric baselines mandated by §5.7: SARIMAX with Gaussian likelihood, and Negative Binomial GLM regression on the same exogenous block. Two separate models, both reported.

### 8.2 Grounding

- §5.2.1: NB primary (VMR = 3.49 rejects Poisson; Normal–NB AIC gap 0.4%). No log/Box-Cox (λ ≈ 0.85).
- §5.2.2: SARIMAX template `(p, 1, q)(P, 1, Q)_7`, p,q,P,Q ∈ {0,1,2}, d = D = 1 fixed, AIC-selected.
- §5.2.5: 10-feature exogenous block fixed (DoW + temp + wind + 7 calendar binaries). No engineering, no consensus — these baselines use the raw 10 directly.
- §5.7 (verbatim): *"Chapter 6 therefore fits a Negative Binomial regression alongside a SARIMA(p, 1, q)(P, 1, Q)_7 baseline."*

### 8.3 SARIMAX pipeline

```text
Inputs: same splits; exogenous X = §5.2.5 raw 10 (DoW as 6 dummies; temp_mean_C and
wind_max_kmh standardised on train; 7 calendar binaries).

Use pmdarima.auto_arima(seasonal=True, m=7, max_p=2, max_q=2, max_P=2, max_Q=2,
d=1, D=1, trace=True, suppress_warnings=True, stepwise=True,
error_action='ignore', information_criterion='aic', X=X_train).

Forecast strategy: rolling-origin weekly refit. Carry X forward at each origin.

Save:
  - artefacts/predictions/sarima.parquet (date, block, actual, predicted, lower_95, upper_95)
  - artefacts/metrics/sarima_metrics.csv (block, MAE, RMSE, MAPE, R²)
  - artefacts/models/sarima_order.txt (p,d,q)(P,D,Q,s)
  - artefacts/metrics/sarima_coefficients.csv (feature, coef, std_err, p_value)
```

### 8.4 NB GLM pipeline (headline parametric per §5.7)

```text
Inputs: same splits; same exogenous X as SARIMAX, plus lag y_{t-7} as a single
autoregressive control (motivated by §5.2.2 lag-7 ACF = 0.490).

Use statsmodels.GLM(family=sm.families.NegativeBinomial(alpha=auto)) with log link.
Estimate dispersion alpha via Pearson chi-squared / df-residual after a Poisson pre-fit.

Forecast strategy: weekly refit on rolling origin. 95% prediction intervals via
the NB pmf (overdispersion-aware).

Save:
  - artefacts/predictions/nbglm.parquet (date, block, actual, predicted, lower_95, upper_95)
  - artefacts/metrics/nbglm_metrics.csv (block, MAE, RMSE, MAPE, R²)
  - artefacts/metrics/nbglm_coefficients.csv (feature, coef, std_err, IRR, p_value)
  - artefacts/metrics/nbglm_dispersion.csv (alpha estimate, deviance, df)
```

### 8.5 Normal-likelihood sensitivity

Per §5.2.1, fit one additional Normal-likelihood GLM on the same X to confirm the 0.4% AIC gap. Report AIC delta. Single row appended to `nbglm_metrics.csv` under block = `sensitivity_normal`.

### 8.6 Figure 6.3 — Coefficient panels

Two side-by-side horizontal bar charts.

- **Left:** SARIMAX exogenous coefficients, sorted by |coef|. Positive NAVY, negative ROSE. Error bars from SE. Asterisks on Wald p < 0.05.
- **Right:** NB GLM IRR (incidence rate ratios = exp(coef)), sorted same. Reference line at IRR = 1. Asterisks on p < 0.05.

### 8.7 Figure 6.4 — ARIMA vs SARIMAX vs NB GLM on val

Two rows.

- **Row 1:** forecast plot. Truth NEUTRAL, ARIMA NAVY, SARIMAX TEAL, NB GLM AMBER. 95% PI band from NB GLM only.
- **Row 2:** grouped bar chart MAE / RMSE / MAPE on val and test for the three baselines.

### 8.8 Sign consistency (report; do not enforce)

Per §5.2.3, all seven retained calendar coefficients are expected negative. Mean temperature expected positive (§5.2.4). Max wind expected negative (§5.2.4). Report sign agreement. If a sign disagrees, investigate before treating it as a finding.

### 8.9 Reporting

- Picked SARIMAX order; NB GLM dispersion alpha.
- All three baselines' val and test MAPE/MAE/RMSE/R².
- Sign-consistency table.
- Normal-likelihood AIC delta (expected ≈ 0.4% per §5.2.1).

---

## 9. Step 4 — Feature engineering (ML models only)

### 9.1 Scope

**Applies to XGBoost, ANN, LSTM, and the 6 hybrids only.** SARIMAX / ARIMA / NB GLM do not consume engineered features — they use the §5.2.5 raw 10 directly (§3.4.3 final paragraph distinguishes the two pipelines).

### 9.2 Goal

Expand the §5.2.5 raw 10 into the 50–100 derived features that §3.4.2 specifies.

### 9.3 Six categories

| Category | Specification |
|---|---|
| Calendar binaries | 7 §5.2.5 survivors (already in G1, keep) |
| Cyclical sin/cos | For each periodic t with period s: `sin(2πt/s)`, `cos(2πt/s)`. Apply to hour, DoW, day-of-year, month |
| Fourier harmonics | s ∈ {7, 365}, k = 1, 2, 3 → 12 features total |
| Lag features | `y_{t-1}, y_{t-7}, y_{t-14}, y_{t-30}`; same 4 lags on temp_mean_C and wind_max_kmh |
| Rolling-window stats | Mean, SD, min, max at windows {3, 7, 14, 30} on target + temp + wind |
| Selected interactions | `is_weekend × temp_mean_C`, `is_public_holiday × is_weekend`, `is_year_end × day_of_week` (theory-motivated per §3.4.2) |

### 9.4 Coding prompt

```text
Implement §3.4.2 feature engineering recipe for daily ED arrivals.

FIRST: attempt to load the upstream pre-computed matrix:
  C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\feature_engineering\G1_fully_engineered.csv

Validate that its column inventory matches the §3.4.2 recipe (six categories below)
and that its row count matches expectations after dropping the first 30 days of lag
warm-up. If it does, use it directly and skip the regeneration step.

If the upstream artefact is missing or its inventory does not match, regenerate
from G1:
  - C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\01_joined_daily_demand\G1_pure_daily_demand.csv
  - Filter is_zero_day == 0
  - Target: total_daily_arrivals
  - Raw exogenous: §5.2.5 raw 10

Produce in order:
  1. The 7 calendar binaries (kept from G1).
  2. Cyclical sin/cos on hour, DoW, day-of-year, month.
  3. Fourier harmonics for s ∈ {7, 365}, k ∈ {1,2,3} (12 features).
  4. Lags y_{t-1}, y_{t-7}, y_{t-14}, y_{t-30} on target, temp_mean_C, wind_max_kmh.
  5. Rolling mean/SD/min/max at {3, 7, 14, 30} on the same three series.
  6. Three theory interactions: is_weekend*temp_mean_C, is_public_holiday*is_weekend,
     is_year_end*day_of_week.

Drop rows where any lag or rolling feature is NaN (first 30 days). Save to
data/processed/engineered_features.parquet with date as index. Print final
shape and column count (expected 50–100).

Seed 42. No standardisation at this stage — standardisation happens inside
each ML model's training procedure on the training fold only.
```

### 9.5 Reporting (no enforcement)

- Final column count.
- First valid row date.
- Any NaN report.

---

## 10. Step 5 — Four-method consensus selection

### 10.1 Scope

**Applies to ML models only** (same scope as §9). Compresses the 50–100 engineered features to a stable subset of 15–40.

### 10.2 Grounding

§3.4.3 cites three reasons: SARIMAX coefficient variance inflation from multicollinearity between adjacent lag and rolling features (does not apply here, since SARIMAX uses the raw 10 — but the same multicollinearity hurts ML model importance attribution); feature-to-sample-ratio risk on 853 days; SHAP-importance dilution.

### 10.3 Algorithm 1 (verbatim from §3.4.3)

| Selector | Implementation | Top-half rule |
|---|---|---|
| Dummy | `sklearn.dummy.DummyRegressor`, ranks by feature variance | Above median variance |
| RF permutation | `permutation_importance(RandomForestRegressor(n_estimators=500, random_state=42), n_repeats=10)` | Above median importance |
| Lasso L1 | `LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, random_state=42)`, ranks by `|coef_|` | Non-zero coefficient |
| Gradient boosting | `XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)`, gain | Above median gain |

Retain feature f if `vote_count(f) ≥ 2`.

### 10.4 Coding prompt

```text
Implement §3.4.3 Algorithm 1: four-method consensus feature selection.

FIRST: attempt to load the upstream pre-computed selection:
  C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\feature_selection\feature_selection_results.csv

Validate it contains the four selector columns (dummy_top, rf_perm_top, lasso_top,
xgb_top) plus vote_count and retained. If so, use it directly and skip the
regeneration step. Copy it to data/processed/selected_features.parquet.

If missing or malformed, regenerate from scratch:

Inputs:
  - data/processed/engineered_features.parquet (from §9)
  - Target: total_daily_arrivals
  - Training fold: 2022-03-01 to 2024-06-30 only (consensus uses full train block
    to avoid leakage; rolling CV is applied later when models are fit on the
    retained features).

Procedure:
  1. Run four selectors independently with parameters above.
  2. Per selector, compute top-half subset.
  3. vote_count = sum of indicators across four subsets.
  4. Retain features with vote_count >= 2.

Outputs:
  - artefacts/metrics/consensus_selection.csv: feature, dummy_top, rf_perm_top,
    lasso_top, xgb_top, vote_count, retained.
  - data/processed/selected_features.parquet: engineered matrix restricted to
    retained features, with same date index.

Audit: every one of the §5.2.5 raw 10 features (and their direct lag/rolling
derivatives) should appear in the retained set. If any drops out, log a warning
and review before proceeding.
```

### 10.5 Figure 6.5 — Consensus vote panel

Two-panel.

- **Top:** horizontal bar chart of engineered features sorted by vote count descending. Vote 4 NAVY, vote 3 TEAL, vote 2 AMBER, vote ≤ 1 NEUTRAL. Dashed vertical at vote = 2.
- **Bottom:** four small heatmap rows showing each selector's ranking. White cell if in top half, light grey otherwise. Asterisks on cells where the §5.2.5 raw 10 features sit.

### 10.6 Reporting

- Final retained count.
- Whether §5.2.5 raw 10 are all retained.
- Top-10 by vote count.

---

## 11. Step 6 — XGBoost, ANN, LSTM standalone

### 11.1 Scope

Three standalone ML models, all fit on `data/processed/selected_features.parquet` (the §10 consensus output).

### 11.2 XGBoost

HPO grid per §3.5.9: n_estimators ∈ {100, 200, 300, 500}; max_depth ∈ {3, 5, 6, 8}; learning_rate ∈ {0.01, 0.05, 0.1, 0.3}; subsample ∈ {0.7, 0.85, 1.0}. Selection by val MAPE. Seed 42.

### 11.3 ANN

Keras MLP. Random search 20 iterations: hidden_layers ∈ {1, 2}; units ∈ {64, 128, 192, 256}; dropout ∈ {0.1, 0.2, 0.3, 0.4}; learning_rate ∈ U(0.0005, 0.01); batch_size ∈ {16, 32, 64}. Early stopping patience 10, ReduceLROnPlateau patience 5. Seeds: numpy 42, tf 42, python 42.

### 11.4 LSTM

Keras LSTM, lookback ∈ {14, 21, 28} d (sequence input from target series + selected exogenous). Optuna TPE 30 trials: units ∈ {64, 96, 128, 192, 256}; dropout ∈ {0.1, 0.2, 0.3, 0.4}; learning_rate ∈ LogUniform(0.0005, 0.01); batch ∈ {16, 32, 64}. MedianPruner. Time budget cap 90 min. If budget hit before 30 trials, use best so far and report trial count.

### 11.5 Coding prompt

```text
For each of XGBoost, ANN, LSTM:

Inputs: data/processed/selected_features.parquet (consensus output from §10).

  1. HPO on training fold via §3.5.9 procedure. Best by val MAPE.
  2. Refit on full train block with best params. Forecast on val (rolling
     weekly refit), then on test (single OOD pass at §17).
  3. Save:
     - artefacts/predictions/{model}.parquet
     - artefacts/metrics/{model}_metrics.csv
     - artefacts/models/{model}_best_params.json
     - artefacts/metrics/{model}_hpo_trace.csv

Seeds 42 throughout. TensorFlow on Windows CPU is not fully deterministic
across runs even with seeds — document the non-determinism in the README,
do not pretend otherwise.
```

### 11.6 Figure 6.6 — Standalone ML

- XGBoost SHAP summary on validation predictions.
- ANN and LSTM training curves (loss vs epoch, val MAPE vs epoch).

---

## 12. Step 7 — Hybrids (3 residual + 3 STL)

### 12.1 Recipes (§3.5.4)

**Residual (Zhang Alg 6):**
- LSTM + XGBoost: base LSTM forecast, XGBoost refines residuals.
- SARIMAX + XGBoost: base SARIMAX forecast, XGBoost refines residuals.
- SARIMAX + LSTM: base SARIMAX forecast, LSTM refines residuals.

`ŷ = f_A(x) + f_B(residuals_of_A)`. Refiner trained on **in-sample training residuals only**; never test or val residuals.

**STL (Alg 7):**
- STL + XGBoost
- STL + ANN
- STL + LSTM

STL with period s = 7, robust. Trend extrapolated linearly. Seasonal forecast via seasonal-naive `Ŝ_{T+h} = S_{T+h-s}`. ML model fitted on residual `R_t` from the same training fold.

**Upstream STL components available at:**
`C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\stl_components\stl_components.csv`

Load and validate columns (`trend`, `seasonal`, `residual`); if present and aligned to G1 dates, use directly. Otherwise compute on the train fold via `statsmodels.tsa.seasonal.STL(y, period=7, robust=True)`.

### 12.2 Coding prompt

```text
For each of the six hybrids:

  1. Residual hybrid:
     a. Fit base f_A on training fold with best §11 params (or best §8 params for
        SARIMAX-based).
     b. Compute residuals r_t = y_t - f_A(x_t) on training fold.
     c. Fit refiner f_B on (x_t, r_t) with its own best params.
     d. Forecast: ŷ = f_A(x) + f_B(x) on val and test.

  2. STL hybrid:
     a. STL decompose training target with period 7, robust.
     b. Trend extrapolated linearly; seasonal via seasonal-naive.
     c. Fit ML refiner on R_t with same selected features.
     d. Forecast: ŷ = T̂ + Ŝ + f_B(x).

Save artefacts/predictions/hybrid_{name}.parquet and
artefacts/metrics/hybrid_{name}_metrics.csv per hybrid.

Critical leakage check: refiner training touches only training-fold residuals.
tests/test_hybrids.py asserts this.
```

### 12.3 Figure 6.7 — STL decomposition example

4-panel STL decomposition of training target. Trend NAVY, seasonal TEAL, residual NEUTRAL, original NEUTRAL.

---

## 13. Step 8 — Task 2 per-specialty

### 13.1 Goal

Same recipe per specialty at the resolution §5.3.1 mandates. Per-specialty calendar and weather coefficients (§5.3.3) are mandatory.

### 13.2 Daily specialties

[Medicine, Orthopaedics, Surgery, Paediatrics, Gynaecology]. Per-specialty exogenous block:

| Specialty | Calendar | Weather | Special |
|---|---|---|---|
| Medicine | 7 §5.2.5 binaries | temp + wind | — |
| Orthopaedics | 7 §5.2.5 binaries | temp only | — |
| Surgery | 7 §5.2.5 binaries | neither (flat) | is_weekend, is_long_weekend, is_public_holiday as explicit interaction columns for sign-reversal |
| Paediatrics | 7 §5.2.5 binaries | wind only | — |
| Gynaecology | 7 §5.2.5 binaries | neither (flat) | — |

For each: ARIMA per §7, SARIMAX + NB GLM per §8, plus winning Task 1 ML architecture (whichever from §11 ranks highest on val).

### 13.3 Weekly specialties

[Maternity, Psychiatry]. Resample target to weekly sum (Mon–Sun). Weather resampled as mean; calendar binaries as any-day-active max. Fit SARIMAX(p, 1, q)(P, 1, Q)_52 with `pmdarima` (m = 52, lower max orders). Honesty over precision — 90% zero days means MAPE may be high.

### 13.4 Coding prompt

```text
For each daily specialty in [Medicine, Orthopaedics, Surgery, Paediatrics, Gynaecology]:
  1. Build per-specialty exogenous block per table in §13.2.
  2. Target = specialty_count / total_daily_arrivals on non-zero days
     (filter is_zero_day == 0). Reconstruct absolute count at delivery
     by multiplying predicted share by Task 1 absolute forecast.
  3. Fit ARIMA, SARIMAX, NB GLM per §7 and §8.
  4. Fit the winning Task 1 ML architecture (best of XGBoost/ANN/LSTM/hybrids
     by §11/§12 val MAPE).

For weekly [Maternity, Psychiatry]:
  1. Resample target to weekly sum.
  2. Resample features per rules above.
  3. Fit SARIMAX(p, 1, q)(P, 1, Q)_52 via auto_arima(m=52, max_p=2, max_q=2,
     max_P=1, max_Q=1, d=1, D=1).
  4. Save weekly results separately, clearly labelled.

Save per-specialty parquet + metrics. Aggregate into
artefacts/metrics/task2_leaderboard.csv with columns specialty, resolution,
model, block, MAE, RMSE, MAPE, R².

Sum-consistency (post hoc per §3.5.10): for each block and date, sum the five
daily-specialty absolute forecasts and compare to Task 1 absolute forecast.
Report mean absolute deviation across block — do NOT enforce as a hard
constraint.
```

### 13.5 Figure 6.8 — Per-specialty MAPE heatmap

7×N matrix heatmap: rows are specialties, columns are (model × block) combinations. Capped colour scale at MAPE = 50%. GREEN < 15%, AMBER 15–30%, ROSE > 30%. Maternity and Psychiatry rows shaded light grey for weekly resolution.

### 13.6 Figure 6.9 — Surgery sign-reversal validation

Two-panel focused on Surgery.

- **Top:** actual Surgery shares as three boxplots (weekdays / weekends / public holidays). Median annotated. NEUTRAL / ROSE / ROSE.
- **Bottom:** SARIMAX-predicted shares for same three groups, dashed TEAL. Annotate SARIMAX coefficients (point + 95% CI) for is_weekend, is_long_weekend, is_public_holiday. Expected positive per §5.3.3.

### 13.7 Reporting

- Per-specialty val and test metrics.
- Sum-consistency mean absolute deviation on val and test.
- Surgery sign-reversal coefficient signs (expected positive; report if otherwise).

---

## 14. Step 9 — Layer 2 hourly disaggregation

### 14.1 Goal

Convert daily forecast into hourly counts per §3.5.8 + §5.4. In scope for Ch6 (methodology), not deferred.

### 14.2 Grounding (§5.4)

- Peak-to-trough 4.8×; proportional allocation only reasonable approach.
- Two day-type curves: Weekday vs Weekend-or-Holiday (Saturday, Sunday, holiday pooled).
- Shift shares stable post-COVID: Day 41.0% (07–14), Evening 40.8% (15–22), Night 18.2% (23–06).
- Per-hour refinement only for 09:00–15:00 where year-to-year relative spread ≤ 10%.

### 14.3 Coding prompt

```text
Implement §5.4 two-tier disaggregation rule.

FIRST: attempt to load the upstream pre-computed curves:
  C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\07_chapter3_alignment\temporal_disaggregation\hourly_proportion_curves.csv

Validate it contains both day-type rows (weekday, weekend_or_holiday) and
all 24 hourly columns. If so, use directly.

If missing, derive from G2:
  C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\outAnalysis\pure\02_joined_hourly_demand\G2_pure_hourly_demand.csv

Procedure:
  1. From train block (2022-03-01 to 2024-06-30):
     p_weekday[h] = mean hourly share on weekdays (Mon-Fri, not holiday)
     p_weekend[h] = mean hourly share on Sat, Sun, public holidays pooled
     Each sums to 1 across 24 hours.

  2. From same train block:
     shift_weekday = [Day_share, Evening_share, Night_share]
     shift_weekend = same
     Each sums to 1.

  3. For 09:00-15:00, retain per-hour proportion from step 1.
     For other hours, use uniform within-shift split: residual_shift_share /
     n_hours_in_shift_outside_refinement.

Forecast rule (per Ch3 eq 3.17):
  Given y_hat_d and day type t:
    For each hour h:
      if h in [9,10,11,12,13,14,15]:
        y_hat_h = y_hat_d * p_t[h]
      else:
        shift_h = shift containing h
        n_out = count of hours in shift_h not in 9-15
        residual = shift_t[shift_h] - sum(p_t[k] for k in 9-15 if k in shift_h)
        y_hat_h = y_hat_d * residual / n_out

Apply to the winning Task 1 daily forecast (best of §8/§11/§12 by val MAPE).
Compare hourly predictions to actual hourly counts from G2.

Save:
  - artefacts/metrics/shift_proportions.csv
  - artefacts/metrics/hourly_refinement.csv
  - artefacts/metrics/disaggregation_metrics.csv (hour-level MAE, shift-level MAE,
    daily reconciliation error on val and test)
```

### 14.4 Figure 6.10 — Weekday vs weekend-or-holiday profile

Two stacked panels.

- **Top:** 24-hour profile. NEUTRAL weekday, TEAL weekend-or-holiday. Shaded 95% CI ribbons. Background vertical bands for Day (07–14 light GREEN), Evening (15–22 light AMBER), Night (23–06 light NEUTRAL).
- **Bottom:** typical week (Mon–Sun) as hourly bars. Actuals NEUTRAL, prediction TEAL. Hour-level MAE annotated.

### 14.5 Reporting

- Day/Evening/Night shift shares within train block. Expected close to 41.0 / 40.8 / 18.2.
- Hourly reconciliation to daily forecast within rounding.
- Hour-level MAE on val for peak-band hours.

---

## 15. Step 10 — Consolidated leaderboard (Table 6.1)

### 15.1 Goal

The single artefact every model decision flows from.

### 15.2 Coding prompt

```text
Read every metrics file produced by Steps 1–9 and assemble:

  artefacts/metrics/leaderboard.csv with columns:
  task, target, model, family, block, MAE, RMSE, MAPE, R², training_time_s,
  inference_latency_ms

  artefacts/tables/table_6_1_publication.csv with columns:
  Task, Family, Model, Val MAPE, Val MAE, Test MAPE, Test MAE

Sort within task by val MAPE ascending. Mark headline rows (SARIMAX, NB GLM,
best ML, best hybrid) with a dagger.
```

### 15.3 Figure 6.11 — Consolidated panel

Three subpanels, 14×5 inches.

- **Left:** Task 1 models, horizontal bar chart of val MAPE. Bars TEAL, sorted ascending. Family colour-key: classical NAVY, ML TEAL, hybrids AMBER.
- **Middle:** Task 2 per-specialty val MAPE for winning model. Horizontal bars sorted descending. Surgery row shaded ROSE.
- **Right:** Layer 2 hour-level MAE by hour, GREEN-AMBER-ROSE banded by year-to-year stability class.

### 15.4 Reporting

Every metric in the leaderboard traceable to a per-step metric file. No hand-computed values.

---

## 16. Step 11 — Pre-COVID secondary analyses

### 16.1 Goal

Three pre-COVID uses §5.5.2 mandates.

### 16.2 Coding prompt

```text
Run three short follow-on analyses using pre-COVID block
(2019-05-01 to 2020-02-29, 302 non-zero days).

A. Sensitivity / ablation:
   Refit SARIMAX and NB GLM on train + pre-COVID combined (1,155 days).
   Forecast on same val and test blocks. Save metrics alongside the
   post-COVID-only result. Report cost or benefit of inclusion.

B. Calendar coefficient stability:
   For Year End, Weekend, Long Weekend, Public Holiday, refit small
   ARIMA-X on pre-COVID block alone. Extract calendar coefficient + 95% CI.
   Compare against post-COVID coefficient. Flag any coefficient whose
   pre- and post-COVID CIs do not overlap.

C. Historical context:
   Compute pre-COVID mean (57.1 per §4.4.3) and percent shifts to train (+2.3%)
   and test (+21.2%). One-row table for operations chapter.

Save to artefacts/metrics/sensitivity/ as
precovid_sensitivity.csv, precovid_calendar_stability.csv, precovid_context.csv.
```

### 16.3 Reporting

- SARIMAX + NB GLM sensitivity ΔMAPE when pre-COVID is included.
- Which (if any) calendar coefficients fail the overlap test.
- Historical-context table.

---

## 17. Step 12 — OOD test pass

### 17.1 Rule

Test block touched **exactly once**. No model selection on test.

### 17.2 Coding prompt

```text
After all val-based model selection is complete and Table 6.1 is locked,
run scripts/07_final_test.py:

  1. Load every fitted model from artefacts/models/.
  2. For each, forecast on the test block (2025-01-01 to 2026-01-31) with
     weekly rolling refit.
  3. Compute MAE, RMSE, MAPE, R² on test.
  4. Append to artefacts/metrics/leaderboard.csv with block = 'test'.
  5. Compute val→test MAPE gap per model.

Save artefacts/metrics/ood_test_report.csv with columns:
  model, val_MAPE, test_MAPE, gap_MAPE, gap_relative.

Discussion in §6.8 of chapter: cite §5.5.2 KS D = 0.44 as the reason for
the gap, not as model failure.
```

### 17.3 Figure 6.12 — Val to test gap

Forest-plot style: one row per model, val MAPE in NAVY, test MAPE in TEAL, joined by light grey line. Mean gap annotated.

---

## 18. Step 13 — Verification harness

### 18.1 Coding prompt

```text
Write scripts/09_verify.py that, on a single run:

  - Asserts every CSV/parquet listed in this plan exists.
  - Confirms train/val/test row counts match 853/184/396.
  - Recomputes one row of each metrics file from the predictions and asserts
    agreement within 0.01.
  - Confirms SARIMAX picked order respects p,q,P,Q ∈ {0,1,2}, d=D=1
    (this is a search-space constraint per §5.2.2, not a result constraint).
  - Asserts shift shares within ±0.01 of 41.0 / 40.8 / 18.2.
  - Asserts all §5.2.5 raw 10 features are retained in §10 consensus output.
  - Reports (without enforcement) Surgery weekend/long-weekend/public-holiday
    SARIMAX coefficient signs.
  - Prints a green tick or red cross per check.

Run before writing chapter prose and again after every code change.
```

### 18.2 Distinction between asserts and reports

The harness **asserts** what the methodology fixes (search spaces, split sizes, file existence). It **reports** what the data produces (model order picked, coefficient signs, MAPE values). A "report" outcome that surprises is a finding for the chapter, not a build failure.

---

## 19. 4-day execution schedule

| Day | Build | Evaluate | Priority |
|---|---|---|---|
| **Mon 18 May** | Repo skeleton; `io.py`, `features.py`, `cv.py`, `metrics.py`; ARIMA + SARIMAX + NB GLM trainers; LaTeX §6.1–§6.2 draft | Smoke ARIMA on train; SARIMAX grid (small) in background; NB GLM dispersion estimate | P1 |
| **Tue 19 May** | `engineering.py` + `consensus.py`; XGBoost + grid HPO; ANN + random HPO; LSTM trainer; LaTeX §6.3 (baselines) | Task 1: full rolling-origin for ARIMA, SARIMAX, NB GLM, XGBoost; ANN HPO running | P1 |
| **Wed 20 May** | LSTM Optuna TPE (overnight); residual hybrids; LaTeX §6.4 | ANN + LSTM rolling-origin; hybrid runs starting | P1 + P2 |
| **Thu 21 May** | STL hybrids; Task 2 per-specialty loop (best Task-1 ML architecture); Layer 2 disaggregation; `08_compare.py`; LaTeX §6.5–§6.7 | Hybrids complete; `07_final_test.py` OOD pass | P2 + P3 |
| **Fri 22 May AM** | Normal-likelihood sensitivity; pre-COVID secondary; LaTeX §6.8–§6.10 (comparison, conclusion); populate tables from leaderboard.csv | Final OOD report; specialty leaderboard; calendar-stability check | P3 |
| **Fri 22 May PM** | Proofread, generate PDF, send to supervisor | — | — |

### Priority tiers

- **P1 — must ship Friday:** ARIMA, SARIMAX, NB GLM, XGBoost, LSTM, ANN on Task 1; rolling CV; OOD test report; leaderboard.
- **P2 — target:** 3 LSTM-based hybrids; Task 2 daily Medicine + Surgery + Orthopaedics; Layer 2 disaggregation.
- **P3 — deferrable:** 3 STL hybrids; Task 2 Paediatrics + Gynaecology daily + Maternity + Psychiatry weekly; Normal-likelihood sensitivity; combined-window ablation.

---

## 20. Risk register and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | LSTM Optuna 30-trial run blows budget (>4 h) | Cap at 15 trials, lookback ∈ {14, 21, 28}, MedianPruner |
| 2 | SARIMAX grid too slow with rolling-origin weekly refit | Restrict (p,q,P,Q) ∈ {0,1,2}; AIC on single train fit picks order; then validate that order on rolling folds (don't search inside each fold) |
| 3 | XGBoost MAPE explodes near zero-arrival days | `is_zero_day == 0` filter at evaluation; report MAPE on non-zero subset |
| 4 | LSTM convergence unstable | Fixed seeds; early stopping patience 10; ReduceLROnPlateau patience 5 |
| 5 | Hybrid coupling leaks test residuals | Refiner reads only train-fold residuals; `tests/test_hybrids.py` asserts |
| 6 | Task 2 weekly Maternity / Psychiatry too sparse for SARIMAX | Fall back to seasonal naive + weekly mean as baseline; document explicitly |
| 7 | OOD test makes everything look bad | Report both val and test; cite §5.5.2 KS D = 0.44 |
| 8 | TensorFlow CPU/GPU friction on Windows | Pin `tensorflow==2.15` CPU; document non-determinism in README |
| 9 | LaTeX figure path not auto-synced | `08_compare.py` writes directly to `c:\Users\BIBINBUSINESS\OneDrive\Desktop\latex code\figures\ch6\` |
| 10 | `main.tex` compile breaks on Friday | Build `chap6_only.tex` standalone for iterative compile during week |
| 11 | NB GLM dispersion fails to converge | Estimate alpha via Pearson chi-squared on Poisson pre-fit; if still fails, fall back to quantile-loss XGBoost per §5.7 footnote |
| 12 | Consensus drops a §5.2.5 raw 10 feature | Warning log + manual review; the raw 10 are settled by §5.2.5 triangulation and should not be overridden by Algorithm 1 |
| 13 | SARIMAX + NB GLM disagree sharply on val | Both reported. Disagreement is a finding for §6.3, not a failure |

---

## 21. Defer matrix

| If behind on… | Cut | Replace with |
|---|---|---|
| Thursday AM (no hybrids) | All 3 STL hybrids | Footnote "STL hybrids deferred to revision" |
| Thursday PM (no Task 2 weekly) | Maternity + Psychiatry weekly | Report 5 daily specialties + placeholder for weekly |
| Friday AM (no LSTM) | LSTM standalone + 2 LSTM hybrids | Report ARIMA, SARIMAX, NB GLM, XGBoost, ANN + SARIMAX+XGB only |
| Friday AM (no Optuna) | LSTM HPO | Defaults (units=128, lookback=21, dropout=0.2) with footnote |
| Friday noon (no Layer 2) | §6.7 hourly figures | Footnote "Layer 2 disaggregation reported in revision" |
| Friday noon (no OOD pass) | §6.8 test numbers | Reuse val metrics, explicit "test deferred" box |
| Friday 14:00 (no sensitivity) | §6.9 Normal sensitivity + ablation | One-paragraph footnote citing Ch5 §5.2.1 |

**Do not cut:** all 7 Task 2 specialties' existence in some form (§5.5.1 mandate); Surgery sign-reversal commentary; the SARIMAX + NB GLM pair (§5.7 mandate).

---

## 22. Always-ship floor

Absolute minimum the supervisor receives Friday afternoon:

- 5 standalone Task 1 models (ARIMA, SARIMAX, NB GLM, XGBoost, ANN)
- Task 2 Medicine + Surgery + Orthopaedics daily
- Leaderboard table (Table 6.1)
- Best model per task named in §6.10
- Clean LaTeX compile
- GitHub repo pushed (Prof Bean already added as collaborator)

If even this is at risk by Friday morning, signal in advance and ask for a one-week extension. **Never send a broken PDF.**

---

## 23. LaTeX outline

Target length: 25–30 pages.

| § | Title | Pages | Contents |
|---|---|---|---|
| 6.1 | Introduction and scope | 1.5 | Recap of 11 models, 2 tasks, link to Ch5 decisions |
| 6.2 | Experimental setup | 3 | Split table; CV diagram; metrics; software stack; seeds; HPO ranges (Tab 6.1 mirrors Ch3 §3.5.9) |
| 6.3 | Parametric baselines | 4 | ARIMA order selection; SARIMAX(p,1,q)(P,1,Q)_7 + diagnostics; NB GLM dispersion + IRR; Normal sensitivity. Fig 6.2 residuals, Fig 6.3 coefficients, Fig 6.4 comparison. Tab 6.2 baseline metrics |
| 6.4 | Engineering + consensus | 2 | §3.4.2 expansion summary; §3.4.3 Algorithm 1; consensus vote panel (Fig 6.5) |
| 6.5 | Standalone ML | 3.5 | XGBoost + SHAP (Fig 6.6), ANN, LSTM (training curves). Tab 6.3 standalone-ML metrics |
| 6.6 | Hybrids | 3.5 | Residual + STL groupings; per-hybrid val MAPE; STL decomposition example (Fig 6.7). Tab 6.4 hybrid metrics |
| 6.7 | Task 2 per-specialty + Layer 2 | 4 | Per-specialty leaderboard Tab 6.5; Fig 6.8 MAPE heatmap; Fig 6.9 Surgery sign-reversal; Fig 6.10 hourly profile |
| 6.8 | OOD test report | 2 | Val→test gap; Tab 6.6 test MAPE; commentary on +18% level drift per §5.5.2; Fig 6.12 |
| 6.9 | Sensitivity and limitations | 1.5 | Normal vs NB AIC; combined-window ablation; year-end widened intervals |
| 6.10 | Conclusion: best model per task | 1 | Selected models + rationale; transition to Chapter 7 |

---

## 24. Acceptance checklist

Friday afternoon go / no-go gate. **All 12 items ticked before sending to Prof Bean.**

1. `leaderboard.csv` exists with ≥ 6 Task 1 models (ARIMA, SARIMAX, NB GLM, XGBoost, ANN, +1), all metrics + R², ranked by MAPE.
2. Rolling-origin CV results reported per model — not a single split.
3. SARIMAX order `(p, 1, q)(P, 1, Q)_7` selected by AIC and reported in §6.3.
4. NB GLM dispersion alpha estimated and reported in §6.3 (§5.7 mandate).
5. 10-feature §5.2.5 inventory used verbatim for SARIMAX, ARIMA-X, NB GLM.
6. Engineering + consensus pipeline run on ML models only; §5.2.5 raw 10 all retained.
7. Test block touched exactly once — single OOD pass in §6.8.
8. ≥ 3 Task 2 daily specialties (Medicine, Surgery, Orthopaedics) with daily metrics; Surgery sign-reversal commented in §6.7.
9. Layer 2 disaggregation results reported in §6.7 (§3.5.8 mandate).
10. Tab 6.1 (HPO ranges), Tab 6.3 (standalone), Tab 6.5 (per-specialty) + Figs 6.4, 6.5, 6.8 present in compiled PDF.
11. Best model per task explicitly named in §6.10.
12. Repo pushed to GitHub with README + `requirements.txt` + reproducibility note (seed = 42).

---

## 25. Crosswalk to chapters 3, 4, 5

| Plan section | Methodology source |
|---|---|
| §2 (settled decisions) | All — direct compilation |
| §4 (data + splits) | Ch4 §4.4.4, §4.5; Ch5 §5.5.2 |
| §6 (reference floor) | New — sub-baseline floor |
| §7 (ARIMA) | Ch3 §3.5.2 Algorithm 2 |
| §8 (SARIMAX + NB GLM) | Ch5 §5.7 + Ch3 §3.5.2 + Ch5 §5.2.1 + Ch5 §5.2.2 + Ch5 §5.2.5 |
| §9 (engineering) | Ch3 §3.4.2 |
| §10 (consensus) | Ch3 §3.4.3 Algorithm 1 |
| §11 (standalone ML) | Ch3 §3.5.9 Table 3.1 |
| §12 (hybrids) | Ch3 §3.5.4 Algorithms 6, 7 |
| §13 (Task 2) | Ch3 §3.5.5 + Ch5 §5.3 |
| §14 (Layer 2) | Ch3 §3.5.8 eq 3.17 + Ch5 §5.4 |
| §15 (leaderboard) | New |
| §16 (pre-COVID) | Ch5 §5.5.2 |
| §17 (OOD test) | Ch5 §5.5.2 |
| §18 (verification) | New |

---

## Document control

- **Version:** 2.0 (merged)
- **Created:** 2026-05-18
- **Status:** Single source of truth. Predecessor drafts (`chapter6_baselines_plan.md`, `CHAPTER_6_MODELLING_BUILD_PLAN.md`) deleted after merge.
- **Live todo list:** Claude's TodoWrite state in the conversation.
