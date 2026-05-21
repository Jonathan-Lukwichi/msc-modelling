# MSc Thesis — Hospital ED Demand Forecasting

**Author:** Jonathan Lukwichi
**Programme:** MEng Industrial Engineering, University of Pretoria
**Thesis title:** *Optimising Hospital Supply Chain Demand Forecasting Using Machine Learning*
**Case study:** Steve Biko Academic Hospital — Emergency Department, daily arrivals

This repository contains the **modelling and evaluation code** for Chapters 6, 7, and 8 of the thesis. It implements the eight standalone forecasters and six hybrid architectures specified in Chapter 3, runs them on the train / val / test splits fixed in Chapter 5 §5.5.2, and produces the artefacts (parquet leaderboard, prediction CSVs, figures) that the LaTeX chapters consume.

The companion LaTeX repository [`Jonathan-Lukwichi/latex-disertation`](https://github.com/Jonathan-Lukwichi/latex-disertation) reads from this repository's `artefacts/figures/` directory at compile time; the two repos are designed to be cloned side by side.

---

## Quick reproduction (non-developer path)

```bash
# 1. Clone both repos as siblings.
git clone https://github.com/Jonathan-Lukwichi/msc-modelling.git
git clone https://github.com/Jonathan-Lukwichi/latex-disertation.git

# 2. Get on the integrated branch.
cd msc-modelling
git checkout claude/review-dissertation-repos-UQtqT

# 3. Install dependencies. Anaconda 3.13 is recommended.
python make.py setup

# 4. Point at your local copy of the dataset (see "Data access" below).
cp configs/paths.yaml configs/paths.local.yaml
# Edit configs/paths.local.yaml to point at G1-G4 CSVs.

# 5. Verify the plumbing.
python make.py verify

# 6. Run the test suite (50 tests, ~30 seconds).
python make.py test

# 7. Cross-validate every numeric claim in the LaTeX chapters
#    against the artefacts on disk. Target: 57 PASS / 0 FAIL / 0 WARN.
python make.py crossval
```

If step 7 prints `All numeric claims within tolerance.` the chapter numbers are reproduced exactly from the committed artefacts. The full pipeline (re-run every model from scratch) takes 6–10 hours and is available via `python make.py pipeline`.

---

## Commands at a glance

| Command                          | Purpose                                                        | Wall-clock |
|----------------------------------|----------------------------------------------------------------|-----------:|
| `python make.py setup`           | `pip install -r requirements.txt`                              |  1 min     |
| `python make.py verify`          | Confirms splits + data files load                              |  5 sec     |
| `python make.py test`            | Pytest (50 tests)                                              | 30 sec     |
| `python make.py crossval`        | Audit every numeric claim in chap6 / 7 / 8 against the data    | 10 sec     |
| `python make.py fill-gaps`       | Recompute MASE + per-horizon + missing test MAPE rows          | 30 sec     |
| `python make.py leaderboard`     | Rebuild the canonical leaderboard from `*_metrics.csv` files   |  5 sec     |
| `python make.py pipeline`        | Full pipeline rerun from scratch (LONG)                        | 6–10 hours |
| `python make.py clean`           | Remove pyc + pytest cache                                      |  2 sec     |
| `python make.py help`            | Show this list                                                 |    —       |

---

## What's inside

| Path                              | Purpose                                                                        |
|-----------------------------------|--------------------------------------------------------------------------------|
| `make.py`                         | Cross-platform one-command runbook (Windows-friendly, no Make required)        |
| `CHAPTER_6_PLAN.md`               | The Chapter 6 methodology contract: what each step does + chapter section IDs  |
| `dissertation_improvement_prompts.md` | 16-prompt refactor plan; sub-set 0/1/2/4/7/8/13/14/15 implemented on this branch |
| `artefacts/RESULTS.md`            | Plain-English results discussion, 23 sub-sections from §0 to §7                 |
| `artefacts/CROSS_VALIDATION_REPORT.md` | Numeric-claim audit (every claim in chap6/7/8 verified)                      |
| `artefacts/leaderboard_canonical.parquet` | 17-field strongly-typed canonical leaderboard, sortable by test MAPE     |
| `configs/`                        | YAML configs: split dates, feature inventories, HPO ranges, model flags         |
| `configs/paths.yaml`              | **Template**: path strings users edit                                          |
| `configs/paths.local.yaml`        | **User-specific**, gitignored, shadows `paths.yaml` at load                    |
| `src/forecasting/`                | Library: I/O, splits, features, engineering, CV, metrics, model families       |
| `src/forecasting/rolling.py`      | Unified rolling-origin forecaster (replaces 5 copy-paste implementations)      |
| `src/forecasting/leaderboard.py`  | Canonical parquet leaderboard writer + reconciliation                          |
| `src/forecasting/hybrids/oof/`    | Out-of-fold residual hybrids (Khashei & Bijari 2011 correction)                |
| `src/forecasting/uq/aci.py`       | Adaptive Conformal Inference (Gibbs & Candès 2021)                              |
| `src/forecasting/drift/`          | KMM, RuLSIF, sliding-window CV                                                  |
| `scripts/01..28`                  | Numbered runners — one script per modelling step                                |
| `scripts/27_cross_validate_claims.py` | The auditor (`python make.py crossval`)                                     |
| `scripts/28_fill_gaps.py`         | MASE + per-horizon + parquet update (`python make.py fill-gaps`)                |
| `tests/`                          | Pytest suite — 50 tests across 9 files                                          |
| `artefacts/figures/`              | Chapter-ready PNG figures (300 dpi)                                            |
| `artefacts/predictions/`          | Per-model val + test prediction CSVs                                           |
| `artefacts/predictions/test/`     | Test-block predictions                                                          |
| `artefacts/metrics/`              | Per-model metric CSVs that feed the leaderboard                                 |

---

## Headline deployment recommendation

For new deployments at South African public hospitals:

| Hospital tier | Recommended forecaster | Test MAPE | Test MASE | UQ method |
|---|---|---:|---:|---|
| **Tertiary** (Steve Biko-class) | XGBoost RMSE-tuned, 23 consensus features | **12.63 %** | **0.724** | ACI (γ = 0.005) — 94.2 % coverage at 95 % nominal |
| **Secondary** (regional) | SARIMAX(1,1,1)(0,1,1)₇ on 10 raw §5.2.5 exog | 13.11 % | 0.732 | NB-pmf prediction interval |
| **Primary** (clinic) | DoW-mean with sliding-450 window | ~14.69 % | — | NB-pmf at the daily floor |

See `artefacts/RESULTS.md` §0–§6sexies for the full rationale and `chap6.tex` / `chap7.tex` / `chap8.tex` in the companion repo for the publication treatment.

---

## Data access

The raw register data is **not** committed:

1. **Confidentiality**: the Casualty register at Steve Biko contains date-linked patient counts that pre-date the public NHI rollout.
2. **File size**: the joined G1–G4 CSVs alone exceed 50 MB.
3. **Regenerability**: the data preparation pipeline is in the sibling `dataAnalysis/` repository.

To run this repository against your own copy:

1. Obtain the four joined CSVs from the upstream pipeline (or your hospital's equivalent data export):
   - `G1_pure_daily_demand.csv` (daily total arrivals + calendar + weather)
   - `G2_pure_hourly_demand.csv` (hourly arrivals; for Layer 2 hourly disaggregation)
   - `G3_pure_clinical_daily.csv` (per-specialty daily arrivals)
   - `G4_pure_clinical_hourly.csv` (per-specialty hourly arrivals)
   - `G1_fully_engineered.csv` (the §3.4.2 ML feature space)

2. Copy `configs/paths.yaml` to `configs/paths.local.yaml` and replace the `<PATH_TO_DATAANALYSIS>` placeholders with the absolute paths to your G1–G4 CSVs.

3. Run `python make.py verify`. If you see `OK: G1 loaded N rows; splits train/val/test = 848/184/396`, the plumbing is correct.

Without the raw data the cross-validation audit still works on the committed prediction CSVs, so non-developers can still verify the chapter numbers without obtaining the raw register.

---

## Dependencies

| Package | Role | Required? |
|---|---|:---:|
| `numpy`, `pandas`, `scikit-learn`, `statsmodels` | Core scientific stack | ✓ |
| `pmdarima` | `auto_arima` for ARIMA / SARIMAX order selection | ✓ |
| `xgboost` | Gradient-boosted trees | ✓ |
| `optuna` | HPO (Optuna TPE sampler) | ✓ |
| `torch` | ANN + LSTM (PyTorch CPU build) | ✓ |
| `pyarrow` | Canonical leaderboard parquet | ✓ |
| `matplotlib`, `seaborn`, `shap` | Figures | ✓ |
| `mapie` | Adaptive Conformal Inference | recommended |
| `densratio` | RuLSIF importance weights | recommended |
| `cvxopt` | KMM constrained quadratic program | recommended |
| `hydra-core`, `dvc[s3]`, `mlflow` | Config + experiment tracking (Prompt 3 — not yet implemented) | optional |
| `mlforecast`, `hierarchicalforecast`, `statsforecast` | Direct multi-output + MinT (Prompts 6 + 9 — not yet implemented) | optional |
| `gluonts[torch]` | DeepAR (Prompt 10 — not yet implemented) | optional |

`python make.py setup` installs the required + recommended set. The optional packages are pinned in `requirements.txt` but install on demand.

---

## Reproducibility

- **Seeds**: every stochastic procedure uses seed `42` (numpy, random, sklearn, xgboost, torch, Optuna).
- **CV folds**: weekly rolling-origin expanding-window, k=10 sub-folds inside the training block.
- **PyTorch on Windows CPU is not bit-exact** across runs even with seeds; the docs are explicit, the chapter discusses this in §3.6.1.
- **`pmdarima.auto_arima`** is deterministic given fixed input and seed.
- **Parquet schema** for the canonical leaderboard is fixed in `src/forecasting/leaderboard.py:CANONICAL_SCHEMA`. Any future schema change should be versioned.

The cross-validation script (`python make.py crossval`) verifies that all 57 numeric claims in the three LaTeX chapters match the on-disk artefacts to ±0.005 tolerance. As of `claude/review-dissertation-repos-UQtqT` HEAD, the result is **57 PASS / 0 FAIL / 0 WARN**.

---

## Testing

```bash
python make.py test   # or directly: python -m pytest tests/ -q
```

50 tests across 9 files:

| Test file | Tests | What |
|---|---:|---|
| `test_io.py` | 4 | Split loader, data file reads |
| `test_features.py` | 5 | Feature builder, scaler |
| `test_metrics.py` | 11 | MAPE, MAE, RMSE, R², MASE, Winkler, coverage, per-horizon |
| `test_cv.py` | 5 | Rolling-origin folds |
| `test_rolling.py` | 7 | `RollingForecaster` (byte-identical ARIMA/XGB, 57-fold count, sliding window, sample weights) |
| `test_leaderboard.py` | 6 | Parquet roundtrip, upsert, LaTeX export, per-quarter drift sensitivity |
| `test_oof_hybrid.py` | 4 | OOF residual variance, refiner HPO independence |
| `test_drift.py` | 4 | KMM weights, sliding-window CV, IWCV fallback |
| `test_aci.py` | 4 | ACI under synthetic drift, Winkler scoring |

---

## Branches

| Branch | Status |
|---|---|
| `main` | The thesis baseline as of 2026-05-19 |
| `claude/review-dissertation-repos-UQtqT` | **Current development branch** — Prompts 0/1/2/4/7/8/13/14/15 of `dissertation_improvement_prompts.md` |

The companion LaTeX repo uses the same branch name for the chap6/7/8 rewrites. Both branches are intended to be merged together once the open PRs are reviewed.

---

## Citing this work

If you reference this codebase in academic work, cite the MSc thesis it accompanies:

> Lukwichi, J. (2026). *Optimising Hospital Supply Chain Demand Forecasting Using Machine Learning: A Case Study of Steve Biko Academic Hospital Emergency Department.* MSc Thesis, Department of Industrial and Systems Engineering, University of Pretoria.

---

## Acknowledgements

- **Prof. W.L. Bean** (University of Pretoria) — thesis supervision.
- **Steve Biko Academic Hospital** — Emergency Department register data.
- **NDoH / NICD** — public health data context.

---

## Licence

Academic use; see `LICENSE` for the full text.
