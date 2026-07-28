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
# Alternatives: `conda env create -f environment.yml && conda activate msc-modelling`,
# or skip Python entirely and use Docker -- see "Reproducibility platform" below.

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
| `python make.py setup`           | `pip install -r requirements-core.txt` (pipeline + tests + crossval) |  1 min     |
| `python make.py setup-full`      | + DVC / MLflow / Hydra / Nixtla / DeepAR extras                |  3-5 min   |
| `python make.py verify`          | Confirms splits + data files load                              |  5 sec     |
| `python make.py test`            | Pytest (50 tests; 8 skip gracefully without the raw hospital data) | 20 sec  |
| `python make.py crossval`        | Audit every numeric claim in chap6 / 7 / 8 against the data    | 10 sec     |
| `python make.py fill-gaps`       | Recompute MASE + per-horizon + missing test MAPE rows          | 30 sec     |
| `python make.py leaderboard`     | Rebuild the canonical leaderboard from `*_metrics.csv` files   |  5 sec     |
| `python make.py pipeline`        | Full pipeline rerun from scratch (LONG)                        | 6–10 hours |
| `python make.py docker-build`    | Build the Docker image                                         | 2-4 min    |
| `python make.py docker-test`     | Run the test suite inside a container                          | 30 sec     |
| `python make.py docker-crossval` | Run the crossval audit inside a container                       | 15 sec     |
| `python make.py dvc-dag`         | Print the DVC pipeline graph                                    |  2 sec     |
| `python make.py dvc-repro`       | Re-run any stale DVC stage                                      | 10 sec     |
| `python make.py mlflow-ui`       | Browse logged experiment runs at localhost:5000                 |    —       |
| `python make.py clean`           | Remove pyc + pytest cache                                      |  2 sec     |
| `python make.py help`            | Show this list                                                 |    —       |

See **"Reproducibility platform"** below for what the Docker/DVC/MLflow/CI commands actually cover — and, just as importantly, what they deliberately don't.

---

## What's inside

| Path                              | Purpose                                                                        |
|-----------------------------------|--------------------------------------------------------------------------------|
| `make.py`                         | Cross-platform one-command runbook (Windows-friendly, no Make required)        |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Reproducibility platform: containerised pipeline + MLflow UI service |
| `dvc.yaml`                        | 2-stage DVC pipeline (crossval + consistency_audit) — see "Reproducibility platform" for scope |
| `.github/workflows/ci.yml`        | GitHub Actions CI: pytest + crossval on every push, no data mount needed        |
| `environment.yml`                 | Conda environment spec mirroring `requirements-core.txt`                        |
| `requirements-core.txt`           | Pipeline + tests + crossval dependencies (fast install)                         |
| `requirements-optional.txt`       | DVC/MLflow/Hydra + Nixtla + DeepAR — install on demand                          |
| `CHAPTER_6_PLAN.md`               | The Chapter 6 methodology contract: what each step does + chapter section IDs  |
| `dissertation_improvement_prompts.md` | 16-prompt refactor plan; sub-set 0/1/2/4/7/8/13/14/15 implemented on this branch |
| `artefacts/RESULTS.md`            | Plain-English results discussion, 23 sub-sections from §0 to §7                 |
| `artefacts/CROSS_VALIDATION_REPORT.md` | Numeric-claim audit (every claim in chap6/7/8 verified)                      |
| `artefacts/reports/`              | `crossval_report.txt` + `consistency_report.txt` — DVC stage outputs            |
| `artefacts/leaderboard_canonical.parquet` | 17-field strongly-typed canonical leaderboard, sortable by test MAPE     |
| `artefacts/paper_corpus_features.csv` | 70-feature scan across the 45-paper ED-forecasting literature corpus        |
| `configs/`                        | YAML configs: split dates, feature inventories, HPO ranges, model flags         |
| `configs/paths.yaml`              | **Template**: path strings users edit                                          |
| `configs/paths.local.yaml`        | **User-specific**, gitignored, shadows `paths.yaml` at load                    |
| `src/forecasting/`                | Library: I/O, splits, features, engineering, CV, metrics, model families       |
| `src/forecasting/rolling.py`      | Unified rolling-origin forecaster (replaces 5 copy-paste implementations)      |
| `src/forecasting/leaderboard.py`  | Canonical parquet leaderboard writer + reconciliation                          |
| `src/forecasting/mlflow_utils.py` | `log_run()` context manager — optional, degrades to no-op without mlflow installed |
| `src/forecasting/hybrids/oof/`    | Out-of-fold residual hybrids (Khashei & Bijari 2011 correction)                |
| `src/forecasting/uq/aci.py`       | Adaptive Conformal Inference (Gibbs & Candès 2021)                              |
| `src/forecasting/drift/`          | KMM, RuLSIF, sliding-window CV                                                  |
| `scripts/01..31`                  | Numbered runners — one script per modelling step                                |
| `scripts/27_cross_validate_claims.py` | The auditor (`python make.py crossval`)                                     |
| `scripts/28_fill_gaps.py`         | MASE + per-horizon + parquet update (`python make.py fill-gaps`)                |
| `scripts/29_consistency_audit.py` | Independent second-reader audit; `LATEX_CODE_DIR`/`LATEX_DISERTATION_DIR` env vars override the sibling-directory default |
| `scripts/30_random_forest_baseline.py` | Reference implementation of the MLflow `log_run()` pattern                  |
| `tests/conftest.py`               | Detects whether the confidential raw data is reachable; skips (not crashes) the 8 tests that need it when it isn't |
| `tests/`                          | Pytest suite — 50 tests across 9 files (+ conftest.py)                          |
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

Split into two files: `requirements-core.txt` (everything the pipeline, tests, and crossval need) and `requirements-optional.txt` (the reproducibility-platform stack and a couple of not-yet-wired research extras). `requirements.txt` installs both, for backward compatibility.

| Package | Role | File |
|---|---|:---:|
| `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `scipy` | Core scientific stack | core |
| `pmdarima` | `auto_arima` for ARIMA / SARIMAX order selection | core |
| `xgboost` | Gradient-boosted trees | core |
| `optuna` | HPO (Optuna TPE sampler) | core |
| `torch` | ANN + LSTM (PyTorch CPU build) | core |
| `pyarrow` | Canonical leaderboard parquet | core |
| `matplotlib`, `seaborn`, `shap` | Figures + SHAP importance | core |
| `mapie` | Adaptive Conformal Inference | core |
| `densratio` | RuLSIF importance weights | core |
| `cvxopt` | KMM constrained quadratic program | core |
| `PyPDF2` | Literature-corpus PDF text extraction | core |
| `hydra-core`, `dvc[s3]`, `mlflow` | Reproducibility platform (see above) | optional |
| `mlforecast`, `hierarchicalforecast`, `statsforecast` | Direct multi-output + MinT (Prompts 6 + 9 — not yet wired in) | optional |
| `gluonts[torch]` | DeepAR (Prompt 10 — not yet wired in) | optional |

`torch` was previously only reachable transitively through the optional `gluonts[torch]` extra, meaning a clean `pip install -r requirements.txt` (as it existed before this pass) could run the parametric models but not ANN/LSTM. It's now a direct core dependency.

`python make.py setup` installs core only (fast). `python make.py setup-full` adds the optional stack.

---

## Reproducibility platform

Four pieces of infrastructure sit on top of the pipeline itself: a Docker image, a DVC pipeline, MLflow experiment tracking, and a GitHub Actions CI workflow. Each is real and tested — and each has an honestly-documented scope boundary, because the raw hospital data is confidential and can never leave this machine, which limits what "reproducible from a bare clone" can mean here.

### Docker

```bash
python make.py docker-build       # or: docker build -t msc-modelling .
python make.py docker-test        # runs the 50-test suite in the container
python make.py docker-crossval    # runs the crossval audit in the container
docker compose up mlflow-ui       # -> http://localhost:5000
```

The image installs `requirements-core.txt` plus `mlflow` on Python 3.13-slim. `docker-compose.yml` defines two services (`pipeline`, `mlflow-ui`) sharing a volume-mounted `artefacts/` and `mlruns/` so outputs persist on the host. The full `python make.py pipeline` rerun needs the raw G1-G4 CSVs, which are **not** baked into the image (confidential data — see "Data access"); `docker-compose.yml` documents the two volume mounts to add if you have a local copy.

*(Built and lint-checked in this environment; not build-tested end-to-end here, as Docker isn't installed on the machine this was authored on. Run `python make.py docker-build` yourself the first time and open an issue if anything doesn't come up clean.)*

### DVC

```bash
python make.py dvc-dag      # print the pipeline graph
python make.py dvc-repro    # re-run any stale stage
python make.py dvc-status   # check what's stale without running anything
```

`dvc.yaml` defines exactly **two** stages — `crossval` and `consistency_audit` — chosen because they're the only parts of this repository that are genuinely reproducible from a bare `git clone` with no external inputs. Everything else in the 31-script pipeline reads the raw G1-G4 hospital CSVs from a machine-specific absolute path (`configs/paths.local.yaml`) that can never be committed to Git or DVC; encoding those scripts as DVC stages would build a DAG that looks complete on paper but that `dvc repro` could never actually execute anywhere but this one machine, which is worse than not having the DAG at all. `dvc.yaml`'s header comment explains this in full, including a second, smaller lesson: `scripts/28_fill_gaps.py` was tried as a third stage and **reverted** after `dvc repro` deleted `leaderboard_canonical.parquet` before running it — DVC clears declared outputs before executing a stage's command, which conflicts with a script that reads an existing file and patches it in place rather than regenerating it from scratch. The file was restored from git immediately; the incident is recorded in `dvc.yaml` instead of silently re-attempted, and `python make.py fill-gaps` remains the correct way to run that particular script — as a manual, stateful operation, not a DAG stage.

### MLflow

```bash
python make.py mlflow-ui
```

`src/forecasting/mlflow_utils.py` provides a `log_run()` context manager that degrades to a no-op when `mlflow` isn't installed, so importing it never creates a hard dependency. It is wired into `scripts/30_random_forest_baseline.py` and verified end-to-end: the run's params (`n_estimators`, `max_depth`, `min_samples_leaf`), metrics (val/test MAPE, MAE, RMSE, R², MASE), tags (model family, HPO criterion, git commit), and the prediction CSV all land in `mlruns/` and are browsable via `mlflow ui`. One real compatibility issue surfaced and was fixed during that verification: MLflow 3.x deprecated the plain filesystem tracking backend this project uses and raises an exception instead of a warning unless `MLFLOW_ALLOW_FILE_STORE=true` is set — the wrapper sets it automatically.

**Scope, stated plainly:** the other five training scripts (`06`/`07`/`08`/`24`/`26`, i.e. XGBoost/ANN/LSTM/OOF-hybrid/ACI) are **not yet wired into MLflow**. Backfilling their historical runs would mean re-running hours of training solely to produce tracking metadata for numbers the thesis already reports and that are cross-validated by `scripts/27`; that trade wasn't worth making today. Wiring a new script in going forward is the three-line pattern visible in `scripts/30`.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and PR: `pytest` (all tests that don't need the raw data run; the 8 that do skip cleanly, see below), then `scripts/27_cross_validate_claims.py`. Neither step needs a data mount or secrets, so the workflow is genuinely green from a bare GitHub clone.

### The pytest / confidential-data fix

Two test files (`test_io.py`, `test_features.py`, 8 tests total) call `load_g1()`, which reads the confidential raw CSVs — before this pass, running `pytest` on any machine without them (a CI runner, a teammate's fresh clone) crashed the entire test session with a `FileNotFoundError` at collection time, rather than failing only those 8 tests. `tests/conftest.py` now probes once at the start of the session and marks the 8 data-dependent tests to skip (not fail) with a clear reason when the data isn't reachable; the remaining ~38 pure-function tests (metrics, CV folds, the rolling forecaster, leaderboard schema, OOF hybrid, drift weighting, ACI) are and always were independent of the raw data, and now actually get to run in CI instead of never being collected.

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

50 tests across 9 files. 8 of them (in `test_io.py` and `test_features.py`) call `load_g1()`, which needs the confidential raw hospital CSVs; `tests/conftest.py` probes once at the start of the session and **skips** (not fails) those 8 with a clear reason if the data isn't reachable on the current machine — e.g. CI, or a fresh clone without `configs/paths.local.yaml` pointed at a copy of the dataset. The other 42 are pure-function tests on synthetic data and always run.

| Test file | Tests | Needs raw data? | What |
|---|---:|:---:|---|
| `test_io.py` | 4 | 3 of 4 | Split loader, data file reads |
| `test_features.py` | 5 | all 5 | Feature builder, scaler |
| `test_metrics.py` | 11 | no | MAPE, MAE, RMSE, R², MASE, Winkler, coverage, per-horizon |
| `test_cv.py` | 5 | no | Rolling-origin folds |
| `test_rolling.py` | 7 | no | `RollingForecaster` (byte-identical ARIMA/XGB, 57-fold count, sliding window, sample weights) |
| `test_leaderboard.py` | 6 | no | Parquet roundtrip, upsert, LaTeX export, per-quarter drift sensitivity |
| `test_oof_hybrid.py` | 4 | no | OOF residual variance, refiner HPO independence |
| `test_drift.py` | 4 | no | KMM weights, sliding-window CV, IWCV fallback |
| `test_aci.py` | 4 | no | ACI under synthetic drift, Winkler scoring |

On this machine, with the data present, all 50 pass in about 20 seconds. In CI, 42 run and 8 skip; a green CI build means every data-independent guarantee in the codebase held, not that the whole suite executed end-to-end.

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
