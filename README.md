# MSc Thesis — Chapter 6: Hospital ED Demand Forecasting

**Author:** Jonathan Lukwichi
**Programme:** MSc Industrial Engineering, University of Pretoria
**Thesis title:** Optimising Hospital Supply Chain Demand Forecasting Using Machine Learning
**Case study:** Steve Biko Academic Hospital — Emergency Department, daily arrivals

This repository contains the **modelling and evaluation code for Chapter 6** of the thesis. It implements every model specified in Chapter 3 (eleven candidates spanning naïve, classical statistical, parametric GLM, machine-learning, deep-learning, and hybrid families), runs them on the train / val / test splits fixed in Chapter 5 §5.5.2, and produces the figures, tables, and discussion the chapter needs.

The repository is the **single source of truth for the modelling work**. The accompanying chapter narrative (`CHAPTER_6_PLAN.md`) and the results / interpretation document (`artefacts/RESULTS.md`) are written from the artefacts this codebase produces.

---

## What's inside

| Path | Purpose |
|---|---|
| `CHAPTER_6_PLAN.md` | The methodology contract — what each step does and the chapter section it satisfies |
| `artefacts/RESULTS.md` | Plain-English results discussion, literature-grounded interpretation, why our best MAPE sits above 10 %, and five paths to break that barrier |
| `configs/` | YAML configs: split dates, feature inventories, HPO ranges, model flags |
| `src/forecasting/` | Library code: data I/O, splits, features, engineering, CV, metrics, model families, hybrids, deployment |
| `scripts/` | Numbered runners — one script per modelling step |
| `tests/` | Smoke tests for the foundation modules |
| `artefacts/figures/` | Chapter-ready PNG figures (300 dpi) |
| `artefacts/tables/` | Wide-form publication tables (Susnjak 2023 column layout) |
| `artefacts/metrics/` | Per-model metric CSVs that feed the leaderboard |

---

## Quick start

```bash
# 1. Install dependencies (Anaconda 3.13 recommended)
pip install -r requirements.txt

# 2. Configure paths to your local data
cp configs/paths.yaml configs/paths.local.yaml
# Open paths.local.yaml in an editor and replace <PATH_TO_DATAANALYSIS>
# with the absolute path to your dataAnalysis/outAnalysis tree.
# paths.local.yaml is gitignored and shadows paths.yaml at load time.

# 3. Sanity-check the data plumbing
python -m src.forecasting.io
python -m src.forecasting.features
python -m src.forecasting.consensus

# 4. Audit the splits and CV procedure honour Ch5 §5.5.2 + Ch3 §3.6.1
python scripts/99_audit_splits.py

# 5. Run the modelling pipeline (in numerical order)
python scripts/01_reference_floor.py           # naïve baselines
python scripts/02_arima.py                     # plain ARIMA
python scripts/03_sarimax.py                   # SARIMAX with §5.2.5 exogenous block
python scripts/04_nbglm.py                     # NB GLM (§5.7 headline parametric)
python scripts/06_xgboost.py                   # XGBoost (k=10 inner CV)
python scripts/07_ann.py                       # ANN (k=10 inner CV)
python scripts/08_lstm.py                      # LSTM (k=10 inner CV)
python scripts/09_hybrids.py                   # 5 of 6 hybrids
python scripts/11_lstm_xgb_hybrid.py           # final LSTM+XGB hybrid
python scripts/14_task2_specialties.py         # Task 2 per-specialty (ARIMA/SARIMAX/XGB)
python scripts/15_task2_ml_hybrids.py          # Task 2 per-specialty (ANN/LSTM + 6 hybrids)

# 6. Build leaderboard and figures
python scripts/10_master_leaderboard.py
python scripts/12_ablation.py                  # design-choice ablation study
python scripts/13_save_for_cloud.py            # save every model as a .pkl for cloud use
```

All scripts are deterministic given seed = 42 (`numpy`, `random`, `sklearn`, `xgboost`, Optuna). PyTorch on Windows CPU is not bit-exact reproducible across runs even with seeds; we document this in the chapter rather than pretend otherwise.

---

## Methodology highlights

- **Splits fixed by Ch5 §5.5.2** — train 2022-03-01 → 2024-06-30 (848 modelling days post zero-day filter), val 2024-07-01 → 2024-12-31 (184 days), test 2025-01-01 → 2026-01-31 (396 days). The test block is out-of-distribution (KS D = 0.44 vs train).
- **Post-COVID only for training** — pre-COVID (2019-05 → 2020-02) and during-COVID (2020-03 → 2022-03) are reserved for sensitivity / calendar-stability analyses (see [scripts/99_audit_splits.py](scripts/99_audit_splits.py) for the verification check).
- **Cross-validation per Ch3 §3.6.1** — rolling-origin expanding-window CV inside the training block, **k = 10** for every ML / DL model so the comparison is fair. The val block is held out for a single fairness check after architecture selection; the test block is touched exactly once.
- **Two parallel parametric baselines per Ch5 §5.7** — Gaussian SARIMAX as the time-series baseline, NB GLM as the headline parametric likelihood. Not nested as an "NB-SARIMAX" because §5.7 prescribes them side-by-side.
- **Two-pipeline feature architecture** — parametric models (SARIMAX, NB GLM, ARIMA-X) use the §5.2.5 raw 10-feature inventory directly; ML / DL models use the §3.4.2 engineered space (50–100 features) reduced by §3.4.3 four-method consensus to ~23 features.

---

## Reproducibility note

- **Seed = 42** for all stochastic procedures (numpy, random, sklearn, xgboost, torch, Optuna sampler).
- **Pinned dependencies** in `requirements.txt` (see `pip install -r requirements.txt`).
- **pmdarima auto_arima** is deterministic given fixed input and seed.
- **TensorFlow on Windows CPU** is not fully deterministic across runs even with seeds; we use PyTorch instead.
- **Anaconda 3.13** is the development environment of record. Python 3.14 was tested but its PyPI wheel availability for PyTorch is intermittent at the time of writing.

---

## Data not in this repository

The raw data files (Casualty Daily Register Excel sheets, weather feeds, calendar) are **not** committed:

- Confidentiality (the hospital register contains date-linked patient counts).
- File size — the joined G1–G4 CSVs alone exceed 50 MB.
- Regenerability — see the upstream ETL repository (`healthforecast_pipeline/`) and the EDA repository (`dataAnalysis/`) for the chain that produces G1–G4 and the §3.4.2 engineered matrix.

To run this repository against your own copy of the data, edit `configs/paths.local.yaml` (template at `configs/paths.yaml`) to point at where your G1–G4 CSVs live. Every script reads through `src.forecasting.io.load_paths()` so a single edit propagates everywhere.

---

## Citing this work

If you reference this codebase in academic work, cite the MSc thesis it accompanies:

> Lukwichi, J. (2026). *Optimising Hospital Supply Chain Demand Forecasting Using Machine Learning: A Case Study of Steve Biko Academic Hospital Emergency Department.* MSc Thesis, Department of Industrial and Systems Engineering, University of Pretoria.

---

## Acknowledgements

- **Prof. Bean** (University of Pretoria) — thesis supervision.
- **Steve Biko Academic Hospital** — Emergency Department register data.
- **NDOH / NICD** — public health data context.

---

## License

Academic use; see `LICENSE` for the full text.
