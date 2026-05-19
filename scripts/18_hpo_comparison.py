"""Fair HPO comparison — 3 methods x 3 models = 9 cells. CRITERION = RMSE.

The optimisation objective is mean cv_RMSE across folds. Once the winning
candidate is selected by RMSE, we ALSO report its cv_MAPE, cv_MAE, cv_R2 so
the chapter can show whether the RMSE-winner is competitive on every metric.

For each model in {XGBoost, ANN, LSTM} and each method in {Grid, Random, Optuna}:
  - Same 5-fold rolling-origin inner CV (subsampled from the 69-fold pool)
  - Same parameter search space
  - Same RMSE selection criterion (this run)
  - Same trial budget = 10 candidates

The trial budget is the only way to compare methods fairly. Grid samples 10
combos from a structured discrete grid; Random samples 10 uniform draws from
the same parameter ranges; Optuna runs 10 TPE trials.

Saves:
  artefacts/metrics/hpo_comparison.csv    -- per-cell summary
  artefacts/metrics/hpo_comparison_full.csv -- every trial across the 9 cells
  artefacts/figures/fig_6_hpo_comparison.png

Run time: ~100 minutes (LSTM dominates).
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings
import random
from itertools import product

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.cv import subsampled_rolling_origin
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")
SEED = 42
N_TRIALS = 10
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Search spaces (shared across the 3 methods per model)
# ---------------------------------------------------------------------------

XGB_SPACE = {
    "n_estimators":  [100, 200, 300, 500],
    "max_depth":     [3, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample":     [0.7, 0.85, 1.0],
}
# A compact 10-point grid for the grid-search variant
XGB_GRID10 = [
    (100, 3, 0.05, 1.0),
    (100, 3, 0.01, 1.0),
    (200, 3, 0.05, 1.0),
    (300, 3, 0.05, 1.0),
    (500, 3, 0.01, 1.0),
    (500, 3, 0.05, 1.0),
    (200, 5, 0.05, 0.85),
    (300, 5, 0.05, 1.0),
    (100, 5, 0.05, 1.0),
    (300, 3, 0.05, 0.85),
]

ANN_SPACE = {
    "hidden_layers": [1, 2],
    "units":         [64, 128, 192, 256],
    "dropout":       [0.1, 0.2, 0.3, 0.4],
    # learning_rate is log-uniform 5e-4 .. 1e-2 in random/optuna
    "batch_size":    [16, 32, 64],
}
ANN_GRID10 = [
    {"hidden_layers": 1, "units": 128, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    {"hidden_layers": 1, "units": 192, "dropout": 0.1, "learning_rate": 0.005, "batch_size": 32},
    {"hidden_layers": 1, "units": 256, "dropout": 0.2, "learning_rate": 0.005, "batch_size": 32},
    {"hidden_layers": 2, "units": 128, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    {"hidden_layers": 2, "units": 192, "dropout": 0.2, "learning_rate": 0.002, "batch_size": 32},
    {"hidden_layers": 2, "units": 256, "dropout": 0.2, "learning_rate": 0.005, "batch_size": 32},
    {"hidden_layers": 2, "units": 256, "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64},
    {"hidden_layers": 1, "units": 128, "dropout": 0.3, "learning_rate": 0.002, "batch_size": 64},
    {"hidden_layers": 2, "units": 64,  "dropout": 0.1, "learning_rate": 0.005, "batch_size": 16},
    {"hidden_layers": 1, "units": 256, "dropout": 0.4, "learning_rate": 0.001, "batch_size": 32},
]

LSTM_SPACE = {
    "lookback":      [14, 21],   # reduced from {14, 21, 28} for time
    "units":         [64, 128],  # reduced from {64,96,128,192,256}
    "dropout":       [0.1, 0.2, 0.3],
    "batch_size":    [32, 64],
}
LSTM_GRID10 = [
    {"lookback": 14, "units": 64,  "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    {"lookback": 14, "units": 128, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    {"lookback": 14, "units": 128, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64},
    {"lookback": 14, "units": 128, "dropout": 0.1, "learning_rate": 0.001, "batch_size": 64},
    {"lookback": 21, "units": 64,  "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    {"lookback": 21, "units": 128, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    {"lookback": 21, "units": 128, "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64},
    {"lookback": 14, "units": 128, "dropout": 0.3, "learning_rate": 0.002, "batch_size": 32},
    {"lookback": 21, "units": 64,  "dropout": 0.1, "learning_rate": 0.0005,"batch_size": 32},
    {"lookback": 14, "units": 64,  "dropout": 0.2, "learning_rate": 0.005, "batch_size": 64},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

splits = Splits.from_config()
g1 = load_g1()
target = g1["total_daily_arrivals"]
eng = load_engineered()
X_consensus = build_selected_X(eng)
df_full = pd.concat([target.rename("y"), X_consensus], axis=1, join="inner").dropna()
train_idx = splits.slice(g1, "train").index.intersection(df_full.index)
X_train = df_full.loc[train_idx].drop(columns=["y"])
y_train = df_full.loc[train_idx, "y"]

# Shared folds
folds = subsampled_rolling_origin(train_idx, n_folds=N_FOLDS, horizon_days=7,
                                    step_days=7, min_train_days=365)
print(f"Train: {len(train_idx)} days, {len(folds)} folds, {N_TRIALS} trials/method\n")


# ---------------------------------------------------------------------------
# Single-fit evaluators (called by every HPO method)
# ---------------------------------------------------------------------------

def xgb_cv_rmse(params: dict) -> dict:
    """Returns dict with cv_RMSE (primary), plus cv_MAPE, cv_MAE, cv_R2."""
    from xgboost import XGBRegressor
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        m = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=SEED, verbosity=0, n_jobs=-1)
        m.fit(X_train.iloc[f.train_idx].values, y_train.iloc[f.train_idx].values)
        yhat = m.predict(X_train.iloc[f.test_idx].values)
        s = score(y_train.iloc[f.test_idx].values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.mean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


def ann_cv_rmse(params: dict) -> dict:
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        _seed_everything(SEED)
        X_tr_fold = X_train.iloc[f.train_idx]
        y_tr_fold = y_train.iloc[f.train_idx]
        X_te_fold = X_train.iloc[f.test_idx]
        y_te_fold = y_train.iloc[f.test_idx]
        mean = X_tr_fold.mean(); std = X_tr_fold.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr_fold - mean) / std).astype(np.float32).values
        Xte = ((X_te_fold - mean) / std).astype(np.float32).values
        y_mean = float(y_tr_fold.mean()); y_std = float(y_tr_fold.std(ddof=0) or 1.0)
        ytr = ((y_tr_fold - y_mean) / y_std).astype(np.float32).values

        n_es = max(28, len(Xtr) // 6)
        n_hidden = [params["units"]] * params["hidden_layers"]
        model = _MLP(n_in=Xtr.shape[1], n_hidden=n_hidden, dropout=params["dropout"])
        Xtr_t = torch.from_numpy(Xtr[:-n_es]); ytr_t = torch.from_numpy(ytr[:-n_es])
        Xes_t = torch.from_numpy(Xtr[-n_es:]); yes_t = torch.from_numpy(ytr[-n_es:])
        model, _, _ = _train_one(Xtr_t, ytr_t, Xes_t, yes_t,
                                   {**params, "seed": SEED}, max_epochs=60)
        model.eval()
        with torch.no_grad():
            yhat_n = model(torch.from_numpy(Xte)).numpy()
        yhat = yhat_n * y_std + y_mean
        s = score(y_te_fold.values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.mean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


def lstm_cv_rmse(params: dict) -> dict:
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        _seed_everything(SEED)
        X_tr_fold = X_train.iloc[f.train_idx]
        y_tr_fold = y_train.iloc[f.train_idx]
        X_te_fold = X_train.iloc[f.test_idx]
        y_te_fold = y_train.iloc[f.test_idx]
        mean = X_tr_fold.mean(); std = X_tr_fold.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr_fold - mean) / std).astype(np.float32).values
        Xte = ((X_te_fold - mean) / std).astype(np.float32).values
        y_mean = float(y_tr_fold.mean()); y_std = float(y_tr_fold.std(ddof=0) or 1.0)
        ytr = ((y_tr_fold - y_mean) / y_std).astype(np.float32).values
        lookback = params["lookback"]
        Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
        n_es = max(28, lookback + 7)
        model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                                Xtr_seq[-n_es:], ytr_seq[-n_es:],
                                {**params, "seed": SEED}, max_epochs=25)
        # Predict the test window via sliding window
        full_X = np.vstack([Xtr, Xte])
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(len(Xte)):
                pos = len(Xtr) + i
                window = full_X[pos - lookback : pos]
                preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
        yhat = np.array(preds) * y_std + y_mean
        s = score(y_te_fold.values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.mean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


# ---------------------------------------------------------------------------
# HPO methods (returns list of trial dicts + best params)
# ---------------------------------------------------------------------------

def grid_search(grid: list, eval_fn) -> list[dict]:
    rows = []
    for i, params in enumerate(grid[:N_TRIALS]):
        if isinstance(params, tuple):
            p = dict(zip(["n_estimators","max_depth","learning_rate","subsample"], params))
        else:
            p = params.copy()
        t0 = time.time()
        s = eval_fn(p)
        rows.append({"trial": i, "method": "Grid", **p, **s,
                      "time_s": round(time.time()-t0, 1)})
    return rows


def random_search(space: dict, eval_fn, model_name: str) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(N_TRIALS):
        if model_name == "xgboost":
            p = {"n_estimators": int(rng.choice(space["n_estimators"])),
                 "max_depth": int(rng.choice(space["max_depth"])),
                 "learning_rate": float(rng.choice(space["learning_rate"])),
                 "subsample": float(rng.choice(space["subsample"]))}
        elif model_name == "ann":
            p = {"hidden_layers": int(rng.choice(space["hidden_layers"])),
                 "units": int(rng.choice(space["units"])),
                 "dropout": float(rng.choice(space["dropout"])),
                 "learning_rate": float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
                 "batch_size": int(rng.choice(space["batch_size"]))}
        else:
            p = {"lookback": int(rng.choice(space["lookback"])),
                 "units": int(rng.choice(space["units"])),
                 "dropout": float(rng.choice(space["dropout"])),
                 "learning_rate": float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
                 "batch_size": int(rng.choice(space["batch_size"]))}
        t0 = time.time()
        s = eval_fn(p)
        rows.append({"trial": i, "method": "Random", **p, **s,
                      "time_s": round(time.time()-t0, 1)})
    return rows


def optuna_search(space: dict, eval_fn, model_name: str) -> list[dict]:
    import optuna
    from optuna.samplers import TPESampler
    rows = []

    def objective(trial):
        if model_name == "xgboost":
            p = {"n_estimators": trial.suggest_categorical("n_estimators", space["n_estimators"]),
                 "max_depth": trial.suggest_categorical("max_depth", space["max_depth"]),
                 "learning_rate": trial.suggest_categorical("learning_rate", space["learning_rate"]),
                 "subsample": trial.suggest_categorical("subsample", space["subsample"])}
        elif model_name == "ann":
            p = {"hidden_layers": trial.suggest_categorical("hidden_layers", space["hidden_layers"]),
                 "units": trial.suggest_categorical("units", space["units"]),
                 "dropout": trial.suggest_categorical("dropout", space["dropout"]),
                 "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True),
                 "batch_size": trial.suggest_categorical("batch_size", space["batch_size"])}
        else:
            p = {"lookback": trial.suggest_categorical("lookback", space["lookback"]),
                 "units": trial.suggest_categorical("units", space["units"]),
                 "dropout": trial.suggest_categorical("dropout", space["dropout"]),
                 "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True),
                 "batch_size": trial.suggest_categorical("batch_size", space["batch_size"])}
        t0 = time.time()
        s = eval_fn(p)
        rows.append({"trial": trial.number, "method": "Optuna", **p, **s,
                      "time_s": round(time.time()-t0, 1)})
        return s["cv_RMSE"]   # <-- objective is RMSE now

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return rows


# ---------------------------------------------------------------------------
# Run the 9 cells
# ---------------------------------------------------------------------------

def run_all(target_models: list[str]):
    all_rows = []
    summary_rows = []

    def _process(model_name: str, eval_fn, grid10, space):
        for method, runner in [
            ("Grid",   lambda: grid_search(grid10, eval_fn)),
            ("Random", lambda: random_search(space, eval_fn, model_name.lower())),
            ("Optuna", lambda: optuna_search(space, eval_fn, model_name.lower())),
        ]:
            print(f"\n=== {model_name} / {method} (RMSE-objective, 10 trials x 5 folds) ===")
            t0 = time.time()
            trials = runner()
            for t in trials: t["model"] = model_name
            all_rows.extend(trials)
            # ★ Winner is chosen by cv_RMSE (not cv_MAPE) per user request
            best = min(trials, key=lambda r: r["cv_RMSE"])
            elapsed = time.time() - t0
            summary_rows.append({
                "model": model_name, "method": method,
                "best_cv_RMSE": best["cv_RMSE"],
                "best_cv_MAPE": best["cv_MAPE"],
                "best_cv_MAE":  best["cv_MAE"],
                "best_cv_R2":   best["cv_R2"],
                "time_min": round(elapsed / 60, 2),
                "best_params": {k:v for k,v in best.items()
                                  if k not in ("trial","method","cv_MAPE","cv_RMSE",
                                                "cv_MAE","cv_R2","time_s","model")},
            })
            print(f"  best cv_RMSE = {best['cv_RMSE']:.3f}   "
                  f"(its cv_MAPE = {best['cv_MAPE']:.2f}%, "
                  f"cv_MAE = {best['cv_MAE']:.2f}, "
                  f"cv_R2 = {best['cv_R2']:+.3f})   "
                  f"[{method}, {elapsed:.0f}s]")

    if "xgboost" in target_models:
        _process("XGBoost", xgb_cv_rmse, XGB_GRID10, XGB_SPACE)
    if "ann" in target_models:
        _process("ANN", ann_cv_rmse, ANN_GRID10, ANN_SPACE)
    if "lstm" in target_models:
        _process("LSTM", lstm_cv_rmse, LSTM_GRID10, LSTM_SPACE)

    full_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)
    out_full = ROOT / "artefacts" / "metrics" / "hpo_comparison_full.csv"
    out_sum = ROOT / "artefacts" / "metrics" / "hpo_comparison.csv"
    out_full.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_full, index=False)
    summary_df.to_csv(out_sum, index=False)
    print(f"\nWrote: {out_full.relative_to(ROOT)}  ({len(full_df)} trials)")
    print(f"Wrote: {out_sum.relative_to(ROOT)}  ({len(summary_df)} cells)")

    if not summary_df.empty:
        print("\n=== SUMMARY (best cv_MAPE per cell) ===")
        pd.set_option("display.float_format", lambda v: f"{v:.3f}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*",
                    default=["xgboost", "ann", "lstm"],
                    help="Subset: xgboost, ann, lstm (default: all)")
    args = p.parse_args()
    run_all(args.models)
