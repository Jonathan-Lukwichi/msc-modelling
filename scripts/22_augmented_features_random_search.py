"""Augmented-feature Random Search experiment.

Combines:
  - The augmented feature set:  §3.4.3 consensus 23 features + lagged
    internal-hospital signals (carry-over, P1/P2/P3 acuity, transfers,
    specialty counts, attendant count, discharges).
  - The empirically best optimizer:  Random Search (from §18 audit).
  - The chapter's deployment criterion:  minimise mean cv_RMSE
    (also report cv_MAPE, cv_MAE, cv_R2 for the winner).
  - All three ML/DL families:  XGBoost, ANN, LSTM.

Then refits the winner on full train and runs rolling weekly refit on val
and test. Compares against the current best (XGBoost RMSE-tuned: val 11.99%,
test 12.63%).

The candidate clinical signals are LAGGED by {1, 7} days to avoid leakage —
yesterday's clinical state predicts today's arrivals.
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, load_g3, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.cv import subsampled_rolling_origin
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")

SEED = 42
N_TRIALS = 10
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Augmented feature builder
# ---------------------------------------------------------------------------

# High-corr-with-target clinical signals from G1_fully_engineered.csv
CLINICAL_SIGNALS = [
    "carry_over_midnight",       # corr 0.40 -- state of the ED queue
    "p2_normal_hours",           # corr 0.53 -- yesterday's bulk acuity
    "p2_after_hours",            # corr 0.37
    "p1_after_hours",            # corr 0.11 -- urgent night-time
    "p3_normal_hours",           # corr 0.25
    "p3_after_hours",            # corr 0.26
    "attendant_count",           # corr 0.54
    "discharges_rht_abscond",    # corr 0.46
    "transfer_in_subtotal",      # corr 0.55 -- external pressure
    "external_transfer_in",      # corr 0.09 but novel info
    "internal_transfer_out",     # corr 0.11
]

# Specialty counts from G3
G3_SIGNALS = ["spec_medicine", "spec_orthopaedics", "spec_surgery",
                "spec_paediatrics", "spec_gynae"]


def build_augmented_features():
    """Construct the augmented feature matrix.

    Returns: (X, target, train_idx, val_idx, test_idx)
    """
    splits = Splits.from_config()
    g1 = load_g1()
    g3 = load_g3()
    eng = load_engineered()

    target = g1["total_daily_arrivals"]
    X_cons = build_selected_X(eng)

    # Build lagged clinical features
    lags = [1, 7]
    lagged = pd.DataFrame(index=eng.index)
    for col in CLINICAL_SIGNALS:
        if col in eng.columns:
            for lag in lags:
                lagged[f"{col}_lag_{lag}"] = eng[col].shift(lag)
    # G3 specialty lags
    g3_aligned = g3.reindex(eng.index)
    for col in G3_SIGNALS:
        if col in g3_aligned.columns:
            for lag in lags:
                lagged[f"{col}_lag_{lag}"] = g3_aligned[col].shift(lag)

    # Combine with consensus
    X_aug = X_cons.join(lagged, how="left")
    print(f"Consensus features: {X_cons.shape[1]}  |  "
          f"Lagged clinical added: {lagged.shape[1]}  |  "
          f"Total: {X_aug.shape[1]}")

    # Inner-join with target and drop rows with NaN
    df = pd.concat([target.rename("y"), X_aug], axis=1, join="inner").dropna()
    train_idx = splits.slice(g1, "train").index.intersection(df.index)
    val_idx = splits.slice(g1, "val").index.intersection(df.index)
    test_idx = splits.slice(g1, "test").index.intersection(df.index)
    print(f"After NaN drop: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    return df, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Search spaces (same as §18 audit)
# ---------------------------------------------------------------------------

XGB_SPACE = {
    "n_estimators":  [100, 200, 300, 500],
    "max_depth":     [3, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample":     [0.7, 0.85, 1.0],
}
ANN_SPACE = {
    "hidden_layers": [1, 2],
    "units":         [64, 128, 192, 256],
    "dropout":       [0.1, 0.2, 0.3, 0.4],
    "batch_size":    [16, 32, 64],
}
LSTM_SPACE = {
    "lookback":      [14, 21],
    "units":         [64, 128],
    "dropout":       [0.1, 0.2, 0.3],
    "batch_size":    [32, 64],
}


# ---------------------------------------------------------------------------
# Per-model CV evaluator (minimise cv_RMSE; report all metrics)
# ---------------------------------------------------------------------------

def xgb_cv(params, X, y, folds):
    from xgboost import XGBRegressor
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        m = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=SEED, verbosity=0, n_jobs=-1)
        m.fit(X.iloc[f.train_idx].values, y.iloc[f.train_idx].values)
        yhat = m.predict(X.iloc[f.test_idx].values)
        s = score(y.iloc[f.test_idx].values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.mean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


def ann_cv(params, X, y, folds):
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        _seed_everything(SEED)
        X_tr, X_te = X.iloc[f.train_idx], X.iloc[f.test_idx]
        y_tr, y_te = y.iloc[f.train_idx], y.iloc[f.test_idx]
        mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr - mean) / std).astype(np.float32).values
        Xte = ((X_te - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std(ddof=0) or 1.0)
        ytr = ((y_tr - y_mean) / y_std).astype(np.float32).values
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
        s = score(y_te.values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.mean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


def lstm_cv(params, X, y, folds):
    import torch
    from src.forecasting.models.lstm import _build_sequences, _train_one, _seed_everything
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        _seed_everything(SEED)
        X_tr, X_te = X.iloc[f.train_idx], X.iloc[f.test_idx]
        y_tr, y_te = y.iloc[f.train_idx], y.iloc[f.test_idx]
        mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr - mean) / std).astype(np.float32).values
        Xte = ((X_te - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std(ddof=0) or 1.0)
        ytr = ((y_tr - y_mean) / y_std).astype(np.float32).values
        lookback = params["lookback"]
        Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
        n_es = max(28, lookback + 7)
        model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                                Xtr_seq[-n_es:], ytr_seq[-n_es:],
                                {**params, "seed": SEED}, max_epochs=25)
        full = np.vstack([Xtr, Xte])
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(len(Xte)):
                pos = len(Xtr) + i
                window = full[pos - lookback : pos]
                preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
        yhat = np.array(preds) * y_std + y_mean
        s = score(y_te.values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.mean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


# ---------------------------------------------------------------------------
# Random Search HPO (minimise cv_RMSE)
# ---------------------------------------------------------------------------

def random_search(space, eval_fn, X_train, y_train, folds, model_name):
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
        s = eval_fn(p, X_train, y_train, folds)
        rows.append({"trial": i, **p, **s, "time_s": round(time.time() - t0, 1)})
    rows.sort(key=lambda r: r["cv_RMSE"])
    return rows


# ---------------------------------------------------------------------------
# Rolling weekly refit on val + test with the winning params
# ---------------------------------------------------------------------------

def rolling_xgb(target, X_full, y_full, target_idx, best_params):
    from xgboost import XGBRegressor
    rows = []
    origin = X_full.index[X_full.index < target_idx[0]][-1]
    remaining = list(target_idx)
    while remaining:
        h = min(7, len(remaining))
        future = remaining[:h]
        tr_dates = X_full.index[X_full.index <= origin]
        m = XGBRegressor(**best_params, objective="reg:squarederror",
                          random_state=SEED, verbosity=0, n_jobs=-1)
        m.fit(X_full.loc[tr_dates].values, y_full.loc[tr_dates].values)
        yhat = m.predict(X_full.loc[future].values)
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        remaining = remaining[h:]
    return pd.DataFrame(rows).set_index("date")


def rolling_ann(target, X_full, y_full, target_idx, best_params):
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    rows = []
    origin = X_full.index[X_full.index < target_idx[0]][-1]
    remaining = list(target_idx)
    while remaining:
        h = min(7, len(remaining))
        future = remaining[:h]
        tr_dates = X_full.index[X_full.index <= origin]
        X_tr = X_full.loc[tr_dates]; y_tr = y_full.loc[tr_dates]
        mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr - mean) / std).astype(np.float32).values
        Xfu = ((X_full.loc[future] - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std(ddof=0) or 1.0)
        ytr = ((y_tr - y_mean) / y_std).astype(np.float32).values
        _seed_everything(SEED)
        n_es = max(28, len(Xtr) // 6)
        n_hidden = [best_params["units"]] * best_params["hidden_layers"]
        model = _MLP(n_in=Xtr.shape[1], n_hidden=n_hidden, dropout=best_params["dropout"])
        Xtr_t = torch.from_numpy(Xtr[:-n_es]); ytr_t = torch.from_numpy(ytr[:-n_es])
        Xes_t = torch.from_numpy(Xtr[-n_es:]); yes_t = torch.from_numpy(ytr[-n_es:])
        model, _, _ = _train_one(Xtr_t, ytr_t, Xes_t, yes_t,
                                   {**best_params, "seed": SEED}, max_epochs=80)
        model.eval()
        with torch.no_grad():
            yh = model(torch.from_numpy(Xfu)).numpy()
        yhat = yh * y_std + y_mean
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        remaining = remaining[h:]
    return pd.DataFrame(rows).set_index("date")


def rolling_lstm(target, X_full, y_full, target_idx, best_params):
    import torch
    from src.forecasting.models.lstm import _build_sequences, _train_one, _seed_everything
    rows = []
    origin = X_full.index[X_full.index < target_idx[0]][-1]
    remaining = list(target_idx)
    lookback = best_params["lookback"]
    while remaining:
        h = min(7, len(remaining))
        future = remaining[:h]
        tr_dates = X_full.index[X_full.index <= origin]
        X_tr = X_full.loc[tr_dates]; y_tr = y_full.loc[tr_dates]
        mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr - mean) / std).astype(np.float32).values
        Xfu = ((X_full.loc[future] - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std(ddof=0) or 1.0)
        ytr = ((y_tr - y_mean) / y_std).astype(np.float32).values
        _seed_everything(SEED)
        Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
        n_es = max(28, lookback + 7)
        model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                                Xtr_seq[-n_es:], ytr_seq[-n_es:],
                                {**best_params, "seed": SEED}, max_epochs=30)
        full = np.vstack([Xtr, Xfu])
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(len(Xfu)):
                pos = len(Xtr) + i
                window = full[pos - lookback : pos]
                preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
        yhat = np.array(preds) * y_std + y_mean
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        remaining = remaining[h:]
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("AUGMENTED FEATURES + RANDOM SEARCH (min cv_RMSE) — 3 models")
    print("=" * 70 + "\n")

    df, train_idx, val_idx, test_idx = build_augmented_features()
    X_full = df.drop(columns=["y"])
    y_full = df["y"]
    X_train = X_full.loc[train_idx]
    y_train = y_full.loc[train_idx]
    folds = subsampled_rolling_origin(train_idx, n_folds=N_FOLDS,
                                        horizon_days=7, step_days=7,
                                        min_train_days=365)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}  Folds: {len(folds)}\n")

    final = []

    for model_name, eval_fn, runner in [
        ("XGBoost", xgb_cv, rolling_xgb),
        ("ANN",     ann_cv, rolling_ann),
        ("LSTM",    lstm_cv, rolling_lstm),
    ]:
        print("\n" + "=" * 70)
        print(f"{model_name} — Random Search (10 trials) on augmented features")
        print("=" * 70)
        t0 = time.time()
        space = (XGB_SPACE if model_name == "XGBoost"
                  else ANN_SPACE if model_name == "ANN" else LSTM_SPACE)
        trials = random_search(space, eval_fn, X_train, y_train, folds,
                                 model_name.lower())
        best = trials[0]
        print(f"\nHPO completed in {(time.time() - t0)/60:.1f} min")
        print(f"Best params: " + str({k: v for k, v in best.items()
                                          if k not in ("trial", "cv_RMSE", "cv_MAPE",
                                                          "cv_MAE", "cv_R2", "time_s")}))
        print(f"  cv_RMSE={best['cv_RMSE']:.3f}  cv_MAPE={best['cv_MAPE']:.2f}%  "
              f"cv_MAE={best['cv_MAE']:.2f}  cv_R2={best['cv_R2']:+.3f}")

        # Rolling val + test
        best_params = {k: v for k, v in best.items()
                        if k not in ("trial", "cv_RMSE", "cv_MAPE", "cv_MAE",
                                       "cv_R2", "time_s")}
        print(f"\nRolling val refit...")
        t0 = time.time()
        val_pred = runner(y_full, X_full, y_full, val_idx, best_params)
        val_pred["actual"] = y_full.loc[val_pred.index]
        val_s = score(val_pred["actual"], val_pred["predicted"])
        print(f"  Val ({time.time() - t0:.0f}s):  MAPE={val_s['MAPE']:.3f}  "
              f"MAE={val_s['MAE']:.2f}  RMSE={val_s['RMSE']:.2f}  R2={val_s['R2']:+.3f}")

        print(f"Rolling test refit...")
        t0 = time.time()
        test_pred = runner(y_full, X_full, y_full, test_idx, best_params)
        test_pred["actual"] = y_full.loc[test_pred.index]
        test_s = score(test_pred["actual"], test_pred["predicted"])
        print(f"  Test ({time.time() - t0:.0f}s): MAPE={test_s['MAPE']:.3f}  "
              f"MAE={test_s['MAE']:.2f}  RMSE={test_s['RMSE']:.2f}  R2={test_s['R2']:+.3f}")

        # Save predictions
        val_pred.reset_index().to_csv(
            ROOT / "artefacts" / "predictions" / f"{model_name.lower()}_augmented.csv",
            index=False)
        test_pred.reset_index().to_csv(
            ROOT / "artefacts" / "predictions" / "test" / f"{model_name.lower()}_augmented.csv",
            index=False)

        final.append({
            "model": model_name,
            "best_params": best_params,
            "cv_RMSE": best["cv_RMSE"], "cv_MAPE": best["cv_MAPE"],
            "val_MAPE": val_s["MAPE"], "val_MAE": val_s["MAE"],
            "val_RMSE": val_s["RMSE"], "val_R2": val_s["R2"],
            "test_MAPE": test_s["MAPE"], "test_MAE": test_s["MAE"],
            "test_RMSE": test_s["RMSE"], "test_R2": test_s["R2"],
        })

    # Save summary
    out = pd.DataFrame(final)
    out.to_csv(ROOT / "artefacts" / "metrics" / "augmented_random_search.csv",
                index=False)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — augmented features + Random Search (min cv_RMSE)")
    print("=" * 70)
    print(out[["model", "cv_RMSE", "cv_MAPE", "val_MAPE", "test_MAPE",
                "val_RMSE", "test_RMSE"]].to_string(index=False))

    print("\nCompare with main study (RMSE-tuned, NO augmentation):")
    print("  XGBoost:  val MAPE 11.99, test MAPE 12.63, val RMSE 9.35, test RMSE 10.30")
    print("  ANN:      val MAPE 11.90, test MAPE 13.24, val RMSE 9.34, test RMSE 11.34")
    print("  LSTM:     val MAPE 12.31, test MAPE 13.76, val RMSE 9.40, test RMSE 11.35")


if __name__ == "__main__":
    main()
