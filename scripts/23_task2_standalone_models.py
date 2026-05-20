"""Task 2 — per-specialty daily forecast with all 6 standalone models.

For each of the five daily specialties (Medicine, Orthopaedics, Surgery,
Paediatrics, Gynaecology) we run:

    ARIMA, SARIMAX, NB-GLM, XGBoost, ANN, LSTM

on the **raw daily count** target (e.g., spec_medicine), exactly mirroring
the Task 1 standalone leaderboard. ML/DL families use Random Search HPO
(the best optimizer per §4quater audit) with min cv_RMSE objective on 5
rolling-origin folds inside the train block.

Features:
  - ARIMA           — no exogenous (univariate AR-I-MA)
  - SARIMAX, NB-GLM — per-specialty exog from configs/features_task2.yaml
                      (DoW + 7 §5.2.5 calendar binaries + specialty-specific
                       weather and Surgery sign-reversal interaction columns)
  - XGBoost, ANN, LSTM — §3.4.3 consensus 23 features (same as Task 1)

Evaluation: rolling weekly refit on val (184 days) AND test (396 days),
floored at 0, scored by MAPE/MAE/RMSE/R2.

Outputs:
  - artefacts/predictions/task2_{specialty}_{model}.csv        (val)
  - artefacts/predictions/test/task2_{specialty}_{model}.csv   (test)
  - artefacts/metrics/task2_standalone_metrics.csv             (consolidated)
  - artefacts/metrics/task2_standalone_hpo_traces.csv          (RS trial log)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, load_g3, Splits
from src.forecasting.features import build_task2_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.cv import subsampled_rolling_origin
from src.forecasting.metrics import score
from src.forecasting.models import arima as arima_m
from src.forecasting.models import sarimax as sarimax_m
from src.forecasting.models import negbin as nb_m

warnings.filterwarnings("ignore")

SEED = 42
N_TRIALS = 10
N_FOLDS = 5

DAILY_SPECIALTIES = {
    "Medicine":     "spec_medicine",
    "Orthopaedics": "spec_orthopaedics",
    "Surgery":      "spec_surgery",
    "Paediatrics":  "spec_paediatrics",
    "Gynaecology":  "spec_gynae",
}

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
# Per-specialty data assembly
# ---------------------------------------------------------------------------

def assemble_data():
    """Load g1 (for train/val/test bounds), g3 (per-specialty target), and the
    engineered/consensus matrix (for ML).
    """
    splits = Splits.from_config()
    g1 = load_g1()
    g3 = load_g3()
    eng = load_engineered()
    X_cons = build_selected_X(eng)

    # Align on common dates after consensus engineering (drops early lag NaNs)
    common = g3.index.intersection(X_cons.index)
    # Restrict to post-COVID train_start onwards (mirror Task 1 setup)
    common = common[(common >= splits.train_start) & (common <= splits.test_end)]
    g3 = g3.loc[common].copy()
    X_cons = X_cons.loc[common].copy()

    train_idx = splits.slice(g3, "train").index
    val_idx = splits.slice(g3, "val").index
    test_idx = splits.slice(g3, "test").index
    print(f"Post-COVID dates: {len(common)}  | "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    return g3, X_cons, train_idx, val_idx, test_idx, splits


# ---------------------------------------------------------------------------
# Random Search CV evaluators (minimise cv_RMSE)
# ---------------------------------------------------------------------------

def xgb_cv(params, X, y, folds):
    from xgboost import XGBRegressor
    mapes, maes, rmses, r2s = [], [], [], []
    for f in folds:
        m = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=SEED, verbosity=0, n_jobs=-1)
        m.fit(X.iloc[f.train_idx].values, y.iloc[f.train_idx].values)
        yhat = np.maximum(m.predict(X.iloc[f.test_idx].values), 0)
        s = score(y.iloc[f.test_idx].values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.nanmean(mapes)),
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
        yhat = np.maximum(yhat_n * y_std + y_mean, 0)
        s = score(y_te.values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.nanmean(mapes)),
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
        yhat = np.maximum(np.array(preds) * y_std + y_mean, 0)
        s = score(y_te.values, yhat)
        mapes.append(s["MAPE"]); maes.append(s["MAE"])
        rmses.append(s["RMSE"]); r2s.append(s["R2"])
    return {"cv_RMSE": float(np.mean(rmses)),
             "cv_MAPE": float(np.nanmean(mapes)),
             "cv_MAE":  float(np.mean(maes)),
             "cv_R2":   float(np.mean(r2s))}


def random_search(space, eval_fn, X_train, y_train, folds, model_name, n_trials=N_TRIALS):
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(n_trials):
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
        else:  # lstm
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
# Rolling weekly refit on val + test with the winning params (count target)
# ---------------------------------------------------------------------------

def rolling_xgb(X_full, y_full, target_idx, best_params):
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
        yhat = np.maximum(m.predict(X_full.loc[future].values), 0)
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        remaining = remaining[h:]
    return pd.DataFrame(rows).set_index("date")


def rolling_ann(X_full, y_full, target_idx, best_params):
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
        yhat = np.maximum(yh * y_std + y_mean, 0)
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        remaining = remaining[h:]
    return pd.DataFrame(rows).set_index("date")


def rolling_lstm(X_full, y_full, target_idx, best_params):
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
        yhat = np.maximum(np.array(preds) * y_std + y_mean, 0)
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        remaining = remaining[h:]
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Per-specialty runner — all 6 models
# ---------------------------------------------------------------------------

def run_specialty(name: str, col: str, g3: pd.DataFrame,
                   X_cons: pd.DataFrame, train_idx, val_idx, test_idx) -> tuple[list, list]:
    """Run all 6 standalone models for one specialty. Returns (metric_rows, trial_rows)."""
    print("\n" + "=" * 70)
    print(f"SPECIALTY: {name}  (G3 col: {col})")
    print("=" * 70)

    y_full = g3[col].astype(float)
    mean_per_day = y_full.loc[train_idx].mean()
    print(f"Mean count on train: {mean_per_day:.2f}/day  "
          f"(zeros: {(y_full.loc[train_idx] == 0).sum()}/{len(train_idx)})")

    # Per-specialty exog (DoW + calendar + weather + interactions)
    X_train_spec, sc = build_task2_exogenous(g3.loc[train_idx], name,
                                                 fit_scaler=True)
    X_full_spec, _ = build_task2_exogenous(g3, name, scaler=sc)
    print(f"Per-specialty exog cols ({X_full_spec.shape[1]}): {list(X_full_spec.columns)}")
    print(f"Consensus exog cols (for ML): {X_cons.shape[1]}")

    metric_rows = []
    trial_rows = []
    out_pred = ROOT / "artefacts" / "predictions"
    out_pred_test = out_pred / "test"
    out_pred_test.mkdir(parents=True, exist_ok=True)

    # ---- (1) ARIMA ---------------------------------------------------------
    print(f"\n--- ARIMA on {name} ---")
    t0 = time.time()
    try:
        ar = arima_m.pick_order(y_full.loc[train_idx])
        print(f"  Picked ARIMA{ar.order}, AIC={ar.aic:.1f}")
        for block, blk_idx in [("val", val_idx), ("test", test_idx)]:
            pred = arima_m.rolling_forecast(y_full, blk_idx, ar.order, step_days=7)
            pred = pred.set_index("date")
            pred["predicted"] = pred["predicted"].clip(lower=0)
            pred["actual"] = y_full.loc[pred.index]
            s = score(pred["actual"], pred["predicted"])
            print(f"  {block}: MAPE={s['MAPE']:.2f}  MAE={s['MAE']:.2f}  "
                  f"RMSE={s['RMSE']:.2f}  R2={s['R2']:+.3f}")
            target_dir = out_pred if block == "val" else out_pred_test
            pred.reset_index().to_csv(
                target_dir / f"task2_{name}_arima.csv", index=False)
            metric_rows.append({"specialty": name, "model": "ARIMA",
                                  "block": block, **s,
                                  "order": str(ar.order), "params": ""})
    except Exception as exc:
        print(f"  ARIMA FAILED: {exc}")
    print(f"  ARIMA took {time.time() - t0:.0f}s")

    # ---- (2) SARIMAX -------------------------------------------------------
    print(f"\n--- SARIMAX on {name} ---")
    t0 = time.time()
    try:
        sm = sarimax_m.pick_order(y_full.loc[train_idx],
                                     X_train_spec.loc[train_idx])
        print(f"  Picked SARIMAX{sm.order}x{sm.seasonal_order}, AIC={sm.aic:.1f}")
        for block, blk_idx in [("val", val_idx), ("test", test_idx)]:
            pred = sarimax_m.rolling_forecast(y_full, X_full_spec, blk_idx,
                                                   sm.order, sm.seasonal_order,
                                                   step_days=7)
            pred = pred.set_index("date")
            pred["predicted"] = pred["predicted"].clip(lower=0)
            pred["actual"] = y_full.loc[pred.index]
            s = score(pred["actual"], pred["predicted"])
            print(f"  {block}: MAPE={s['MAPE']:.2f}  MAE={s['MAE']:.2f}  "
                  f"RMSE={s['RMSE']:.2f}  R2={s['R2']:+.3f}")
            target_dir = out_pred if block == "val" else out_pred_test
            pred.reset_index().to_csv(
                target_dir / f"task2_{name}_sarimax.csv", index=False)
            metric_rows.append({"specialty": name, "model": "SARIMAX",
                                  "block": block, **s,
                                  "order": f"{sm.order}x{sm.seasonal_order}",
                                  "params": ""})
    except Exception as exc:
        print(f"  SARIMAX FAILED: {exc}")
    print(f"  SARIMAX took {time.time() - t0:.0f}s")

    # ---- (3) NB-GLM --------------------------------------------------------
    print(f"\n--- NB-GLM on {name} ---")
    t0 = time.time()
    try:
        for block, blk_idx in [("val", val_idx), ("test", test_idx)]:
            pred = nb_m.rolling_forecast(y_full, X_full_spec, blk_idx,
                                              step_days=7)
            pred = pred.set_index("date")
            pred["predicted"] = pred["predicted"].clip(lower=0)
            pred["actual"] = y_full.loc[pred.index]
            s = score(pred["actual"], pred["predicted"])
            print(f"  {block}: MAPE={s['MAPE']:.2f}  MAE={s['MAE']:.2f}  "
                  f"RMSE={s['RMSE']:.2f}  R2={s['R2']:+.3f}")
            target_dir = out_pred if block == "val" else out_pred_test
            pred.reset_index().to_csv(
                target_dir / f"task2_{name}_nbglm.csv", index=False)
            metric_rows.append({"specialty": name, "model": "NB-GLM",
                                  "block": block, **s, "order": "",
                                  "params": ""})
    except Exception as exc:
        print(f"  NB-GLM FAILED: {exc}")
    print(f"  NB-GLM took {time.time() - t0:.0f}s")

    # ---- (4-6) ML / DL with Random Search on §3.4.3 consensus features -----
    y_train = y_full.loc[train_idx]
    X_cons_train = X_cons.loc[train_idx]
    folds = subsampled_rolling_origin(train_idx, n_folds=N_FOLDS,
                                        horizon_days=7, step_days=7,
                                        min_train_days=365)

    for model_name, eval_fn, runner, space in [
        ("XGBoost", xgb_cv,  rolling_xgb,  XGB_SPACE),
        ("ANN",     ann_cv,  rolling_ann,  ANN_SPACE),
        ("LSTM",    lstm_cv, rolling_lstm, LSTM_SPACE),
    ]:
        print(f"\n--- {model_name} on {name}  (Random Search, min cv_RMSE) ---")
        t0 = time.time()
        try:
            trials = random_search(space, eval_fn, X_cons_train, y_train, folds,
                                     model_name.lower())
            for tr in trials:
                trial_rows.append({"specialty": name, "model": model_name, **tr})
            best = trials[0]
            best_params = {k: v for k, v in best.items()
                            if k not in ("trial", "cv_RMSE", "cv_MAPE", "cv_MAE",
                                            "cv_R2", "time_s")}
            print(f"  HPO {(time.time() - t0)/60:.1f} min  best params: {best_params}")
            print(f"  cv_RMSE={best['cv_RMSE']:.3f}  cv_MAPE={best['cv_MAPE']:.2f}  "
                  f"cv_MAE={best['cv_MAE']:.2f}  cv_R2={best['cv_R2']:+.3f}")

            for block, blk_idx in [("val", val_idx), ("test", test_idx)]:
                tb = time.time()
                pred = runner(X_cons, y_full, blk_idx, best_params)
                pred["actual"] = y_full.loc[pred.index]
                s = score(pred["actual"], pred["predicted"])
                print(f"  {block} ({time.time() - tb:.0f}s): MAPE={s['MAPE']:.2f}  "
                      f"MAE={s['MAE']:.2f}  RMSE={s['RMSE']:.2f}  R2={s['R2']:+.3f}")
                target_dir = out_pred if block == "val" else out_pred_test
                pred.reset_index().to_csv(
                    target_dir / f"task2_{name}_{model_name.lower()}.csv",
                    index=False)
                metric_rows.append({"specialty": name, "model": model_name,
                                      "block": block, **s, "order": "",
                                      "params": str(best_params),
                                      "cv_RMSE": best["cv_RMSE"],
                                      "cv_MAPE": best["cv_MAPE"]})
        except Exception as exc:
            print(f"  {model_name} FAILED: {exc}")

    return metric_rows, trial_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TASK 2 — Per-specialty daily forecast  (5 specialties × 6 models)")
    print("Target: raw daily count.   ML/DL: Random Search (min cv_RMSE).")
    print("=" * 70 + "\n")

    g3, X_cons, train_idx, val_idx, test_idx, _ = assemble_data()

    all_metrics, all_trials = [], []
    for name, col in DAILY_SPECIALTIES.items():
        m_rows, t_rows = run_specialty(name, col, g3, X_cons,
                                              train_idx, val_idx, test_idx)
        all_metrics.extend(m_rows)
        all_trials.extend(t_rows)

    # ---- Save consolidated metrics ----------------------------------------
    out_metrics = ROOT / "artefacts" / "metrics"
    out_metrics.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out_metrics / "task2_standalone_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path.relative_to(ROOT)}")

    trials_df = pd.DataFrame(all_trials)
    trials_path = out_metrics / "task2_standalone_hpo_traces.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Wrote: {trials_path.relative_to(ROOT)}")

    # ---- Pivot summary printed -------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL — Task 2 per-specialty MAPE (val | test)")
    print("=" * 70)
    pivot = metrics_df.pivot_table(
        index=["specialty"], columns=["model", "block"],
        values="MAPE", aggfunc="first"
    )
    print(pivot.round(2).to_string())

    print("\n" + "=" * 70)
    print("FINAL — Task 2 per-specialty MAE  (val | test)")
    print("=" * 70)
    pivot_mae = metrics_df.pivot_table(
        index=["specialty"], columns=["model", "block"],
        values="MAE", aggfunc="first"
    )
    print(pivot_mae.round(3).to_string())

    print("\n" + "=" * 70)
    print("FINAL — Task 2 per-specialty RMSE (val | test)")
    print("=" * 70)
    pivot_rmse = metrics_df.pivot_table(
        index=["specialty"], columns=["model", "block"],
        values="RMSE", aggfunc="first"
    )
    print(pivot_rmse.round(3).to_string())


if __name__ == "__main__":
    main()
