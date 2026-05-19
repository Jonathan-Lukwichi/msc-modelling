"""Re-run XGBoost / ANN / LSTM with the RMSE-best hyperparameters from §18.

After §18's HPO-fairness audit picked the best params by cv_RMSE:
  XGBoost (Grid winner):  n_est=500, depth=3, lr=0.01, sub=1.0
  ANN     (Random winner): 2 hidden × 192 units, dropout 0.2, lr~1.9e-3, batch 64
  LSTM    (Optuna winner): lookback=14, units=128, dropout 0.2, lr~2.3e-3, batch 32

Refit each on the full train block and run rolling weekly refit on val AND test.
Save under artefacts/predictions/{model}_rmse.csv and
artefacts/predictions/test/{model}_rmse.csv. Overwrite best_params JSON files
so downstream hybrids (script 09 / 15) pick up the new params.

Total runtime: ~30 min (LSTM dominates).
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")

# RMSE-best params from §4ter audit
XGB_PARAMS = {"n_estimators": 500, "max_depth": 3,
                "learning_rate": 0.01, "subsample": 1.0}
ANN_PARAMS = {"hidden_layers": 2, "units": 192, "dropout": 0.2,
                "learning_rate": 0.001887, "batch_size": 64, "seed": 42}
LSTM_PARAMS = {"lookback": 14, "units": 128, "dropout": 0.2,
                "learning_rate": 0.002333, "batch_size": 32, "seed": 42}


# ----- Data setup -----
splits = Splits.from_config()
g1 = load_g1()
target = g1["total_daily_arrivals"]
eng = load_engineered()
X_consensus = build_selected_X(eng)
df = pd.concat([target.rename("y"), X_consensus], axis=1, join="inner").dropna()

train_idx = splits.slice(g1, "train").index.intersection(df.index)
val_idx = splits.slice(g1, "val").index.intersection(df.index)
test_idx = splits.slice(g1, "test").index.intersection(df.index)
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}\n")


# ----- XGBoost (rolling weekly refit) -----
def rolling_xgb(target_block_idx):
    from xgboost import XGBRegressor
    rows = []
    origin = df.index[df.index < target_block_idx[0]][-1]
    val_remaining = list(target_block_idx)
    while val_remaining:
        h = min(7, len(val_remaining))
        future = val_remaining[:h]
        tr_dates = df.index[df.index <= origin]
        m = XGBRegressor(**XGB_PARAMS, objective="reg:squarederror",
                          random_state=42, verbosity=0, n_jobs=-1)
        m.fit(df.loc[tr_dates].drop(columns=["y"]).values,
              df.loc[tr_dates, "y"].values)
        yhat = m.predict(df.loc[future].drop(columns=["y"]).values)
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future[-1]
        val_remaining = val_remaining[h:]
    return pd.DataFrame(rows).set_index("date")


# ----- ANN (rolling weekly refit) -----
def rolling_ann(target_block_idx):
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    rows = []
    origin = df.index[df.index < target_block_idx[0]][-1]
    val_remaining = list(target_block_idx)
    while val_remaining:
        h = min(7, len(val_remaining))
        future = val_remaining[:h]
        tr_dates = df.index[df.index <= origin]
        X_tr = df.loc[tr_dates].drop(columns=["y"])
        y_tr = df.loc[tr_dates, "y"]
        mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr - mean) / std).astype(np.float32).values
        Xfu = ((df.loc[future].drop(columns=["y"]) - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std(ddof=0) or 1.0)
        ytr = ((y_tr - y_mean) / y_std).astype(np.float32).values

        _seed_everything(42)
        n_es = max(28, len(Xtr) // 6)
        n_hidden = [ANN_PARAMS["units"]] * ANN_PARAMS["hidden_layers"]
        model = _MLP(n_in=Xtr.shape[1], n_hidden=n_hidden,
                      dropout=ANN_PARAMS["dropout"])
        Xtr_t = torch.from_numpy(Xtr[:-n_es]); ytr_t = torch.from_numpy(ytr[:-n_es])
        Xes_t = torch.from_numpy(Xtr[-n_es:]); yes_t = torch.from_numpy(ytr[-n_es:])
        model, _, _ = _train_one(Xtr_t, ytr_t, Xes_t, yes_t, ANN_PARAMS,
                                  max_epochs=100)
        model.eval()
        Xfu_t = torch.from_numpy(Xfu)
        with torch.no_grad():
            yhat_n = model(Xfu_t).numpy()
        yhat = yhat_n * y_std + y_mean
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        val_remaining = val_remaining[h:]
        origin = future[-1]
    return pd.DataFrame(rows).set_index("date")


# ----- LSTM (rolling weekly refit) -----
def rolling_lstm(target_block_idx):
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    lookback = LSTM_PARAMS["lookback"]
    rows = []
    origin = df.index[df.index < target_block_idx[0]][-1]
    val_remaining = list(target_block_idx)
    while val_remaining:
        h = min(7, len(val_remaining))
        future = val_remaining[:h]
        tr_dates = df.index[df.index <= origin]
        X_tr = df.loc[tr_dates].drop(columns=["y"])
        y_tr = df.loc[tr_dates, "y"]
        mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_tr - mean) / std).astype(np.float32).values
        Xfu = ((df.loc[future].drop(columns=["y"]) - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std(ddof=0) or 1.0)
        ytr = ((y_tr - y_mean) / y_std).astype(np.float32).values

        _seed_everything(42)
        Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
        n_es = max(28, lookback + 7)
        model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                                Xtr_seq[-n_es:], ytr_seq[-n_es:],
                                LSTM_PARAMS, max_epochs=40)
        model.eval()
        full = np.vstack([Xtr, Xfu])
        preds = []
        with torch.no_grad():
            for i in range(len(Xfu)):
                pos = len(Xtr) + i
                window = full[pos - lookback : pos]
                preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
        yhat = np.array(preds) * y_std + y_mean
        for d, yp in zip(future, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        val_remaining = val_remaining[h:]
        origin = future[-1]
    return pd.DataFrame(rows).set_index("date")


# ----- Run each model on val and test -----
def run_model(name, runner):
    print(f"\n=== {name.upper()} (RMSE-best params) ===")
    t0 = time.time()
    val_pred = runner(val_idx)
    val_pred["actual"] = target.loc[val_pred.index]
    val_pred["block"] = "val"
    val_s = score(val_pred["actual"], val_pred["predicted"])
    print(f"  Val:   MAPE={val_s['MAPE']:6.3f}  MAE={val_s['MAE']:5.2f}  "
          f"RMSE={val_s['RMSE']:5.2f}  R2={val_s['R2']:+5.3f}  "
          f"({time.time()-t0:.0f}s)")

    t0 = time.time()
    test_pred = runner(test_idx)
    test_pred["actual"] = target.loc[test_pred.index]
    test_pred["block"] = "test"
    test_s = score(test_pred["actual"], test_pred["predicted"])
    print(f"  Test:  MAPE={test_s['MAPE']:6.3f}  MAE={test_s['MAE']:5.2f}  "
          f"RMSE={test_s['RMSE']:5.2f}  R2={test_s['R2']:+5.3f}  "
          f"({time.time()-t0:.0f}s)")

    val_pred.reset_index().to_csv(
        ROOT / "artefacts" / "predictions" / f"{name}_rmse.csv", index=False)
    test_pred.reset_index().to_csv(
        ROOT / "artefacts" / "predictions" / "test" / f"{name}_rmse.csv", index=False)
    pd.DataFrame([{"block": "val", **val_s},
                   {"block": "test", **test_s}]).to_csv(
        ROOT / "artefacts" / "metrics" / f"{name}_rmse_metrics.csv", index=False)
    return val_s, test_s


# Save best_params files for downstream hybrid script
def save_params():
    (ROOT / "artefacts" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "artefacts" / "models" / "xgboost_rmse_best_params.json").write_text(
        json.dumps(XGB_PARAMS, indent=2))
    (ROOT / "artefacts" / "models" / "ann_rmse_best_params.json").write_text(
        json.dumps(ANN_PARAMS, indent=2))
    (ROOT / "artefacts" / "models" / "lstm_rmse_best_params.json").write_text(
        json.dumps(LSTM_PARAMS, indent=2))


if __name__ == "__main__":
    save_params()
    run_model("xgboost", rolling_xgb)
    run_model("ann", rolling_ann)
    run_model("lstm", rolling_lstm)
    print("\nDone.")
