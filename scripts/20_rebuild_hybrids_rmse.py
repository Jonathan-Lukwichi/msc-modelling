"""Rebuild the 6 hybrids using the RMSE-optimised XGBoost / ANN / LSTM params.

Reuses the SARIMAX base (unchanged — it uses §5.2.5 raw 10 exogenous, AIC-picked
order). Replaces the XGBoost / ANN / LSTM REFINER hyperparameters with the
RMSE-best values from §18 audit:

  XGBoost refiner: n_est=500, depth=3, lr=0.01, sub=1.0
  ANN refiner:     2 hidden × 192 units, dropout 0.2, lr~1.9e-3, batch 64
  LSTM refiner:    lookback=14, units=128, dropout 0.2, lr~2.3e-3, batch 32

Six hybrids rebuilt:
  Residual: SARIMAX+XGB, SARIMAX+LSTM, LSTM+XGB
  STL:      STL+XGB, STL+ANN, STL+LSTM

Each saved as artefacts/predictions/hybrid_{name}_rmse.csv with val + test.
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")

# RMSE-best params (from §18 audit)
XGB_PARAMS = {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01, "subsample": 1.0}
ANN_PARAMS = {"hidden_layers": 2, "units": 192, "dropout": 0.2,
                "learning_rate": 0.001887, "batch_size": 64, "seed": 42}
LSTM_PARAMS = {"lookback": 14, "units": 128, "dropout": 0.2,
                "learning_rate": 0.002333, "batch_size": 32, "seed": 42}


splits = Splits.from_config()
g1 = load_g1()
target = g1["total_daily_arrivals"]
eng = load_engineered()
X_cons = build_selected_X(eng)
df = pd.concat([target.rename("y"), X_cons], axis=1, join="inner").dropna()
train_idx = splits.slice(g1, "train").index.intersection(df.index)
val_idx = splits.slice(g1, "val").index.intersection(df.index)
test_idx = splits.slice(g1, "test").index.intersection(df.index)
X_train = df.loc[train_idx].drop(columns=["y"])
y_train = df.loc[train_idx, "y"]


# ---------------------------------------------------------------------------
# Refiner builders (use RMSE-best params)
# ---------------------------------------------------------------------------

def xgb_refiner(X, resid):
    from xgboost import XGBRegressor
    m = XGBRegressor(**XGB_PARAMS, objective="reg:squarederror",
                      random_state=42, verbosity=0, n_jobs=-1)
    m.fit(X.values, resid.values)
    return m


def ann_refiner(X, resid):
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    _seed_everything(42)
    mean = X.mean(); std = X.std(ddof=0).replace(0, 1.0)
    Xs = ((X - mean) / std).astype(np.float32).values
    r_mean, r_std = float(resid.mean()), float(resid.std(ddof=0) or 1.0)
    rs = ((resid - r_mean) / r_std).astype(np.float32).values
    n_es = max(28, len(Xs) // 6)
    model = _MLP(n_in=Xs.shape[1],
                  n_hidden=[ANN_PARAMS["units"]] * ANN_PARAMS["hidden_layers"],
                  dropout=ANN_PARAMS["dropout"])
    Xtr_t = torch.from_numpy(Xs[:-n_es]); rtr_t = torch.from_numpy(rs[:-n_es])
    Xes_t = torch.from_numpy(Xs[-n_es:]); res_t = torch.from_numpy(rs[-n_es:])
    model, _, _ = _train_one(Xtr_t, rtr_t, Xes_t, res_t, ANN_PARAMS, max_epochs=80)
    return model, mean, std, r_mean, r_std


def lstm_refiner(X, resid):
    import torch
    from src.forecasting.models.lstm import _build_sequences, _train_one, _seed_everything
    _seed_everything(42)
    mean = X.mean(); std = X.std(ddof=0).replace(0, 1.0)
    Xs = ((X - mean) / std).astype(np.float32).values
    r_mean, r_std = float(resid.mean()), float(resid.std(ddof=0) or 1.0)
    rs = ((resid - r_mean) / r_std).astype(np.float32).values
    lookback = LSTM_PARAMS["lookback"]
    Xs_seq, rs_seq = _build_sequences(Xs, rs, lookback)
    n_es = 28
    model, _ = _train_one(Xs_seq[:-n_es], rs_seq[:-n_es],
                            Xs_seq[-n_es:], rs_seq[-n_es:],
                            LSTM_PARAMS, max_epochs=40)
    return model, mean, std, r_mean, r_std, lookback


def predict_xgb(model, X_block):
    return pd.Series(model.predict(X_block.values), index=X_block.index)


def predict_ann(fit_artifacts, X_block):
    import torch
    model, mean, std, r_mean, r_std = fit_artifacts
    Xs = ((X_block - mean) / std).astype(np.float32).values
    model.eval()
    with torch.no_grad():
        rn = model(torch.from_numpy(Xs)).numpy()
    return pd.Series(rn * r_std + r_mean, index=X_block.index)


def predict_lstm(fit_artifacts, X_full, target_idx):
    import torch
    model, mean, std, r_mean, r_std, lookback = fit_artifacts
    Xs = ((X_full - mean) / std).astype(np.float32).values
    rows = []
    train_end_pos = X_full.index.get_loc(target_idx[0]) - 1
    model.eval()
    for i, date in enumerate(target_idx):
        pos = train_end_pos + 1 + i
        if pos < lookback:
            continue
        window = Xs[pos - lookback : pos]
        with torch.no_grad():
            yn = float(model(torch.from_numpy(window[None, :, :])).item())
        rows.append({"date": date, "predicted": yn * r_std + r_mean})
    return pd.DataFrame(rows).set_index("date")["predicted"]


# ---------------------------------------------------------------------------
# Base-model train-residuals helpers
# ---------------------------------------------------------------------------

def sarimax_train_residuals():
    from pmdarima import ARIMA as PmARIMA
    cfg_path = ROOT / "artefacts" / "models" / "sarimax_order.txt"
    cfg = dict(line.partition("=")[::2]
                 for line in cfg_path.read_text().strip().splitlines())
    order = eval(cfg["order"].strip())
    seasonal_order = eval(cfg["seasonal_order"].strip())
    X_train_sarimax, _ = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    model = PmARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
    model.fit(target.loc[train_idx].values, X=X_train_sarimax.values)
    fitted = model.predict_in_sample(X=X_train_sarimax.values)
    resid = pd.Series(target.loc[train_idx].values - np.asarray(fitted),
                       index=train_idx, name="residual").iloc[30:]
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
    sigma = resid.std()
    if sigma > 0:
        resid = resid[resid.abs() <= 5 * sigma]
    return resid


def lstm_train_in_sample_preds():
    """Refit LSTM (RMSE-best params) on full train; return in-sample predictions."""
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    _seed_everything(42)
    X_tr = X_train
    mean = X_tr.mean(); std = X_tr.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_tr - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0) or 1.0)
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values
    lookback = LSTM_PARAMS["lookback"]
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
    n_es = max(28, lookback + 7)
    model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                            Xtr_seq[-n_es:], ytr_seq[-n_es:],
                            LSTM_PARAMS, max_epochs=40)
    model.eval()
    with torch.no_grad():
        yhat_n = model(torch.from_numpy(Xtr_seq)).numpy()
    yhat = yhat_n * y_std + y_mean
    idx = X_tr.index[lookback:]
    return pd.Series(yhat, index=idx)


# ---------------------------------------------------------------------------
# Load base-model predictions for val and test
# ---------------------------------------------------------------------------

def load_val_test_preds(name):
    """Return (val_pred_series, test_pred_series)."""
    val_path = ROOT / "artefacts" / "predictions" / f"{name}.csv"
    test_path = ROOT / "artefacts" / "predictions" / "test" / f"{name}.csv"
    val = pd.read_csv(val_path, parse_dates=["date"]).set_index("date")["predicted"]
    test = pd.read_csv(test_path, parse_dates=["date"]).set_index("date")["predicted"]
    return val, test


def load_lstm_rmse_preds():
    val = pd.read_csv(ROOT / "artefacts" / "predictions" / "lstm_rmse.csv",
                       parse_dates=["date"]).set_index("date")["predicted"]
    test = pd.read_csv(ROOT / "artefacts" / "predictions" / "test" / "lstm_rmse.csv",
                       parse_dates=["date"]).set_index("date")["predicted"]
    return val, test


# ---------------------------------------------------------------------------
# Hybrid builders
# ---------------------------------------------------------------------------

def write_hybrid(name, val_pred, test_pred):
    val_pred = val_pred.copy()
    val_pred["actual"] = target.loc[val_pred.index]
    val_pred["block"] = "val"
    test_pred = test_pred.copy()
    test_pred["actual"] = target.loc[test_pred.index]
    test_pred["block"] = "test"
    val_pred.reset_index().to_csv(
        ROOT / "artefacts" / "predictions" / f"hybrid_{name}_rmse.csv", index=False)
    test_pred.reset_index().to_csv(
        ROOT / "artefacts" / "predictions" / "test" / f"hybrid_{name}_rmse.csv", index=False)
    val_s = score(val_pred["actual"], val_pred["predicted"])
    test_s = score(test_pred["actual"], test_pred["predicted"])
    pd.DataFrame([
        {"block": "val", **val_s},
        {"block": "test", **test_s},
    ]).to_csv(ROOT / "artefacts" / "metrics" / f"hybrid_{name}_rmse_metrics.csv",
                index=False)
    print(f"  {name:<24s} | "
          f"val RMSE={val_s['RMSE']:5.2f}  MAPE={val_s['MAPE']:5.2f}%  | "
          f"test RMSE={test_s['RMSE']:5.2f}  MAPE={test_s['MAPE']:5.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()
    print("Computing SARIMAX in-sample residuals (~30s)...")
    sarimax_resid = sarimax_train_residuals()

    sarimax_val, sarimax_test = load_val_test_preds("sarimax")

    # RESIDUAL: SARIMAX + XGB
    print("\n[1/6] SARIMAX + XGB (RMSE-best refiner)...")
    t0 = time.time()
    common = X_train.index.intersection(sarimax_resid.index)
    xgb = xgb_refiner(X_train.loc[common], sarimax_resid.loc[common])
    val_pred = pd.DataFrame({"predicted": (sarimax_val.reindex(val_idx).values
                                              + predict_xgb(xgb, X_train.reindex(val_idx)).values)},
                              index=val_idx)
    val_pred.index.name = "date"
    test_pred = pd.DataFrame({"predicted": (sarimax_test.reindex(test_idx).values
                                               + predict_xgb(xgb, df.loc[test_idx].drop(columns=["y"])).values)},
                              index=test_idx)
    test_pred.index.name = "date"
    write_hybrid("sarimax_xgb", val_pred, test_pred)
    print(f"  ({time.time() - t0:.0f}s)")

    # RESIDUAL: SARIMAX + LSTM
    print("\n[2/6] SARIMAX + LSTM (RMSE-best refiner)...")
    t0 = time.time()
    fa = lstm_refiner(X_train.loc[common], sarimax_resid.loc[common])
    refiner_val = predict_lstm(fa, df.drop(columns=["y"]), val_idx)
    refiner_test = predict_lstm(fa, df.drop(columns=["y"]), test_idx)
    val_pred = pd.DataFrame({"predicted": (sarimax_val.reindex(val_idx).values
                                              + refiner_val.reindex(val_idx).fillna(0).values)},
                              index=val_idx)
    val_pred.index.name = "date"
    test_pred = pd.DataFrame({"predicted": (sarimax_test.reindex(test_idx).values
                                               + refiner_test.reindex(test_idx).fillna(0).values)},
                              index=test_idx)
    test_pred.index.name = "date"
    write_hybrid("sarimax_lstm", val_pred, test_pred)
    print(f"  ({time.time() - t0:.0f}s)")

    # RESIDUAL: LSTM + XGB
    print("\n[3/6] LSTM + XGB (RMSE-best LSTM base + XGB refiner)...")
    t0 = time.time()
    lstm_val, lstm_test = load_lstm_rmse_preds()
    in_sample = lstm_train_in_sample_preds()
    train_resid_lstm = y_train.loc[in_sample.index] - in_sample
    sigma = train_resid_lstm.std()
    train_resid_lstm = train_resid_lstm[train_resid_lstm.abs() <= 5 * sigma]
    common = X_train.index.intersection(train_resid_lstm.index)
    xgb2 = xgb_refiner(X_train.loc[common], train_resid_lstm.loc[common])
    val_pred = pd.DataFrame({"predicted": (lstm_val.reindex(val_idx).values
                                              + predict_xgb(xgb2, X_train.reindex(val_idx)).values)},
                              index=val_idx)
    val_pred.index.name = "date"
    test_pred = pd.DataFrame({"predicted": (lstm_test.reindex(test_idx).values
                                               + predict_xgb(xgb2, df.loc[test_idx].drop(columns=["y"])).values)},
                              index=test_idx)
    test_pred.index.name = "date"
    write_hybrid("lstm_xgb", val_pred, test_pred)
    print(f"  ({time.time() - t0:.0f}s)")

    # STL hybrids — need decomposition + forecasts
    from src.forecasting.hybrids import stl_hybrid as S
    decomp = S.decompose_train(y_train, period=7)

    def stl_forecast(idx):
        trend = S.forecast_trend(decomp.trend, idx, method="damped_linear")
        seas = S.forecast_seasonal(decomp.seasonal, idx, period=7)
        return trend, seas

    for refiner_kind, name, builder, predictor in [
        ("xgb", "stl_xgb",
         lambda Xc, rc: xgb_refiner(Xc, rc),
         lambda mdl, Xb: predict_xgb(mdl, Xb)),
        ("ann", "stl_ann",
         lambda Xc, rc: ann_refiner(Xc, rc),
         lambda mdl, Xb: predict_ann(mdl, Xb)),
        ("lstm", "stl_lstm",
         lambda Xc, rc: lstm_refiner(Xc, rc),
         lambda mdl, Xb: predict_lstm(mdl, df.drop(columns=["y"]),
                                        Xb.index if hasattr(Xb, "index") else Xb)),
    ]:
        print(f"\n[{4 if name=='stl_xgb' else 5 if name=='stl_ann' else 6}/6] "
              f"STL + {refiner_kind.upper()} (RMSE-best refiner)...")
        t0 = time.time()
        common = X_train.index.intersection(decomp.residual.index)
        ref = builder(X_train.loc[common], decomp.residual.loc[common])
        trend_v, seas_v = stl_forecast(val_idx)
        trend_t, seas_t = stl_forecast(test_idx)
        if refiner_kind == "lstm":
            refiner_v = predictor(ref, X_train.reindex(val_idx)).reindex(val_idx).fillna(0)
            refiner_t = predictor(ref, X_train.reindex(test_idx)).reindex(test_idx).fillna(0)
        else:
            refiner_v = predictor(ref, X_train.reindex(val_idx)).reindex(val_idx).fillna(0)
            refiner_t = predictor(ref, df.loc[test_idx].drop(columns=["y"])).reindex(test_idx).fillna(0)
        val_pred = pd.DataFrame({"predicted": (trend_v.values + seas_v.values
                                                  + refiner_v.values)}, index=val_idx)
        val_pred.index.name = "date"
        test_pred = pd.DataFrame({"predicted": (trend_t.values + seas_t.values
                                                   + refiner_t.values)}, index=test_idx)
        test_pred.index.name = "date"
        write_hybrid(name, val_pred, test_pred)
        print(f"  ({time.time() - t0:.0f}s)")

    print(f"\nTotal: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
