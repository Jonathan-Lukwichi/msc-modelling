"""Task 2 extension — ML + hybrid models for the 5 daily specialties.

Adds the following to each of Medicine / Orthopaedics / Surgery / Paediatrics /
Gynaecology (target = share-of-header per §4.4.4):

  Standalone ML / DL:
    - ANN  (1-layer × 128 units, dropout 0.2, lr 1e-3, batch 32)
    - LSTM (lookback 21, 128 units, dropout 0.2, lr 1e-3, batch 32)

  Residual hybrids (Zhang Alg 6 per §3.5.4 / §12.1):
    - SARIMAX + XGBoost
    - SARIMAX + LSTM
    - LSTM + XGBoost

  STL hybrids (Alg 7 per §3.5.4 / §12.1):
    - STL + XGBoost
    - STL + ANN
    - STL + LSTM

Hyperparameters are light defaults — no per-specialty HPO. The chapter's
methodological mandate (Ch5 §5.3.3) is per-specialty *coefficients*, not
per-specialty hyperparameter search. The §3.5.9 HPO ranges are honoured at the
Task-1 level; Task-2 models inherit a sensible default-architecture choice.

Outputs:
  - artefacts/predictions/task2_{specialty}_{model}.csv  (one per cell)
  - artefacts/metrics/task2_ml_hybrid_metrics.csv         (rollup)
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g3, Splits
from src.forecasting.features import StandardScaler
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DAILY_SPECIALTIES = {
    "Medicine":     "spec_medicine",
    "Orthopaedics": "spec_orthopaedics",
    "Surgery":      "spec_surgery",
    "Paediatrics":  "spec_paediatrics",
    "Gynaecology":  "spec_gynae",
}


# Default architectures (no HPO per specialty)
ANN_PARAMS = {"hidden_layers": 1, "units": 128, "dropout": 0.2,
               "learning_rate": 0.001, "batch_size": 32, "seed": 42}
LSTM_PARAMS = {"lookback": 21, "units": 128, "dropout": 0.2,
                "learning_rate": 0.001, "batch_size": 32, "seed": 42}
XGB_PARAMS = {"n_estimators": 200, "max_depth": 4,
               "learning_rate": 0.05, "subsample": 0.85}


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_combined_features(
    g3: pd.DataFrame, specialty: str,
    train_idx: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """Combine §3.4.3 consensus features + per-specialty weather + Surgery
    sign-reversal interactions.

    Returns (X_full, X_train, scaler_or_None).
    """
    # Load engineered + consensus 23 features
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    # G3 has share-relevant columns; engineered matrix has the same dates as G1.
    # We use only the engineered features (these include calendar + lag of total).

    # Per-specialty weather block (raw values; scale on train)
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "features_task2.yaml").read_text())
    spec_cfg = cfg["daily_specialties"][specialty]
    weather_cols = spec_cfg.get("weather", []) or []
    interactions = spec_cfg.get("interactions", []) or []

    extra = pd.DataFrame(index=g3.index)
    if weather_cols:
        for c in weather_cols:
            if c in g3.columns:
                extra[c] = g3[c]
    for c in interactions:
        if c in g3.columns:
            extra[f"{specialty.lower()}_{c}"] = g3[c].astype(int)

    X_full = X_consensus.join(extra, how="inner")

    # Scale only the new weather cols (consensus features keep their original scale
    # since XGBoost and tree-based don't need scaling, and ANN/LSTM rescale
    # internally per-fit).
    scaler = None
    if weather_cols:
        scaler = StandardScaler.fit(X_full.loc[train_idx.intersection(X_full.index)],
                                      weather_cols)
        X_full = scaler.transform(X_full)

    X_train = X_full.loc[train_idx.intersection(X_full.index)]
    return X_full, X_train, scaler


# ---------------------------------------------------------------------------
# Single-fit predict helpers (per cell)
# ---------------------------------------------------------------------------

def fit_predict_ann(X_train, y_train, X_val):
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    _seed_everything(42)
    mean = X_train.mean(); std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    Xva = ((X_val - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0) or 1.0)
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values
    n_es = max(28, len(Xtr) // 6)
    n_hidden = [ANN_PARAMS["units"]] * ANN_PARAMS["hidden_layers"]
    model = _MLP(n_in=Xtr.shape[1], n_hidden=n_hidden, dropout=ANN_PARAMS["dropout"])
    Xtr_t = torch.from_numpy(Xtr[:-n_es])
    ytr_t = torch.from_numpy(ytr[:-n_es])
    Xes_t = torch.from_numpy(Xtr[-n_es:])
    yes_t = torch.from_numpy(ytr[-n_es:])
    model, _, _ = _train_one(Xtr_t, ytr_t, Xes_t, yes_t,
                              ANN_PARAMS, max_epochs=100)
    model.eval()
    with torch.no_grad():
        yhat_norm = model(torch.from_numpy(Xva)).numpy()
    return yhat_norm * y_std + y_mean


def fit_predict_lstm(X_train, y_train, X_val):
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything, _LSTMNet,
    )
    _seed_everything(42)
    mean = X_train.mean(); std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    Xva = ((X_val - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0) or 1.0)
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values
    lookback = LSTM_PARAMS["lookback"]
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
    n_es = max(28, lookback + 7)
    model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                            Xtr_seq[-n_es:], ytr_seq[-n_es:],
                            LSTM_PARAMS, max_epochs=40)
    model.eval()
    full = np.vstack([Xtr, Xva])
    preds = []
    with torch.no_grad():
        for i in range(len(Xva)):
            pos = len(Xtr) + i
            window = full[pos - lookback : pos]
            preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
    return np.array(preds) * y_std + y_mean


def fit_predict_xgb(X_train, y_train, X_val):
    from xgboost import XGBRegressor
    m = XGBRegressor(**XGB_PARAMS, objective="reg:squarederror",
                      random_state=42, verbosity=0, n_jobs=-1)
    m.fit(X_train.values, y_train.values)
    return m.predict(X_val.values)


# ---------------------------------------------------------------------------
# SARIMAX (refit per specialty, light auto_arima)
# ---------------------------------------------------------------------------

def fit_sarimax_share(y_train, X_train, X_val, X_full, val_idx):
    """Fit SARIMAX once on train, return (val_pred, train_residuals, order, model)."""
    from pmdarima import auto_arima
    np.random.seed(42)
    base = auto_arima(
        y_train.values, X=X_train.values,
        start_p=0, start_q=0, max_p=2, max_q=2,
        max_P=2, max_Q=2, d=1, D=1, seasonal=True, m=7,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic",
        random_state=42,
    )
    # In-sample residuals (trim 30-day warmup per residual.py convention)
    fitted = base.predict_in_sample(X=X_train.values)
    resid = pd.Series(y_train.values - np.asarray(fitted), index=y_train.index,
                       name="residual").iloc[30:]
    # Trim extreme residuals
    sigma = resid.std()
    resid = resid[resid.abs() <= 5 * sigma]
    # Single-shot val forecast (no rolling — refiner correction smooths things)
    yhat = base.predict(n_periods=len(val_idx), X=X_val.values)
    val_pred = pd.Series(np.asarray(yhat), index=val_idx)
    return val_pred, resid, base.order, base.seasonal_order, base


# ---------------------------------------------------------------------------
# Residual hybrids (share target)
# ---------------------------------------------------------------------------

def residual_sarimax_xgb(y_train, X_train, X_val, X_full,
                          val_idx, sarimax_resid, sarimax_val_pred):
    from xgboost import XGBRegressor
    common = X_train.index.intersection(sarimax_resid.index)
    m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                      subsample=0.85, objective="reg:squarederror",
                      random_state=42, verbosity=0, n_jobs=-1)
    m.fit(X_train.loc[common].values, sarimax_resid.loc[common].values)
    refiner_val = m.predict(X_val.values)
    return sarimax_val_pred.values + refiner_val


def residual_sarimax_lstm(y_train, X_train, X_val, X_full,
                            val_idx, sarimax_resid, sarimax_val_pred):
    """Light LSTM refiner on SARIMAX residuals (lookback 14, 64 units)."""
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    _seed_everything(42)
    common = X_train.index.intersection(sarimax_resid.index)
    X_t = X_train.loc[common]
    r = sarimax_resid.loc[common]
    mean = X_t.mean(); std = X_t.std(ddof=0).replace(0, 1.0)
    Xs = ((X_t - mean) / std).astype(np.float32).values
    r_mean = float(r.mean()); r_std = float(r.std(ddof=0) or 1.0)
    rs = ((r - r_mean) / r_std).astype(np.float32).values
    lookback = 14
    Xs_seq, rs_seq = _build_sequences(Xs, rs, lookback)
    n_es = 28
    params = {"units": 64, "dropout": 0.2, "learning_rate": 0.001,
               "batch_size": 32, "seed": 42}
    model, _ = _train_one(Xs_seq[:-n_es], rs_seq[:-n_es],
                            Xs_seq[-n_es:], rs_seq[-n_es:],
                            params, max_epochs=30)
    # Predict refiner contribution on val (sliding window over X_train tail + X_val)
    X_val_scaled = ((X_val - mean) / std).astype(np.float32).values
    full = np.vstack([Xs, X_val_scaled])
    refiner_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(len(X_val)):
            pos = len(Xs) + i
            if pos < lookback:
                refiner_preds.append(0.0)
                continue
            window = full[pos - lookback : pos]
            yhat_norm = float(model(torch.from_numpy(window[None, :, :])).item())
            refiner_preds.append(yhat_norm * r_std + r_mean)
    return sarimax_val_pred.values + np.array(refiner_preds)


def residual_lstm_xgb(y_train, X_train, X_val, X_full, val_idx):
    """LSTM base; XGBoost refiner on LSTM in-sample residuals."""
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    from xgboost import XGBRegressor
    _seed_everything(42)
    mean = X_train.mean(); std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    Xva = ((X_val - mean) / std).astype(np.float32).values
    y_mean = float(y_train.mean()); y_std = float(y_train.std(ddof=0) or 1.0)
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values
    lookback = LSTM_PARAMS["lookback"]
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
    n_es = max(28, lookback + 7)
    model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                            Xtr_seq[-n_es:], ytr_seq[-n_es:],
                            LSTM_PARAMS, max_epochs=30)
    model.eval()
    # In-sample LSTM preds
    with torch.no_grad():
        yhat_norm = model(torch.from_numpy(Xtr_seq)).numpy()
    in_sample = pd.Series(yhat_norm * y_std + y_mean,
                            index=X_train.index[lookback:])
    train_resid = y_train.loc[in_sample.index] - in_sample
    sigma = train_resid.std()
    train_resid = train_resid[train_resid.abs() <= 5 * sigma]

    # XGBoost refiner on residuals
    refiner = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.85, objective="reg:squarederror",
                            random_state=42, verbosity=0, n_jobs=-1)
    common = X_train.index.intersection(train_resid.index)
    refiner.fit(X_train.loc[common].values, train_resid.loc[common].values)

    # LSTM forecast on val
    full = np.vstack([Xtr, Xva])
    lstm_val = []
    with torch.no_grad():
        for i in range(len(Xva)):
            pos = len(Xtr) + i
            if pos < lookback:
                lstm_val.append(float(y_train.iloc[-1]))
                continue
            window = full[pos - lookback : pos]
            lstm_val.append(float(model(torch.from_numpy(window[None, :, :])).item())
                            * y_std + y_mean)
    refiner_val = refiner.predict(X_val.values)
    return np.array(lstm_val) + refiner_val


# ---------------------------------------------------------------------------
# STL hybrids (share target)
# ---------------------------------------------------------------------------

def stl_hybrid(y_train, X_train, X_val, X_full, val_idx, refiner_kind: str):
    from src.forecasting.hybrids import stl_hybrid as S
    decomp = S.decompose_train(y_train, period=7)
    trend_fc = S.forecast_trend(decomp.trend, val_idx, method="damped_linear")
    seasonal_fc = S.forecast_seasonal(decomp.seasonal, val_idx, period=7)
    if refiner_kind == "xgb":
        refiner = S.fit_xgb_refiner_on_residual(X_train, decomp.residual)
    elif refiner_kind == "ann":
        refiner = S.fit_ann_refiner_on_residual(X_train, decomp.residual)
    elif refiner_kind == "lstm":
        refiner = S.fit_lstm_refiner_on_residual(X_train, decomp.residual)
    else:
        raise ValueError(refiner_kind)
    refiner_val = S.refiner_predict_val(refiner_kind, refiner, X_full, val_idx)
    refiner_val_aligned = refiner_val.reindex(val_idx).fillna(0)
    combined = trend_fc.values + seasonal_fc.values + refiner_val_aligned.values
    return combined


# ---------------------------------------------------------------------------
# Per-specialty driver
# ---------------------------------------------------------------------------

def run_specialty(g3: pd.DataFrame, splits: Splits, name: str, col: str) -> list[dict]:
    print(f"\n=== Daily specialty: {name} ===")
    t0 = time.time()

    target_share = (g3[col] / g3["total_daily_arrivals"]).rename("share")
    train_idx_full = splits.slice(g3, "train").index
    val_idx_full = splits.slice(g3, "val").index

    # Restrict to where share is finite
    share_train = target_share.loc[train_idx_full].dropna()
    share_val = target_share.loc[val_idx_full].dropna()
    train_idx = share_train.index
    val_idx = share_val.index

    # Combined features
    X_full, X_train, _ = build_combined_features(g3, name, train_idx)
    # Align
    train_idx = train_idx.intersection(X_full.index)
    val_idx = val_idx.intersection(X_full.index)
    X_train = X_full.loc[train_idx]
    X_val = X_full.loc[val_idx]
    y_train = share_train.loc[train_idx]
    y_val = share_val.loc[val_idx]
    print(f"  Train: {len(y_train)}  Val: {len(y_val)}  X cols: {X_train.shape[1]}")

    out_pred = ROOT / "artefacts" / "predictions"
    out_pred.mkdir(parents=True, exist_ok=True)
    rows = []

    def _record(model_name: str, yhat: np.ndarray):
        pred = pd.DataFrame({"predicted": yhat, "actual": y_val.values,
                              "block": "val"}, index=val_idx)
        pred.index.name = "date"
        pred.to_csv(out_pred / f"task2_{name}_{model_name}.csv")
        m = score(pred["actual"], pred["predicted"])
        rows.append({"specialty": name, "resolution": "daily",
                      "model": model_name, "block": "val", **m,
                      "n_train": len(y_train), "n_val": len(y_val)})
        print(f"  {model_name:>20s}: MAPE={m['MAPE']:6.3f} "
              f"MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:+.3f}")
        return pred["predicted"]

    # Standalone ANN
    try:
        t1 = time.time()
        yhat = fit_predict_ann(X_train, y_train, X_val)
        ann_val = _record("ann", yhat)
        print(f"    (ANN took {time.time() - t1:.1f}s)")
    except Exception as exc:
        print(f"  ANN FAILED: {exc}")
        ann_val = None

    # Standalone LSTM
    try:
        t1 = time.time()
        yhat = fit_predict_lstm(X_train, y_train, X_val)
        lstm_val = _record("lstm", yhat)
        print(f"    (LSTM took {time.time() - t1:.1f}s)")
    except Exception as exc:
        print(f"  LSTM FAILED: {exc}")
        lstm_val = None

    # SARIMAX (need fit for hybrids)
    print("  Fitting SARIMAX for hybrid base...")
    t1 = time.time()
    sarimax_val_pred, sarimax_resid, order_s, seasonal_s, base_model = fit_sarimax_share(
        y_train, X_train, X_val, X_full, val_idx,
    )
    print(f"    SARIMAX{order_s}x{seasonal_s} ({time.time() - t1:.1f}s)")

    # Residual hybrid: SARIMAX + XGB
    try:
        t1 = time.time()
        yhat = residual_sarimax_xgb(y_train, X_train, X_val, X_full, val_idx,
                                      sarimax_resid, sarimax_val_pred)
        _record("hybrid_sarimax_xgb", yhat)
        print(f"    (SARIMAX+XGB took {time.time() - t1:.1f}s)")
    except Exception as exc:
        print(f"  SARIMAX+XGB FAILED: {exc}")

    # Residual hybrid: SARIMAX + LSTM
    try:
        t1 = time.time()
        yhat = residual_sarimax_lstm(y_train, X_train, X_val, X_full, val_idx,
                                       sarimax_resid, sarimax_val_pred)
        _record("hybrid_sarimax_lstm", yhat)
        print(f"    (SARIMAX+LSTM took {time.time() - t1:.1f}s)")
    except Exception as exc:
        print(f"  SARIMAX+LSTM FAILED: {exc}")

    # Residual hybrid: LSTM + XGB
    try:
        t1 = time.time()
        yhat = residual_lstm_xgb(y_train, X_train, X_val, X_full, val_idx)
        _record("hybrid_lstm_xgb", yhat)
        print(f"    (LSTM+XGB took {time.time() - t1:.1f}s)")
    except Exception as exc:
        print(f"  LSTM+XGB FAILED: {exc}")

    # STL hybrids
    for kind in ("xgb", "ann", "lstm"):
        try:
            t1 = time.time()
            yhat = stl_hybrid(y_train, X_train, X_val, X_full, val_idx, kind)
            _record(f"hybrid_stl_{kind}", yhat)
            print(f"    (STL+{kind.upper()} took {time.time() - t1:.1f}s)")
        except Exception as exc:
            print(f"  STL+{kind.upper()} FAILED: {exc}")

    print(f"  Total for {name}: {time.time() - t0:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    splits = Splits.from_config()
    g3 = load_g3()
    print(f"G3 loaded: {g3.shape}")

    all_rows = []
    for name, col in DAILY_SPECIALTIES.items():
        try:
            all_rows.extend(run_specialty(g3, splits, name, col))
        except Exception as exc:
            print(f"\n{name} crashed: {exc}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out = ROOT / "artefacts" / "metrics" / "task2_ml_hybrid_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote: {out.relative_to(ROOT)}")
    print(f"Total rows: {len(df)}")
    if not df.empty:
        print("\nSummary by specialty:")
        pivot = df.pivot_table(index="model", columns="specialty",
                                values="MAPE", aggfunc="first")
        pd.set_option("display.float_format", lambda v: f"{v:.3f}")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
