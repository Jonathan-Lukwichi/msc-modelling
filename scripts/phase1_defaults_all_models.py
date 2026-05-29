"""Phase 1 — All ML models (standalone + hybrids) with Chapter 5 default parameters.

For each of the 9 models:
  1. 10-fold rolling-origin CV inside train (per-fold MAPE/MAE/RMSE/R²)
  2. Rolling-origin weekly refit on val (and test)
  3. Aggregate daily predictions to weekly / monthly / yearly totals
  4. Persist per-fold + per-day + per-aggregation tables
  5. Save figures: CV-fold bar chart, predicted-vs-actual line plot

Outputs:
  artefacts/phase1_defaults/
    cv_folds_{model}.csv          per-fold metrics + average row
    daily_{model}.csv             per-day actual+predicted + per-row metrics
    weekly_{model}.csv            weekly aggregated metrics
    monthly_{model}.csv           monthly aggregated metrics
    yearly_{model}.csv            yearly aggregated metrics
    summary_phase1.csv            one-row-per-model headline metrics
    figures/{model}_cv_folds.png
    figures/{model}_pred_vs_actual.png
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score
from src.forecasting.cv import subsampled_rolling_origin


# -------------------------------------------------------------------------
# Chapter 5 "reasonable" default parameters (mid-range from HPO grids)
# -------------------------------------------------------------------------
DEFAULTS = {
    "xgboost": {"n_estimators": 200, "max_depth": 5,
                "learning_rate": 0.1, "subsample": 0.85},
    "ann":     {"hidden_layers": 2, "units": 128, "dropout": 0.2,
                "learning_rate": 0.001, "batch_size": 32, "seed": 42},
    "lstm":    {"lookback": 14, "units": 96, "dropout": 0.2,
                "learning_rate": 0.001, "batch_size": 32, "seed": 42},
    # Chapter 5 §5.2.2 SARIMA template mid-range (Chapter 5 PDF reference)
    "arima":   {"order": (1, 1, 1)},                  # (p, 1, q) mid of {0,1,2}
    "sarimax": {"order": (1, 1, 1),
                "seasonal_order": (1, 1, 1, 7)},     # m=7, mid of template
}

OUT = ROOT / "artefacts" / "phase1_defaults"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_data():
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    # Inner join on dates (engineered has lag warmup NaNs)
    df = pd.concat([target.rename("y"), X_consensus],
                   axis=1, join="inner").dropna()
    y = df["y"]
    X = df.drop(columns=["y"])
    train_idx = splits.slice(pd.DataFrame(index=y.index), "train").index
    val_idx   = splits.slice(pd.DataFrame(index=y.index), "val").index
    test_idx  = splits.slice(pd.DataFrame(index=y.index), "test").index
    return splits, y, X, train_idx, val_idx, test_idx


# -------------------------------------------------------------------------
# Cross-validation helper (10-fold rolling-origin inside train)
# -------------------------------------------------------------------------
def cv_score(model_name: str, X: pd.DataFrame, y: pd.Series,
             train_idx: pd.DatetimeIndex, fit_predict_fn, n_folds: int = 10):
    """Run 10-fold rolling-origin CV; return DataFrame with per-fold metrics."""
    X_tr = X.loc[train_idx]
    y_tr = y.loc[train_idx]
    folds = subsampled_rolling_origin(X_tr.index, n_folds=n_folds,
                                      horizon_days=7, step_days=7,
                                      min_train_days=365)
    rows = []
    for f in folds:
        Xtr = X_tr.iloc[f.train_idx]
        ytr = y_tr.iloc[f.train_idx]
        Xte = X_tr.iloc[f.test_idx]
        yte = y_tr.iloc[f.test_idx]
        yhat = fit_predict_fn(Xtr, ytr, Xte)
        m = score(yte.values, np.asarray(yhat).ravel())
        rows.append({"fold": f.fold_id, "origin": f.origin.date().isoformat(),
                     **m})
    df = pd.DataFrame(rows)
    # Average row at the end
    avg = {"fold": "AVG", "origin": ""}
    for k in ("MAPE", "MAE", "RMSE", "R2"):
        avg[k] = df[k].mean()
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df


# -------------------------------------------------------------------------
# Per-model fit_predict definitions (Chapter 5 defaults)
# -------------------------------------------------------------------------
def xgb_fit_predict(Xtr, ytr, Xte, params=None):
    from xgboost import XGBRegressor
    p = params or DEFAULTS["xgboost"]
    m = XGBRegressor(**p, objective="reg:squarederror",
                     random_state=42, verbosity=0, n_jobs=-1)
    m.fit(Xtr.values, ytr.values)
    return m.predict(Xte.values)


def ann_fit_predict(Xtr, ytr, Xte, params=None):
    """Standardise on Xtr then fit MLP per Chapter 5 default."""
    import torch, torch.nn as nn, torch.optim as optim
    p = params or DEFAULTS["ann"]
    torch.manual_seed(p["seed"])
    mean = Xtr.mean(); std = Xtr.std(ddof=0).replace(0, 1.0)
    Xtr_s = ((Xtr - mean) / std).values.astype(np.float32)
    Xte_s = ((Xte - mean) / std).values.astype(np.float32)
    ytr_v = ytr.values.astype(np.float32)
    y_mean, y_std = float(ytr_v.mean()), float(ytr_v.std() or 1.0)
    ytr_n = (ytr_v - y_mean) / y_std

    in_dim = Xtr_s.shape[1]
    layers = []
    last = in_dim
    for _ in range(p["hidden_layers"]):
        layers += [nn.Linear(last, p["units"]), nn.ReLU(),
                   nn.Dropout(p["dropout"])]
        last = p["units"]
    layers.append(nn.Linear(last, 1))
    net = nn.Sequential(*layers)

    opt = optim.Adam(net.parameters(), lr=p["learning_rate"])
    loss_fn = nn.MSELoss()
    bs = p["batch_size"]
    n = len(Xtr_s)
    for epoch in range(60):
        order = np.random.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i+bs]
            xb = torch.from_numpy(Xtr_s[ix])
            yb = torch.from_numpy(ytr_n[ix])
            opt.zero_grad()
            out = net(xb).squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        yhat_n = net(torch.from_numpy(Xte_s)).squeeze(-1).numpy()
    return yhat_n * y_std + y_mean


def lstm_fit_predict(Xtr, ytr, Xte, params=None):
    import torch, torch.nn as nn, torch.optim as optim
    p = params or DEFAULTS["lstm"]
    torch.manual_seed(p["seed"])
    L = p["lookback"]
    mean = Xtr.mean(); std = Xtr.std(ddof=0).replace(0, 1.0)
    Xtr_s = ((Xtr - mean) / std).values.astype(np.float32)
    Xte_s = ((Xte - mean) / std).values.astype(np.float32)
    ytr_v = ytr.values.astype(np.float32)
    y_mean, y_std = float(ytr_v.mean()), float(ytr_v.std() or 1.0)
    ytr_n = (ytr_v - y_mean) / y_std

    # Build sequences inside train
    def make_seqs(Xs, ys, L):
        seqs, tgts = [], []
        for i in range(L, len(Xs)):
            seqs.append(Xs[i-L:i])
            if ys is not None:
                tgts.append(ys[i])
        seqs = np.stack(seqs).astype(np.float32)
        if ys is None:
            return seqs, None
        return seqs, np.array(tgts, dtype=np.float32)

    Xtr_seq, ytr_seq = make_seqs(Xtr_s, ytr_n, L)

    class Net(nn.Module):
        def __init__(self, in_dim, units, dropout):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, units, batch_first=True,
                                dropout=0.0, num_layers=1)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(units, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    net = Net(Xtr_seq.shape[2], p["units"], p["dropout"])
    opt = optim.Adam(net.parameters(), lr=p["learning_rate"])
    loss_fn = nn.MSELoss()
    bs = p["batch_size"]; n = len(Xtr_seq)
    for epoch in range(50):
        order = np.random.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i+bs]
            xb = torch.from_numpy(Xtr_seq[ix])
            yb = torch.from_numpy(ytr_seq[ix])
            opt.zero_grad()
            out = net(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    net.eval()
    # Build test sequences using last L rows of train + test
    full = np.vstack([Xtr_s[-L:], Xte_s])
    Xte_seq, _ = make_seqs(full, None, L)
    with torch.no_grad():
        yhat_n = net(torch.from_numpy(Xte_seq)).squeeze(-1).numpy()
    return yhat_n * y_std + y_mean


# -------------------------------------------------------------------------
# Rolling-origin weekly refit on val (and test) — same protocol as production
# -------------------------------------------------------------------------
def rolling_forecast_generic(fit_predict_fn, X_full, y_full, block_idx,
                             step_days=7):
    rows = []
    block_start, block_end = block_idx[0], block_idx[-1]
    origin_pos = y_full.index.get_loc(block_start) - 1
    while origin_pos < y_full.index.get_loc(block_end):
        h = int(min(step_days,
                    y_full.index.get_loc(block_end) - origin_pos))
        Xtr = X_full.iloc[: origin_pos + 1]
        ytr = y_full.iloc[: origin_pos + 1]
        Xfu = X_full.iloc[origin_pos + 1 : origin_pos + 1 + h]
        yhat = fit_predict_fn(Xtr, ytr, Xfu)
        dates = X_full.index[origin_pos + 1 : origin_pos + 1 + h]
        for d, p in zip(dates, np.asarray(yhat).ravel()):
            rows.append({"date": d, "predicted": float(p)})
        origin_pos += step_days
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Aggregation helpers — daily → weekly / monthly / yearly
# -------------------------------------------------------------------------
def aggregate_metrics(pred_df: pd.DataFrame, freq: str):
    """pred_df must have columns date, actual, predicted. freq in {W, M, Y}."""
    pred_df = pred_df.set_index("date")
    grouped = pred_df.resample(freq).agg(
        actual=("actual", "sum"),
        predicted=("predicted", "sum"),
    ).reset_index()
    rows = []
    for _, r in grouped.iterrows():
        if r["actual"] == 0:
            continue
        rows.append({
            "period": r["date"].date().isoformat(),
            "actual": float(r["actual"]),
            "predicted": float(r["predicted"]),
            "abs_error": abs(r["actual"] - r["predicted"]),
            "pct_error": abs(r["actual"] - r["predicted"]) / r["actual"] * 100,
        })
    df = pd.DataFrame(rows)
    # Average row at the end
    if not df.empty:
        avg = {"period": "AVG",
               "actual": df["actual"].mean(),
               "predicted": df["predicted"].mean(),
               "abs_error": df["abs_error"].mean(),
               "pct_error": df["pct_error"].mean()}
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------
def plot_cv_folds(cv_df: pd.DataFrame, model_name: str):
    df = cv_df[cv_df["fold"] != "AVG"].copy()
    df["fold"] = df["fold"].astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["fold"], df["RMSE"], color="#1f77b4", alpha=0.85, label="RMSE")
    ax2 = ax.twinx()
    ax2.plot(df["fold"], df["MAPE"], "o-", color="#ff7f0e", label="MAPE %")
    avg_rmse = cv_df[cv_df["fold"] == "AVG"]["RMSE"].iloc[0]
    avg_mape = cv_df[cv_df["fold"] == "AVG"]["MAPE"].iloc[0]
    ax.axhline(avg_rmse, color="#1f77b4", ls="--", alpha=0.6,
               label=f"Avg RMSE = {avg_rmse:.2f}")
    ax2.axhline(avg_mape, color="#ff7f0e", ls="--", alpha=0.6,
                label=f"Avg MAPE = {avg_mape:.2f}%")
    ax.set_xlabel("Rolling-origin fold")
    ax.set_ylabel("RMSE", color="#1f77b4")
    ax2.set_ylabel("MAPE (%)", color="#ff7f0e")
    ax.set_title(f"{model_name} — 10-fold CV (Chapter 5 defaults)")
    fig.legend(loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(FIG / f"{model_name}_cv_folds.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_actual(daily_df: pd.DataFrame, model_name: str,
                        target_name: str = "Daily ED arrivals"):
    df = daily_df[daily_df["date"] != "AVG"].copy()
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["actual"], lw=1.2, color="#333",
            label="Actual", alpha=0.85)
    ax.plot(df["date"], df["predicted"], lw=1.2, color="#d62728",
            label="Predicted", alpha=0.85)
    mape = df["pct_error"].mean()
    rmse = float(np.sqrt(((df["actual"] - df["predicted"]) ** 2).mean()))
    ax.set_title(f"{model_name} — {target_name} (val window)  "
                 f"MAPE {mape:.2f}%  RMSE {rmse:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_name)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"{model_name}_pred_vs_actual.png", dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------------
# Hybrid model orchestration
# -------------------------------------------------------------------------
def stl_hybrid_fit_predict(refiner_kind: str, Xtr, ytr, Xte):
    """STL decomposition + ML refiner on residual."""
    from src.forecasting.hybrids.stl_hybrid import (
        decompose_train, forecast_trend, forecast_seasonal,
    )
    decomp = decompose_train(ytr)
    val_idx = Xte.index
    trend_fcst = forecast_trend(decomp.trend, val_idx)
    seas_fcst = forecast_seasonal(decomp.seasonal, val_idx)
    # Train refiner on residual
    if refiner_kind == "xgboost":
        resid_pred = xgb_fit_predict(Xtr, decomp.residual, Xte)
    elif refiner_kind == "ann":
        resid_pred = ann_fit_predict(Xtr, decomp.residual, Xte)
    elif refiner_kind == "lstm":
        resid_pred = lstm_fit_predict(Xtr, decomp.residual, Xte)
    else:
        raise ValueError(refiner_kind)
    return trend_fcst.values + seas_fcst.values + np.asarray(resid_pred).ravel()


def sarimax_hybrid_fit_predict(refiner_kind: str, Xtr, ytr, Xte):
    """SARIMAX residual hybrid: fit SARIMAX on raw-10 exog, refine residuals with ML."""
    from src.forecasting.features import build_task1_exogenous
    from src.forecasting.models.sarimax import fit_with_order
    # Load cached SARIMAX order
    order = (1, 1, 1); seas = (0, 1, 1, 7)
    try:
        cfg = {}
        for line in (ROOT / "artefacts" / "models" / "sarimax_order.txt"
                     ).read_text().strip().splitlines():
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
        order = eval(cfg["order"]); seas = eval(cfg["seasonal_order"])
    except Exception:
        pass
    g1 = load_g1()
    Xexog_train, scaler = build_task1_exogenous(g1.loc[Xtr.index],
                                                 fit_scaler=True)
    Xexog_test, _ = build_task1_exogenous(g1.loc[Xte.index], scaler=scaler)
    fit = fit_with_order(ytr, Xexog_train, order, seas)
    # Get SARIMAX out-of-sample point forecast
    from pmdarima import ARIMA as PmARIMA
    m = PmARIMA(order=order, seasonal_order=seas, suppress_warnings=True)
    m.fit(ytr.values, X=Xexog_train.values)
    sarimax_pred = m.predict(n_periods=len(Xte), X=Xexog_test.values)
    # Compute training residuals = y - in_sample_pred
    in_sample = m.predict_in_sample(X=Xexog_train.values)
    resid = pd.Series(ytr.values - in_sample, index=Xtr.index)
    # Refine
    if refiner_kind == "xgboost":
        rp = xgb_fit_predict(Xtr, resid, Xte)
    elif refiner_kind == "lstm":
        rp = lstm_fit_predict(Xtr, resid, Xte)
    else:
        raise ValueError(refiner_kind)
    return np.asarray(sarimax_pred).ravel() + np.asarray(rp).ravel()


def arima_fit_predict(Xtr, ytr, Xte, params=None):
    """ARIMA(p,1,q) — Chapter 5 default order from {0,1,2} template."""
    from pmdarima import ARIMA as PmARIMA
    p = params or DEFAULTS["arima"]
    m = PmARIMA(order=p["order"], suppress_warnings=True)
    m.fit(ytr.values)
    return m.predict(n_periods=len(Xte))


def sarimax_fit_predict(Xtr, ytr, Xte, params=None):
    """SARIMAX(p,1,q)(P,1,Q)_7 with raw-10 exogenous block (Chapter 5)."""
    from pmdarima import ARIMA as PmARIMA
    from src.forecasting.features import build_task1_exogenous
    p = params or DEFAULTS["sarimax"]
    g1 = load_g1()
    Xexog_train, scaler = build_task1_exogenous(g1.loc[Xtr.index],
                                                 fit_scaler=True)
    Xexog_test, _ = build_task1_exogenous(g1.loc[Xte.index], scaler=scaler)
    m = PmARIMA(order=p["order"], seasonal_order=p["seasonal_order"],
                suppress_warnings=True)
    m.fit(ytr.values, X=Xexog_train.values)
    return m.predict(n_periods=len(Xte), X=Xexog_test.values)


def lstm_xgb_hybrid_fit_predict(Xtr, ytr, Xte):
    """LSTM gives the level, XGB refines residual."""
    lstm_pred_tr = lstm_fit_predict(Xtr, ytr, Xtr)  # in-sample-ish pred
    lstm_pred_te = lstm_fit_predict(Xtr, ytr, Xte)
    # Align lengths (LSTM drops the first L rows)
    L = DEFAULTS["lstm"]["lookback"]
    # Use last len(Xtr)-L predictions for residual
    Xtr_for_resid = Xtr.iloc[L:]
    ytr_for_resid = ytr.iloc[L:]
    if len(lstm_pred_tr) == len(ytr):
        # Predicted whole train range
        resid = ytr.values - np.asarray(lstm_pred_tr).ravel()
    else:
        # LSTM only predicts from position L onward
        resid = ytr_for_resid.values - np.asarray(lstm_pred_tr).ravel()[
            :len(ytr_for_resid)]
        Xtr = Xtr_for_resid
    resid_ser = pd.Series(resid, index=Xtr.index)
    rp = xgb_fit_predict(Xtr, resid_ser, Xte)
    return np.asarray(lstm_pred_te).ravel() + np.asarray(rp).ravel()


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
MODEL_REGISTRY = {
    # Classical baselines (Chapter 5 §5.2.2 template)
    "arima":             lambda Xtr, ytr, Xte: arima_fit_predict(Xtr, ytr, Xte),
    "sarimax":           lambda Xtr, ytr, Xte: sarimax_fit_predict(Xtr, ytr, Xte),
    # ML standalones
    "xgboost":           lambda Xtr, ytr, Xte: xgb_fit_predict(Xtr, ytr, Xte),
    "ann":               lambda Xtr, ytr, Xte: ann_fit_predict(Xtr, ytr, Xte),
    "lstm":              lambda Xtr, ytr, Xte: lstm_fit_predict(Xtr, ytr, Xte),
    # Hybrids
    "stl_xgb":           lambda Xtr, ytr, Xte: stl_hybrid_fit_predict("xgboost", Xtr, ytr, Xte),
    "stl_ann":           lambda Xtr, ytr, Xte: stl_hybrid_fit_predict("ann", Xtr, ytr, Xte),
    "stl_lstm":          lambda Xtr, ytr, Xte: stl_hybrid_fit_predict("lstm", Xtr, ytr, Xte),
    "sarimax_xgb":       lambda Xtr, ytr, Xte: sarimax_hybrid_fit_predict("xgboost", Xtr, ytr, Xte),
    "sarimax_lstm":      lambda Xtr, ytr, Xte: sarimax_hybrid_fit_predict("lstm", Xtr, ytr, Xte),
    "lstm_xgb":          lambda Xtr, ytr, Xte: lstm_xgb_hybrid_fit_predict(Xtr, ytr, Xte),
}


def run_one_model(name: str, fit_predict_fn, splits, y, X, train_idx, val_idx):
    print(f"\n{'='*70}\n[{name}] starting...\n{'='*70}")
    t0 = time.time()

    # 1) 10-fold CV
    print(f"  [1/4] 10-fold rolling-origin CV...")
    n_folds = 10 if "lstm" not in name else 5  # speed
    try:
        cv_df = cv_score(name, X, y, train_idx, fit_predict_fn, n_folds=n_folds)
    except Exception as exc:
        print(f"    FAILED CV: {exc}")
        return None
    cv_df.to_csv(OUT / f"cv_folds_{name}.csv", index=False)
    print(f"    avg RMSE = {cv_df[cv_df['fold']=='AVG']['RMSE'].iloc[0]:.3f}  "
          f"avg MAPE = {cv_df[cv_df['fold']=='AVG']['MAPE'].iloc[0]:.2f}%  "
          f"({time.time()-t0:.0f}s)")

    # 2) Rolling-origin weekly refit on val
    print(f"  [2/4] Rolling-origin weekly refit on val...")
    try:
        pred = rolling_forecast_generic(fit_predict_fn, X, y, val_idx,
                                         step_days=7)
    except Exception as exc:
        print(f"    FAILED val: {exc}")
        return None
    pred["actual"] = y.loc[pred["date"]].values

    # 3) Daily metrics + per-row table
    daily_rows = []
    for _, r in pred.iterrows():
        daily_rows.append({
            "date": r["date"].date().isoformat(),
            "actual": float(r["actual"]),
            "predicted": float(r["predicted"]),
            "abs_error": abs(r["actual"] - r["predicted"]),
            "pct_error": abs(r["actual"] - r["predicted"]) / max(r["actual"], 1e-9) * 100,
        })
    daily_df = pd.DataFrame(daily_rows)
    avg = {"date": "AVG",
           "actual": daily_df["actual"].mean(),
           "predicted": daily_df["predicted"].mean(),
           "abs_error": daily_df["abs_error"].mean(),
           "pct_error": daily_df["pct_error"].mean()}
    daily_df_out = pd.concat([daily_df, pd.DataFrame([avg])], ignore_index=True)
    daily_df_out.to_csv(OUT / f"daily_{name}.csv", index=False)

    # 4) Weekly / Monthly / Yearly aggregation
    pred_for_agg = pred.copy()
    weekly = aggregate_metrics(pred_for_agg.copy(), "W")
    monthly = aggregate_metrics(pred_for_agg.copy(), "ME")
    yearly = aggregate_metrics(pred_for_agg.copy(), "YE")
    weekly.to_csv(OUT / f"weekly_{name}.csv", index=False)
    monthly.to_csv(OUT / f"monthly_{name}.csv", index=False)
    yearly.to_csv(OUT / f"yearly_{name}.csv", index=False)

    # Headline val metrics on raw daily preds
    m = score(daily_df["actual"].values, daily_df["predicted"].values)

    # Figures
    try:
        plot_cv_folds(cv_df, name)
        plot_pred_vs_actual(daily_df_out, name)
    except Exception as exc:
        print(f"    [warn] figure failed: {exc}")

    print(f"  [done] val MAPE = {m['MAPE']:.2f}%  RMSE = {m['RMSE']:.3f}  "
          f"(total {time.time()-t0:.0f}s)")

    return {
        "model": name,
        "cv_avg_RMSE": float(cv_df[cv_df['fold']=='AVG']['RMSE'].iloc[0]),
        "cv_avg_MAPE": float(cv_df[cv_df['fold']=='AVG']['MAPE'].iloc[0]),
        "val_MAPE": float(m["MAPE"]),
        "val_MAE":  float(m["MAE"]),
        "val_RMSE": float(m["RMSE"]),
        "val_R2":   float(m["R2"]),
        "weekly_avg_pct_error":  float(weekly[weekly["period"]=="AVG"]["pct_error"].iloc[0])
                                  if len(weekly) > 0 else np.nan,
        "monthly_avg_pct_error": float(monthly[monthly["period"]=="AVG"]["pct_error"].iloc[0])
                                  if len(monthly) > 0 else np.nan,
        "yearly_avg_pct_error":  float(yearly[yearly["period"]=="AVG"]["pct_error"].iloc[0])
                                  if len(yearly) > 0 else np.nan,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        help="comma-list of models to run; default 'all'")
    args = parser.parse_args()

    picks = (list(MODEL_REGISTRY.keys()) if args.models == "all"
             else [m.strip() for m in args.models.split(",")])
    print(f"Models to run: {picks}")

    splits, y, X, train_idx, val_idx, test_idx = load_data()
    print(f"Loaded data: y={len(y)}, X={X.shape}, "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Chapter 5 defaults:\n{json.dumps(DEFAULTS, indent=2)}")

    summary = []
    for name in picks:
        fn = MODEL_REGISTRY.get(name)
        if fn is None:
            print(f"[skip] unknown model: {name}")
            continue
        res = run_one_model(name, fn, splits, y, X, train_idx, val_idx)
        if res is not None:
            summary.append(res)
            pd.DataFrame(summary).to_csv(OUT / "summary_phase1.csv",
                                         index=False)

    print("\n" + "="*70)
    print("PHASE 1 SUMMARY (all models, Chapter 5 defaults)")
    print("="*70)
    if summary:
        df = pd.DataFrame(summary)
        print(df.to_string(index=False))
        df.to_csv(OUT / "summary_phase1.csv", index=False)


if __name__ == "__main__":
    main()
