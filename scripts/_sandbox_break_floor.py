"""Sandbox: 6 attempts to break the 12 % MAPE floor.

Throwaway script — should be deleted after the test.
  #1  Prophet (Meta) with weekly + yearly seasonality, rolling weekly refit
  #2  LightGBM (gradient boosting alternative to XGBoost)
  #3  GRU (PyTorch — LSTM variant with fewer params)
  #4  log1p-transformed XGBoost (variance-stabilising for counts)
  #5  Fourier-features-only XGBoost (no lag features)
  #6  Huber-loss XGBoost (robust to outliers)
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.features import build_task1_exogenous
from src.forecasting.metrics import score


def main():
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    train_idx = splits.slice(g1, "train").index
    val_idx = splits.slice(g1, "val").index
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    df_full = pd.concat([target.rename("y"), X_consensus], axis=1,
                          join="inner").dropna()
    train_idx_a = train_idx.intersection(df_full.index)
    val_idx_a = val_idx.intersection(df_full.index)
    print(f"Train: {len(train_idx_a)}  Val: {len(val_idx_a)}\n")
    results = []

    # ---------- TEST 1: PROPHET ----------
    print("=" * 70)
    print("[1] PROPHET (weekly + yearly seasonality, rolling weekly refit)")
    print("=" * 70)
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        t0 = time.time()
        rows = []
        all_dates = list(target.index)
        origin_pos = all_dates.index(val_idx_a[0]) - 1
        end_pos = all_dates.index(val_idx_a[-1])
        while origin_pos < end_pos:
            h = int(min(7, end_pos - origin_pos))
            fit_dates = all_dates[: origin_pos + 1]
            fit_df = pd.DataFrame({"ds": fit_dates,
                                    "y": target.loc[fit_dates].values})
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                         daily_seasonality=False,
                         changepoint_prior_scale=0.05,
                         seasonality_mode="additive")
            m.fit(fit_df)
            future_dates = all_dates[origin_pos + 1 : origin_pos + 1 + h]
            future = pd.DataFrame({"ds": future_dates})
            fc = m.predict(future)
            for d, yp in zip(future_dates, fc["yhat"].values):
                rows.append({"date": d, "predicted": float(yp)})
            origin_pos += 7
        preds = pd.DataFrame(rows).set_index("date")
        preds["actual"] = target.loc[preds.index]
        s = score(preds["actual"], preds["predicted"])
        s["time_s"] = round(time.time() - t0, 1)
        results.append({"model": "Prophet", **s})
        print(f"  MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  "
              f"RMSE={s['RMSE']:.3f}  R2={s['R2']:+.3f}  [{s['time_s']}s]")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ---------- TEST 2: LIGHTGBM ----------
    print("\n" + "=" * 70)
    print("[2] LightGBM (gradient boosting alternative)")
    print("=" * 70)
    try:
        from lightgbm import LGBMRegressor
        t0 = time.time()
        rows = []
        origin = train_idx_a[-1]
        val_remaining = list(val_idx_a)
        while val_remaining:
            h = min(7, len(val_remaining))
            future = val_remaining[:h]
            tr_dates = df_full.index[df_full.index <= origin]
            m = LGBMRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.05, subsample=0.85,
                                random_state=42, n_jobs=-1, verbose=-1)
            m.fit(df_full.loc[tr_dates].drop(columns=["y"]).values,
                  df_full.loc[tr_dates, "y"].values)
            yhat = m.predict(df_full.loc[future].drop(columns=["y"]).values)
            for d, yp in zip(future, yhat):
                rows.append({"date": d, "predicted": float(yp)})
            origin = future[-1]
            val_remaining = val_remaining[h:]
        preds = pd.DataFrame(rows).set_index("date")
        preds["actual"] = target.loc[preds.index]
        s = score(preds["actual"], preds["predicted"])
        s["time_s"] = round(time.time() - t0, 1)
        results.append({"model": "LightGBM", **s})
        print(f"  MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  "
              f"RMSE={s['RMSE']:.3f}  R2={s['R2']:+.3f}  [{s['time_s']}s]")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ---------- TEST 3: GRU ----------
    print("\n" + "=" * 70)
    print("[3] GRU (PyTorch, lookback=14, 128 units, single fit)")
    print("=" * 70)
    try:
        import torch
        from torch import nn, optim
        import random
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        t0 = time.time()
        lookback = 14
        X_all = df_full.drop(columns=["y"]).values.astype(np.float32)
        y_all = df_full["y"].values.astype(np.float32)
        # Standardise based on train segment
        n_train = len(train_idx_a)
        Xmean = X_all[:n_train].mean(axis=0)
        Xstd = X_all[:n_train].std(axis=0) + 1e-9
        Xn = ((X_all - Xmean) / Xstd).astype(np.float32)
        ymean = float(y_all[:n_train].mean())
        ystd = float(y_all[:n_train].std() + 1e-9)
        yn = ((y_all - ymean) / ystd).astype(np.float32)

        class GRUNet(nn.Module):
            def __init__(self, n_features, units=128, dropout=0.2):
                super().__init__()
                self.gru = nn.GRU(n_features, units, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(units, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(self.drop(out[:, -1, :])).squeeze(-1)

        def make_seq(X, y, lb):
            n = len(y) - lb
            Xs = np.empty((n, lb, X.shape[1]), dtype=np.float32)
            ys = np.empty(n, dtype=np.float32)
            for i in range(n):
                Xs[i] = X[i : i + lb]
                ys[i] = y[i + lb]
            return Xs, ys

        Xs, ys = make_seq(Xn, yn, lookback)
        train_n = n_train - lookback
        X_tr = torch.from_numpy(Xs[:train_n])
        y_tr_t = torch.from_numpy(ys[:train_n])
        model = GRUNet(Xn.shape[1], units=128, dropout=0.2)
        opt = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        bs = 32
        for epoch in range(50):
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                loss = loss_fn(model(X_tr[idx]), y_tr_t[idx])
                loss.backward()
                opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i, date in enumerate(val_idx_a):
                pos = n_train + i
                if pos < lookback:
                    continue
                window = Xn[pos - lookback : pos]
                preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
        yhat = np.array(preds) * ystd + ymean
        actual_vals = target.loc[val_idx_a].values[-len(yhat):]
        s = score(actual_vals, yhat)
        s["time_s"] = round(time.time() - t0, 1)
        results.append({"model": "GRU", **s})
        print(f"  MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  "
              f"RMSE={s['RMSE']:.3f}  R2={s['R2']:+.3f}  [{s['time_s']}s]")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {e}")

    # ---------- TEST 4: log1p-XGB ----------
    print("\n" + "=" * 70)
    print("[4] log1p-transformed XGBoost (variance-stabilising for counts)")
    print("=" * 70)
    try:
        from xgboost import XGBRegressor
        t0 = time.time()
        rows = []
        origin = train_idx_a[-1]
        val_remaining = list(val_idx_a)
        while val_remaining:
            h = min(7, len(val_remaining))
            future = val_remaining[:h]
            tr_dates = df_full.index[df_full.index <= origin]
            m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                              subsample=1.0, random_state=42, n_jobs=-1,
                              verbosity=0)
            m.fit(df_full.loc[tr_dates].drop(columns=["y"]).values,
                  np.log1p(df_full.loc[tr_dates, "y"].values))
            yhat = np.expm1(m.predict(df_full.loc[future].drop(columns=["y"]).values))
            for d, yp in zip(future, yhat):
                rows.append({"date": d, "predicted": float(yp)})
            origin = future[-1]
            val_remaining = val_remaining[h:]
        preds = pd.DataFrame(rows).set_index("date")
        preds["actual"] = target.loc[preds.index]
        s = score(preds["actual"], preds["predicted"])
        s["time_s"] = round(time.time() - t0, 1)
        results.append({"model": "XGB+log1p", **s})
        print(f"  MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  "
              f"RMSE={s['RMSE']:.3f}  R2={s['R2']:+.3f}  [{s['time_s']}s]")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ---------- TEST 5: Fourier-only XGB ----------
    print("\n" + "=" * 70)
    print("[5] Fourier-features-only XGBoost (no lag features)")
    print("=" * 70)
    try:
        from xgboost import XGBRegressor
        t0 = time.time()
        t_index = np.arange(len(df_full))
        feats = {}
        for k in range(1, 6):
            feats[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * t_index / 7)
            feats[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * t_index / 7)
        for k in range(1, 4):
            feats[f"annual_sin_{k}"] = np.sin(2 * np.pi * k * t_index / 365.25)
            feats[f"annual_cos_{k}"] = np.cos(2 * np.pi * k * t_index / 365.25)
        fourier_df = pd.DataFrame(feats, index=df_full.index)
        X_basic, _ = build_task1_exogenous(g1, fit_scaler=True)
        fourier_df = fourier_df.join(X_basic.filter(regex="^is_"),
                                        how="left").fillna(0)
        rows = []
        origin = train_idx_a[-1]
        val_remaining = list(val_idx_a)
        while val_remaining:
            h = min(7, len(val_remaining))
            future = val_remaining[:h]
            tr_dates = fourier_df.index[fourier_df.index <= origin]
            tr_dates = tr_dates.intersection(df_full.index)
            m = XGBRegressor(n_estimators=200, max_depth=4,
                              learning_rate=0.05, subsample=0.85,
                              random_state=42, n_jobs=-1, verbosity=0)
            m.fit(fourier_df.loc[tr_dates].values,
                  target.loc[tr_dates].values)
            yhat = m.predict(fourier_df.loc[future].values)
            for d, yp in zip(future, yhat):
                rows.append({"date": d, "predicted": float(yp)})
            origin = future[-1]
            val_remaining = val_remaining[h:]
        preds = pd.DataFrame(rows).set_index("date")
        preds["actual"] = target.loc[preds.index]
        s = score(preds["actual"], preds["predicted"])
        s["time_s"] = round(time.time() - t0, 1)
        results.append({"model": "XGB+Fourier_only", **s})
        print(f"  MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  "
              f"RMSE={s['RMSE']:.3f}  R2={s['R2']:+.3f}  [{s['time_s']}s]")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ---------- TEST 6: Huber-XGB ----------
    print("\n" + "=" * 70)
    print("[6] Huber-loss XGBoost (robust to outlier days)")
    print("=" * 70)
    try:
        from xgboost import XGBRegressor
        t0 = time.time()
        rows = []
        origin = train_idx_a[-1]
        val_remaining = list(val_idx_a)
        while val_remaining:
            h = min(7, len(val_remaining))
            future = val_remaining[:h]
            tr_dates = df_full.index[df_full.index <= origin]
            m = XGBRegressor(n_estimators=200, max_depth=4,
                              learning_rate=0.05, subsample=0.85,
                              random_state=42, n_jobs=-1, verbosity=0,
                              objective="reg:pseudohubererror",
                              huber_slope=1.0)
            m.fit(df_full.loc[tr_dates].drop(columns=["y"]).values,
                  df_full.loc[tr_dates, "y"].values)
            yhat = m.predict(df_full.loc[future].drop(columns=["y"]).values)
            for d, yp in zip(future, yhat):
                rows.append({"date": d, "predicted": float(yp)})
            origin = future[-1]
            val_remaining = val_remaining[h:]
        preds = pd.DataFrame(rows).set_index("date")
        preds["actual"] = target.loc[preds.index]
        s = score(preds["actual"], preds["predicted"])
        s["time_s"] = round(time.time() - t0, 1)
        results.append({"model": "XGB+Huber", **s})
        print(f"  MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  "
              f"RMSE={s['RMSE']:.3f}  R2={s['R2']:+.3f}  [{s['time_s']}s]")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "=" * 70 + "\nSUMMARY (sorted by MAPE)\n" + "=" * 70)
    df_res = pd.DataFrame(results).sort_values("MAPE")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(df_res.to_string(index=False))
    print("\nReference (existing main study): "
          "XGBoost k=10 CV val MAPE = 12.02 %  /  best ensemble = 11.66 %")


if __name__ == "__main__":
    main()
