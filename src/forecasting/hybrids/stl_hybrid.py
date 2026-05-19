"""STL-decomposition hybrids per plan §12.1 / Ch3 §3.5.4 Alg 7.

  y_t = T_t + S_t + R_t            (STL, period=7, robust)
  T_hat = linear extrapolation
  S_hat = seasonal-naive  S_{T+h} = S_{T+h-s}
  ML refiner: f_B(X_t) -> R_t

  y_hat = T_hat + S_hat + f_B(X_future)

Three concrete hybrids:
  - STL + XGBoost
  - STL + ANN
  - STL + LSTM
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class StlDecomp:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series


def decompose_train(y_train: pd.Series, period: int = 7) -> StlDecomp:
    """STL with period=7, robust=True (Ch5 §5.2.2 lag-7 dominant)."""
    res = STL(y_train, period=period, robust=True).fit()
    return StlDecomp(
        trend=pd.Series(res.trend, index=y_train.index, name="trend"),
        seasonal=pd.Series(res.seasonal, index=y_train.index, name="seasonal"),
        residual=pd.Series(res.resid, index=y_train.index, name="residual"),
    )


def forecast_trend(trend_train: pd.Series, val_idx: pd.DatetimeIndex,
                    method: str = "damped_linear") -> pd.Series:
    """Forecast the STL trend component over the val period.

    ``method='last'``: hold the last training trend value constant.
    ``method='damped_linear'`` (default): linear slope from last 30 days but
    dampened by a factor of 0.9^step to avoid runaway over 184 days
    (Gardner-McKenzie damped trend; standard in stlf when long-horizon).
    """
    n_tail = min(30, len(trend_train))
    last_level = trend_train.iloc[-1]
    if method == "last":
        return pd.Series([last_level] * len(val_idx), index=val_idx,
                          name="trend_forecast")
    if method == "damped_linear":
        y = trend_train.iloc[-n_tail:].values
        x = np.arange(n_tail)
        slope, intercept = np.polyfit(x, y, deg=1)
        h = len(val_idx)
        phi = 0.9
        # Damped multi-step: sum_{i=1..t} phi^i = phi * (1 - phi^t) / (1 - phi)
        steps = np.arange(1, h + 1)
        damping = phi * (1 - phi ** steps) / (1 - phi)
        forecast = last_level + slope * damping
        return pd.Series(forecast, index=val_idx, name="trend_forecast")
    raise ValueError(f"Unknown method: {method}")


def forecast_seasonal(seasonal_train: pd.Series, val_idx: pd.DatetimeIndex,
                      period: int = 7) -> pd.Series:
    """Seasonal-naive: S_{T+h} = S_{T+h-s}.

    Use the trailing `period` days of the trained seasonal component to seed
    the forecast, then tile.
    """
    tail = seasonal_train.iloc[-period:].values
    h = len(val_idx)
    n_repeats = (h + period - 1) // period
    tiled = np.tile(tail, n_repeats)[:h]
    return pd.Series(tiled, index=val_idx, name="seasonal_forecast")


def fit_xgb_refiner_on_residual(
    X_train: pd.DataFrame, residual: pd.Series, seed: int = 42,
):
    """XGBoost refiner on STL residuals."""
    from xgboost import XGBRegressor
    common_idx = X_train.index.intersection(residual.index)
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
        objective="reg:squarederror", random_state=seed, verbosity=0, n_jobs=-1,
    )
    model.fit(X_train.loc[common_idx].values,
              residual.loc[common_idx].values)
    return model


def fit_ann_refiner_on_residual(
    X_train: pd.DataFrame, residual: pd.Series, seed: int = 42,
):
    """ANN refiner on STL residuals (light defaults)."""
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    _seed_everything(seed)
    common_idx = X_train.index.intersection(residual.index)
    X = X_train.loc[common_idx]
    r = residual.loc[common_idx]
    mean = X.mean()
    std = X.std(ddof=0).replace(0, 1.0)
    Xs = ((X - mean) / std).astype(np.float32)
    r_mean, r_std = float(r.mean()), float(r.std(ddof=0)) if r.std(ddof=0) > 0 else 1.0
    rs = ((r - r_mean) / r_std).astype(np.float32)
    n_es = 28
    Xtr_t = torch.from_numpy(Xs.values[:-n_es])
    rtr_t = torch.from_numpy(rs.values[:-n_es])
    Xes_t = torch.from_numpy(Xs.values[-n_es:])
    res_t = torch.from_numpy(rs.values[-n_es:])
    params = {"hidden_layers": 2, "units": 128, "dropout": 0.2,
              "learning_rate": 0.001, "batch_size": 32, "seed": seed}
    model, _, _ = _train_one(Xtr_t, rtr_t, Xes_t, res_t, params, max_epochs=80)
    return model, mean, std, r_mean, r_std


def fit_lstm_refiner_on_residual(
    X_train: pd.DataFrame, residual: pd.Series, seed: int = 42,
):
    """LSTM refiner on STL residuals (light defaults, lookback=14)."""
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    _seed_everything(seed)
    common_idx = X_train.index.intersection(residual.index)
    X = X_train.loc[common_idx]
    r = residual.loc[common_idx]
    mean = X.mean()
    std = X.std(ddof=0).replace(0, 1.0)
    Xs = ((X - mean) / std).astype(np.float32).values
    r_mean, r_std = float(r.mean()), float(r.std(ddof=0)) if r.std(ddof=0) > 0 else 1.0
    rs = ((r - r_mean) / r_std).astype(np.float32).values
    lookback = 14
    Xs_seq, rs_seq = _build_sequences(Xs, rs, lookback)
    n_es = 28
    params = {"units": 64, "dropout": 0.2, "learning_rate": 0.001,
              "batch_size": 32, "seed": seed}
    model, _ = _train_one(Xs_seq[:-n_es], rs_seq[:-n_es],
                           Xs_seq[-n_es:], rs_seq[-n_es:],
                           params, max_epochs=40)
    return model, mean, std, r_mean, r_std, lookback


def refiner_predict_val(refiner_kind: str, refiner, X_full: pd.DataFrame,
                         val_idx: pd.DatetimeIndex) -> pd.Series:
    """Dispatch on refiner kind. XGBoost takes flat X; LSTM/ANN need scaling."""
    import torch
    if refiner_kind == "xgb":
        X_val = X_full.loc[val_idx]
        yhat = refiner.predict(X_val.values)
        return pd.Series(yhat, index=val_idx, name="refiner_pred")
    if refiner_kind == "ann":
        model, mean, std, r_mean, r_std = refiner
        X_val = X_full.loc[val_idx]
        Xs = ((X_val - mean) / std).astype(np.float32)
        model.eval()
        with torch.no_grad():
            yhat_norm = model(torch.from_numpy(Xs.values)).numpy()
        return pd.Series(yhat_norm * r_std + r_mean, index=val_idx,
                          name="refiner_pred")
    if refiner_kind == "lstm":
        model, mean, std, r_mean, r_std, lookback = refiner
        Xs = ((X_full - mean) / std).astype(np.float32).values
        rows = []
        train_end_pos = X_full.index.get_loc(val_idx[0]) - 1
        model.eval()
        for i, date in enumerate(val_idx):
            pos = train_end_pos + 1 + i
            if pos < lookback:
                continue
            window = Xs[pos - lookback : pos]
            with torch.no_grad():
                yhat_norm = float(model(torch.from_numpy(window[None, :, :])).item())
            rows.append({"date": date, "predicted": yhat_norm * r_std + r_mean})
        return pd.DataFrame(rows).set_index("date")["predicted"]
    raise ValueError(f"Unknown refiner kind: {refiner_kind}")
