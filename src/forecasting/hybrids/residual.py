"""Residual hybrids per Zhang (2003), plan §12.1.

  y_hat = f_A(x) + f_B(residuals_of_A_on_training_fold)

.. deprecated:: Prompt 4 of the Ch6 refactor
    This module uses **in-sample** training residuals from a single fit on
    the whole train fold. That recipe is subject to the selection bias
    documented by Khashei & Bijari (2011, ASOC 11(2):2664-2675) and
    methodologically critiqued by Hewamalage, Bergmeir & Bandara (2021,
    IJF 37(1):388-427) -- the refiner under-states the generalisation
    residual variance and overfits.

    Use ``src.forecasting.hybrids.oof.oof_residuals.OOFResidualHybrid``
    instead. It builds **out-of-fold** residuals via the shared
    ``RollingForecaster``, applies consistent residual standardisation,
    and supports nested Optuna HPO over the refiner -- removing the
    HPO-inheritance bug documented in RESULTS.md §4sexies.

    The functions below remain importable for backward compatibility with
    ``scripts/09_hybrids.py`` and ``scripts/11_lstm_xgb_hybrid.py`` until
    those scripts are migrated.

Implemented as a generic residual-refinement wrapper that consumes:
  - already-saved base predictions on val (artefacts/predictions/{base}.csv)
  - in-sample training residuals from a fresh base fit on the train fold
  - a refiner (XGBoost or LSTM) trained on (X_train, residuals_train)

Three concrete hybrids:
  - SARIMAX + XGBoost
  - SARIMAX + LSTM
  - LSTM    + XGBoost

Critical leakage check: the refiner sees only training-fold residuals; val and
test residuals are never used in fitting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def _load_val_predictions(base_name: str) -> pd.Series:
    """Load the val predictions for a base model that's already been run."""
    path = ROOT / "artefacts" / "predictions" / f"{base_name}.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["predicted"]


# ---------------------------------------------------------------------------
# SARIMAX base helpers
# ---------------------------------------------------------------------------

def sarimax_train_residuals(
    target: pd.Series, X_train: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    warmup_days: int = 30,
) -> pd.Series:
    """Refit SARIMAX once on the training fold and return in-sample residuals.

    The first ``warmup_days`` residuals are dropped because differencing
    initialisation makes them unstable (SD can exceed 100 on series with
    mean ~60 — that polluted the refiner training set in the first build).
    """
    from pmdarima import ARIMA as PmARIMA
    train_idx = X_train.index
    y_train = target.loc[train_idx]
    model = PmARIMA(order=order, seasonal_order=seasonal_order,
                    suppress_warnings=True)
    model.fit(y_train.values, X=X_train.values)
    fitted = model.predict_in_sample(X=X_train.values)
    resid = y_train.values - np.asarray(fitted)
    resid_series = pd.Series(resid, index=train_idx, name="residual")
    if warmup_days > 0:
        resid_series = resid_series.iloc[warmup_days:]
    # Defensive: also drop any |resid| > 5 * sigma observed in the rest
    sigma = resid_series.std()
    resid_series = resid_series[resid_series.abs() <= 5 * sigma]
    return resid_series


# ---------------------------------------------------------------------------
# LSTM base helpers
# ---------------------------------------------------------------------------

def lstm_train_in_sample(
    target: pd.Series, X_train: pd.DataFrame,
    params: dict, seed: int = 42,
) -> pd.Series:
    """Fit an LSTM on the training fold and return in-sample one-step predictions."""
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything,
    )
    _seed_everything(seed)

    y_train = target.loc[X_train.index]
    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values

    lookback = params["lookback"]
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
    n_es = max(28, lookback + 7)
    X_es_seq, y_es_seq = Xtr_seq[-n_es:], ytr_seq[-n_es:]
    Xtr_seq_fit, ytr_seq_fit = Xtr_seq[:-n_es], ytr_seq[:-n_es]
    model, _ = _train_one(Xtr_seq_fit, ytr_seq_fit, X_es_seq, y_es_seq,
                           params, max_epochs=50)

    model.eval()
    with torch.no_grad():
        yhat_norm = model(torch.from_numpy(Xtr_seq)).numpy()
    yhat = yhat_norm * y_std + y_mean
    # Predictions correspond to indices lookback..len(target)-1
    idx = X_train.index[lookback:]
    return pd.Series(yhat, index=idx, name="lstm_in_sample"), (model, mean, std, y_mean, y_std)


def lstm_predict_val(
    fit_artifacts, X_full: pd.DataFrame, val_idx: pd.DatetimeIndex,
    lookback: int,
) -> pd.Series:
    """Predict val days using a pre-fit LSTM, sliding window over X_full."""
    import torch
    model, mean, std, y_mean, y_std = fit_artifacts
    Xs = ((X_full - mean) / std).astype(np.float32).values
    rows = []
    train_end_pos = X_full.index.get_loc(val_idx[0]) - 1
    for i, date in enumerate(val_idx):
        pos = train_end_pos + 1 + i
        if pos < lookback:
            continue
        window = Xs[pos - lookback : pos]
        window_t = torch.from_numpy(window[None, :, :])
        model.eval()
        with torch.no_grad():
            yhat_norm = float(model(window_t).item())
        rows.append({"date": date, "predicted": yhat_norm * y_std + y_mean})
    return pd.DataFrame(rows).set_index("date")["predicted"]


# ---------------------------------------------------------------------------
# Refiner: XGBoost on residuals
# ---------------------------------------------------------------------------

def fit_xgb_refiner(
    X_train_aligned: pd.DataFrame, residuals: pd.Series,
    seed: int = 42,
):
    """Fit XGBoost on (X, residuals) with light defaults (refiner is small)."""
    from xgboost import XGBRegressor
    common_idx = X_train_aligned.index.intersection(residuals.index)
    X = X_train_aligned.loc[common_idx].values
    r = residuals.loc[common_idx].values
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
        objective="reg:squarederror", random_state=seed, verbosity=0, n_jobs=-1,
    )
    model.fit(X, r)
    return model


def fit_lstm_refiner(
    X_train_aligned: pd.DataFrame, residuals: pd.Series,
    seed: int = 42,
):
    """Fit a small LSTM on (X, residuals). Lookback fixed at 14 for refiner."""
    import torch
    from src.forecasting.models.lstm import (
        _build_sequences, _train_one, _seed_everything, _LSTMNet,
    )
    _seed_everything(seed)
    common_idx = X_train_aligned.index.intersection(residuals.index)
    X = X_train_aligned.loc[common_idx]
    r = residuals.loc[common_idx]
    mean = X.mean()
    std = X.std(ddof=0).replace(0, 1.0)
    Xs = ((X - mean) / std).astype(np.float32).values
    r_mean, r_std = float(r.mean()), float(r.std(ddof=0)) if r.std(ddof=0) > 0 else 1.0
    rs = ((r - r_mean) / r_std).astype(np.float32).values
    lookback = 14
    Xs_seq, rs_seq = _build_sequences(Xs, rs, lookback)
    n_es = 28
    Xes, res_ = Xs_seq[-n_es:], rs_seq[-n_es:]
    Xfit, rfit = Xs_seq[:-n_es], rs_seq[:-n_es]
    params = {"units": 64, "dropout": 0.2, "learning_rate": 0.001,
              "batch_size": 32, "seed": seed}
    model, _ = _train_one(Xfit, rfit, Xes, res_, params, max_epochs=40)
    return model, mean, std, r_mean, r_std, lookback


# ---------------------------------------------------------------------------
# Refiner predict on val
# ---------------------------------------------------------------------------

def xgb_refiner_predict(model, X_val: pd.DataFrame) -> pd.Series:
    yhat = model.predict(X_val.values)
    return pd.Series(yhat, index=X_val.index, name="refiner_pred")


def lstm_refiner_predict(fit_artifacts, X_full: pd.DataFrame,
                          val_idx: pd.DatetimeIndex) -> pd.Series:
    import torch
    model, mean, std, r_mean, r_std, lookback = fit_artifacts
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
