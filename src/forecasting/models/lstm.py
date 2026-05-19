"""LSTM standalone trainer (plan §11.4) — PyTorch implementation.

Optuna TPE 30 trials per §3.5.9, capped at 60-minute time budget.
Sequence input: target series + selected exogenous features, lookback in
{14, 21, 28}. MedianPruner. Early stopping on val MAPE.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn, optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, units: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=units,
                             num_layers=1, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


def _build_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Sliding-window dataset. X is (T, F), y is (T,). Returns (N, lookback, F), (N,)."""
    N = len(y) - lookback
    Xs = np.empty((N, lookback, X.shape[1]), dtype=np.float32)
    ys = np.empty(N, dtype=np.float32)
    for i in range(N):
        Xs[i] = X[i : i + lookback]
        ys[i] = y[i + lookback]
    return Xs, ys


@dataclass
class LstmBest:
    params: dict
    val_mape: float
    val_mae: float
    val_rmse: float
    val_r2: float


def _train_one(
    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
    params: dict, max_epochs: int = 80,
):
    from src.forecasting.metrics import score
    _seed_everything(params.get("seed", 42))
    model = _LSTMNet(n_features=X_train_seq.shape[2], units=params["units"],
                      dropout=params["dropout"])
    opt = optim.Adam(model.parameters(), lr=params["learning_rate"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5,
                                                  factor=0.5)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train_seq)
    yt = torch.from_numpy(y_train_seq)
    Xv = torch.from_numpy(X_val_seq)
    yv_np = y_val_seq

    bs = params["batch_size"]
    n = Xt.shape[0]
    best_state = None
    best_mape = float("inf")
    bad = 0
    patience = 8
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            yhat = model(Xv).numpy()
        m = score(yv_np, yhat)
        sched.step(m["MAPE"])
        if m["MAPE"] < best_mape:
            best_mape = m["MAPE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yhat = model(Xv).numpy()
    return model, score(yv_np, yhat)


def tpe_search_cv(
    X_train: pd.DataFrame, y_train: pd.Series,
    n_trials: int = 15, n_folds: int = 3,
    time_budget_minutes: int = 60, seed: int = 42,
) -> tuple[LstmBest, pd.DataFrame]:
    """Optuna TPE via inner rolling-origin CV (§3.6.1).

    Trial count and fold count reduced from the plan defaults (30 / 8) to keep
    the total time budget tractable: 15 trials x 3 folds x ~2 min/fit = ~90 min.
    The trade-off is acknowledged in RESULTS.md.

    No val data consumed.
    """
    from src.forecasting.cv import subsampled_rolling_origin
    from src.forecasting.metrics import score
    _seed_everything(seed)
    folds = subsampled_rolling_origin(X_train.index, n_folds=n_folds,
                                        horizon_days=7, step_days=7,
                                        min_train_days=365)
    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lookback": trial.suggest_categorical("lookback", [14, 21, 28]),
            "units": trial.suggest_categorical("units", [64, 96, 128, 192, 256]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4]),
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "seed": seed,
        }
        fold_mapes = []
        for fold in folds:
            X_tr_fold = X_train.iloc[fold.train_idx]
            y_tr_fold = y_train.iloc[fold.train_idx]
            X_te_fold = X_train.iloc[fold.test_idx]
            y_te_fold = y_train.iloc[fold.test_idx]
            mean = X_tr_fold.mean()
            std = X_tr_fold.std(ddof=0).replace(0, 1.0)
            Xtr = ((X_tr_fold - mean) / std).astype(np.float32).values
            Xte = ((X_te_fold - mean) / std).astype(np.float32).values
            y_mean, y_std = float(y_tr_fold.mean()), float(y_tr_fold.std(ddof=0))
            ytr_norm = ((y_tr_fold - y_mean) / y_std).astype(np.float32).values

            lookback = params["lookback"]
            Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr_norm, lookback)
            n_es = max(28, lookback + 7)
            Xfit_seq, yfit_seq = Xtr_seq[:-n_es], ytr_seq[:-n_es]
            Xes_seq, yes_seq = Xtr_seq[-n_es:], ytr_seq[-n_es:]
            model, _ = _train_one(Xfit_seq, yfit_seq, Xes_seq, yes_seq,
                                    params, max_epochs=40)
            # Predict the 7 test days with sliding window: last `lookback` of Xtr + slide
            full_X = np.vstack([Xtr, Xte])
            preds = []
            model.eval()
            with torch.no_grad():
                for i in range(len(Xte)):
                    pos = len(Xtr) + i
                    window = full_X[pos - lookback : pos]
                    preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
            yhat = np.array(preds) * y_std + y_mean
            fold_mapes.append(score(y_te_fold.values, yhat)["MAPE"])
        cv_mape = float(np.mean(fold_mapes))
        trial_rows.append({"trial": trial.number, **params, "cv_MAPE": cv_mape,
                           "cv_std_MAPE": float(np.std(fold_mapes))})
        return cv_mape

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials,
                    timeout=time_budget_minutes * 60,
                    show_progress_bar=False)

    best_params = study.best_params
    best_params["seed"] = seed
    trace = pd.DataFrame(trial_rows).sort_values("cv_MAPE")
    best_cv_mape = float(study.best_value)
    return LstmBest(
        params=best_params,
        val_mape=best_cv_mape, val_mae=float("nan"),
        val_rmse=float("nan"), val_r2=float("nan"),
    ), trace


def tpe_search(X_train, y_train, X_val, y_val,
                n_trials=30, time_budget_minutes=60, seed=42):
    """DEPRECATED: HPO on val. Kept for legacy callers; use tpe_search_cv()."""
    _seed_everything(seed)
    # Standardise on train
    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    Xva = ((X_val - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values
    yva = ((y_val - y_mean) / y_std).astype(np.float32).values

    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lookback": trial.suggest_categorical("lookback", [14, 21, 28]),
            "units": trial.suggest_categorical("units", [64, 96, 128, 192, 256]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4]),
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "seed": seed,
        }
        # Build sequences using params['lookback']
        # Concatenate train + (lookback of train tail) for val sequence anchoring
        warmup = Xtr[-params["lookback"]:]
        X_val_padded = np.vstack([warmup, Xva])
        y_val_padded = np.concatenate([ytr[-params["lookback"]:], yva])
        Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, params["lookback"])
        Xva_seq, yva_seq = _build_sequences(X_val_padded, y_val_padded, params["lookback"])
        # yva_seq is the last len(yva) entries
        Xva_seq = Xva_seq[-len(yva):]
        yva_seq = yva_seq[-len(yva):]
        _, m = _train_one(Xtr_seq, ytr_seq, Xva_seq, yva_seq, params, max_epochs=60)
        trial_rows.append({"trial": trial.number, **params, **m})
        return m["MAPE"]

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials,
                    timeout=time_budget_minutes * 60,
                    show_progress_bar=False)

    best_params = study.best_params
    best_params["seed"] = seed
    trace = pd.DataFrame(trial_rows).sort_values("MAPE")

    # Re-train best to get original-scale metrics
    warmup = Xtr[-best_params["lookback"]:]
    X_val_padded = np.vstack([warmup, Xva])
    y_val_padded = np.concatenate([ytr[-best_params["lookback"]:], yva])
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, best_params["lookback"])
    Xva_seq, yva_seq = _build_sequences(X_val_padded, y_val_padded, best_params["lookback"])
    Xva_seq = Xva_seq[-len(yva):]
    yva_seq = yva_seq[-len(yva):]
    model, _ = _train_one(Xtr_seq, ytr_seq, Xva_seq, yva_seq, best_params)
    model.eval()
    with torch.no_grad():
        yhat_norm = model(torch.from_numpy(Xva_seq)).numpy()
    yhat = yhat_norm * y_std + y_mean
    from src.forecasting.metrics import score
    orig = score(y_val.values, yhat)
    return LstmBest(
        params=best_params,
        val_mape=orig["MAPE"], val_mae=orig["MAE"],
        val_rmse=orig["RMSE"], val_r2=orig["R2"],
    ), trace


def rolling_forecast(
    X_full: pd.DataFrame, y_full: pd.Series,
    block_index: pd.DatetimeIndex,
    params: dict,
    step_days: int = 7, seed: int = 42,
) -> pd.DataFrame:
    """Rolling weekly refit on the LSTM."""
    rows = []
    block_start = block_index[0]
    block_end = block_index[-1]
    lookback = params["lookback"]

    origin_pos = y_full.index.get_loc(block_start) - 1
    while origin_pos < y_full.index.get_loc(block_end):
        n_remaining = y_full.index.get_loc(block_end) - origin_pos
        h = int(min(step_days, n_remaining))
        X_train = X_full.iloc[: origin_pos + 1]
        y_train = y_full.iloc[: origin_pos + 1]

        mean = X_train.mean()
        std = X_train.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_train - mean) / std).astype(np.float32).values
        y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
        ytr = ((y_train - y_mean) / y_std).astype(np.float32).values

        # Build train sequences
        Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
        # Hold last 28 train points for early stopping
        n_es = max(28, lookback + 7)
        X_es_seq, y_es_seq = Xtr_seq[-n_es:], ytr_seq[-n_es:]
        Xtr_seq = Xtr_seq[:-n_es]
        ytr_seq = ytr_seq[:-n_es]

        _seed_everything(seed)
        model, _ = _train_one(Xtr_seq, ytr_seq, X_es_seq, y_es_seq, params,
                               max_epochs=50)

        # Predict h days ahead step by step using sliding window of observed history
        # For each future day, use lookback of (train tail + already-predicted future).
        # We predict h <= 7 days using observed exog (no recursion needed for X);
        # for y we use observed history (lag-7 is in the engineered features, but
        # the LSTM uses the whole window, so we rely on observed exog).
        X_future = X_full.iloc[origin_pos + 1 : origin_pos + 1 + h]
        Xfu = ((X_future - mean) / std).astype(np.float32).values

        # Sliding-window prediction: at each step the window is last `lookback`
        # rows of (train + already-predicted future) for X; we use the most
        # recent observed exog. The target prediction at each step is independent
        # given the window.
        full_X_seq = np.vstack([Xtr, Xfu])
        preds_norm = []
        model.eval()
        with torch.no_grad():
            for i in range(h):
                window_start = len(Xtr) + i - lookback
                if window_start < 0:
                    # not enough history; pad with first row
                    pad = np.tile(full_X_seq[0:1], (-window_start, 1))
                    window = np.vstack([pad, full_X_seq[: len(Xtr) + i]])
                else:
                    window = full_X_seq[window_start : len(Xtr) + i]
                window_t = torch.from_numpy(window[None, :, :])
                preds_norm.append(float(model(window_t).item()))
        yhat = np.array(preds_norm) * y_std + y_mean
        for date, y_pred in zip(X_future.index, yhat):
            rows.append({"date": date, "predicted": float(y_pred)})
        origin_pos += step_days

    return pd.DataFrame(rows)
