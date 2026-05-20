"""ANN (MLP) standalone trainer (plan §11.3) — PyTorch implementation.

Random search HPO over §3.5.9 Table 3.1 ranges (20 iter). Early stopping on
validation MAPE, ReduceLROnPlateau. Seed 42 for numpy + torch + random;
note that PyTorch CPU is not bit-exactly reproducible across runs even with
seeds, but it is much closer than TensorFlow on Windows CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, optim


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _MLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in n_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class AnnBest:
    params: dict
    val_mape: float
    val_mae: float
    val_rmse: float
    val_r2: float


def _train_one(
    X_train_t, y_train_t, X_val_t, y_val_t,
    params: dict, max_epochs: int = 200,
) -> tuple[_MLP, dict, list[dict]]:
    """Train one configuration, return best-weights model + final val metrics."""
    from src.forecasting.metrics import score
    _seed_everything(params.get("seed", 42))

    n_hidden = [params["units"]] * params["hidden_layers"]
    model = _MLP(n_in=X_train_t.shape[1], n_hidden=n_hidden,
                  dropout=params["dropout"])
    opt = optim.Adam(model.parameters(), lr=params["learning_rate"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5,
                                                  factor=0.5)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_mape = float("inf")
    patience = 10
    bad = 0
    history = []

    bs = params["batch_size"]
    n = X_train_t.shape[0]
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            yhat_val = model(X_val_t).numpy()
        val_metrics = score(y_val_t.numpy(), yhat_val)
        history.append({"epoch": epoch, "train_mse": epoch_loss, **val_metrics})
        sched.step(val_metrics["MAPE"])

        if val_metrics["MAPE"] < best_val_mape:
            best_val_mape = val_metrics["MAPE"]
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
        yhat_val_final = model(X_val_t).numpy()
    final_metrics = score(y_val_t.numpy(), yhat_val_final)
    return model, final_metrics, history


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "hidden_layers": int(rng.choice([1, 2])),
        "units": int(rng.choice([64, 128, 192, 256])),
        "dropout": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        "learning_rate": float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
        "batch_size": int(rng.choice([16, 32, 64])),
        "seed": 42,
    }


def random_search_cv(
    X_train: pd.DataFrame, y_train: pd.Series,
    n_iter: int = 20, n_folds: int = 5, seed: int = 42,
) -> tuple[AnnBest, pd.DataFrame]:
    """20-iter random search via inner rolling-origin CV inside train (§3.6.1).

    No val data consumed. Each trial is evaluated by the mean MAPE across
    n_folds evenly-spaced rolling-origin folds.
    """
    from src.forecasting.cv import subsampled_rolling_origin
    from src.forecasting.metrics import score
    _seed_everything(seed)
    folds = subsampled_rolling_origin(X_train.index, n_folds=n_folds,
                                        horizon_days=7, step_days=7,
                                        min_train_days=365)
    rng = np.random.default_rng(seed)
    rows = []
    best = None
    for i in range(n_iter):
        params = _sample_params(rng)
        fold_scores = []
        for fold in folds:
            X_tr_fold = X_train.iloc[fold.train_idx]
            y_tr_fold = y_train.iloc[fold.train_idx]
            X_te_fold = X_train.iloc[fold.test_idx]
            y_te_fold = y_train.iloc[fold.test_idx]
            # Hold-out for early-stopping inside the fold: last 28 of train
            n_es = max(28, len(X_tr_fold) // 6)
            mean = X_tr_fold.mean()
            std = X_tr_fold.std(ddof=0).replace(0, 1.0)
            Xtr = ((X_tr_fold - mean) / std).astype(np.float32)
            Xes = Xtr.iloc[-n_es:]
            Xtr = Xtr.iloc[:-n_es]
            Xte = ((X_te_fold - mean) / std).astype(np.float32)
            y_mean, y_std = float(y_tr_fold.mean()), float(y_tr_fold.std(ddof=0))
            ytr_norm = ((y_tr_fold - y_mean) / y_std).astype(np.float32)
            yes_norm = ytr_norm.iloc[-n_es:]
            ytr_norm = ytr_norm.iloc[:-n_es]

            X_train_t = torch.from_numpy(Xtr.values)
            y_train_t = torch.from_numpy(ytr_norm.values)
            X_es_t = torch.from_numpy(Xes.values)
            y_es_t = torch.from_numpy(yes_norm.values)
            X_te_t = torch.from_numpy(Xte.values)

            model, _, _ = _train_one(X_train_t, y_train_t, X_es_t, y_es_t,
                                       params, max_epochs=80)
            model.eval()
            with torch.no_grad():
                yhat_norm = model(X_te_t).numpy()
            yhat = yhat_norm * y_std + y_mean
            fold_scores.append(score(y_te_fold.values, yhat))

        df_fold = pd.DataFrame(fold_scores)
        mean_scores = {f"cv_{c}": float(df_fold[c].mean()) for c in df_fold.columns}
        row = {"trial": i, **params, **mean_scores,
                "cv_std_MAPE": float(df_fold["MAPE"].std())}
        rows.append(row)
        if best is None or mean_scores["cv_MAPE"] < best["cv_MAPE"]:
            best = {**params, **mean_scores}

    trace = pd.DataFrame(rows).sort_values("cv_MAPE")
    best_params = {k: best[k] for k in ["hidden_layers", "units", "dropout",
                                          "learning_rate", "batch_size", "seed"]}
    return AnnBest(
        params=best_params,
        val_mape=best["cv_MAPE"], val_mae=best["cv_MAE"],
        val_rmse=best["cv_RMSE"], val_r2=best["cv_R2"],
    ), trace


def random_search(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    n_iter: int = 20, seed: int = 42,
) -> tuple[AnnBest, pd.DataFrame]:
    """DEPRECATED: HPO on val. Kept for legacy callers; use random_search_cv()."""
    _seed_everything(seed)
    # Standardise X on train (ANN is scale-sensitive)
    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32)
    Xva = ((X_val - mean) / std).astype(np.float32)
    # Also standardise y (helps optimisation)
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32)
    yva = ((y_val - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.from_numpy(Xtr.values)
    y_train_t = torch.from_numpy(ytr.values)
    X_val_t = torch.from_numpy(Xva.values)
    y_val_t_norm = torch.from_numpy(yva.values)

    rng = np.random.default_rng(seed)
    rows = []
    best = None
    for i in range(n_iter):
        params = _sample_params(rng)
        _, metrics_norm, _ = _train_one(X_train_t, y_train_t,
                                         X_val_t, y_val_t_norm, params)
        # Recompute metrics in original scale by re-predicting val
        # (lazy: store both for now)
        row = {"trial": i, **params, **metrics_norm}
        rows.append(row)
        if best is None or metrics_norm["MAPE"] < best["MAPE"]:
            best = {**params, **metrics_norm}

    trace = pd.DataFrame(rows).sort_values("MAPE")
    # Re-run best to get original-scale metrics
    best_params = {k: best[k] for k in ["hidden_layers", "units", "dropout",
                                          "learning_rate", "batch_size", "seed"]}
    model, _, _ = _train_one(X_train_t, y_train_t, X_val_t, y_val_t_norm,
                              best_params)
    model.eval()
    with torch.no_grad():
        yhat_val_norm = model(X_val_t).numpy()
    yhat_val = yhat_val_norm * y_std + y_mean
    from src.forecasting.metrics import score
    orig_metrics = score(y_val.values, yhat_val)

    return AnnBest(
        params=best_params,
        val_mape=orig_metrics["MAPE"], val_mae=orig_metrics["MAE"],
        val_rmse=orig_metrics["RMSE"], val_r2=orig_metrics["R2"],
    ), trace


def rolling_forecast(
    X_full: pd.DataFrame, y_full: pd.Series,
    block_index: pd.DatetimeIndex,
    params: dict,
    step_days: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Rolling weekly refit — thin wrapper over RollingForecaster."""
    from src.forecasting.rolling import RollingForecaster, FoldPrediction

    def factory(X_train, y_train, sample_weight=None):
        mean = X_train.mean(); std = X_train.std(ddof=0).replace(0, 1.0)
        Xtr = ((X_train - mean) / std).astype(np.float32)
        y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
        ytr = ((y_train - y_mean) / y_std).astype(np.float32)
        X_train_t = torch.from_numpy(Xtr.values)
        y_train_t = torch.from_numpy(ytr.values)
        n_es = min(28, len(X_train_t) // 6)
        X_es_t, y_es_t = X_train_t[-n_es:], y_train_t[-n_es:]
        X_train_t, y_train_t = X_train_t[:-n_es], y_train_t[:-n_es]
        _seed_everything(seed)
        model, _, _ = _train_one(
            X_train_t, y_train_t, X_es_t, y_es_t,
            {**params, "seed": seed}, max_epochs=120,
        )
        model.eval()

        class _Fitted:
            def predict(self, X_future, h):
                Xfu = ((X_future - mean) / std).astype(np.float32)
                with torch.no_grad():
                    yhat_norm = model(torch.from_numpy(Xfu.values)).numpy()
                yhat = yhat_norm * y_std + y_mean
                return FoldPrediction(yhat=np.asarray(yhat, dtype=float))

        return _Fitted()

    rf = RollingForecaster(
        model_factory=factory,
        step_days=step_days, horizon_days=step_days, min_train_days=1,
    )
    out = rf.fit_predict(X=X_full, y=y_full, eval_index=block_index)
    return out.reset_index().rename(columns={"yhat": "predicted"})[["date", "predicted"]]
