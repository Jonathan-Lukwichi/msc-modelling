"""XGBoost standalone trainer (plan §11.2).

Fits on the §3.4.3 consensus-selected feature set (23 features from
src.forecasting.consensus). Grid HPO over §3.5.9 Table 3.1 ranges. Selection
by val MAPE. Rolling-origin weekly refit on val with the winning params.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass
class XgbBest:
    params: dict
    val_mape: float
    val_mae: float
    val_rmse: float
    val_r2: float


GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.7, 0.85, 1.0],
}


def _from_metrics(y_actual, y_pred):
    from src.forecasting.metrics import mape, mae, rmse, r2
    return {"MAPE": mape(y_actual, y_pred), "MAE": mae(y_actual, y_pred),
            "RMSE": rmse(y_actual, y_pred), "R2": r2(y_actual, y_pred)}


def grid_search_cv(
    X_train: pd.DataFrame, y_train: pd.Series,
    grid: dict | None = None,
    n_folds: int = 8,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[XgbBest, pd.DataFrame]:
    """Grid HPO via rolling-origin inner CV inside the train block (§3.6.1).

    For each hyperparam combo:
      - Run rolling-origin folds (8 evenly spaced from the 69 available)
      - Average MAPE/MAE/RMSE/R^2 across folds
      - Rank by mean CV MAPE

    No val data is consumed in HPO. The val block is reserved for a single
    held-out check on the chosen architecture after refit on full train.
    """
    from src.forecasting.cv import subsampled_rolling_origin, evaluate_with_cv
    grid = grid or GRID
    folds = subsampled_rolling_origin(X_train.index, n_folds=n_folds,
                                        horizon_days=7, step_days=7,
                                        min_train_days=365)
    if not folds:
        raise ValueError("Not enough train data for rolling-origin CV")

    rows = []
    keys = list(grid.keys())
    best = None
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))

        def fit_predict(X_tr, y_tr, X_te, params=params, seed=seed):
            m = XGBRegressor(**params, objective="reg:squarederror",
                              random_state=seed, verbosity=0, n_jobs=-1)
            m.fit(X_tr.values, y_tr.values)
            return m.predict(X_te.values)

        result = evaluate_with_cv(folds, X_train, y_train, fit_predict,
                                    _from_metrics)
        mean_scores = result["mean"]
        row = {**params, **{f"cv_{k}": v for k, v in mean_scores.items()},
                "cv_std_MAPE": result["std"]["MAPE"]}
        rows.append(row)
        if best is None or mean_scores["MAPE"] < best["MAPE"]:
            best = {**params, **mean_scores}
        if verbose:
            print(f"  {params}: cv_MAPE={mean_scores['MAPE']:.3f} "
                  f"+/- {result['std']['MAPE']:.3f}")
    trace = pd.DataFrame(rows).sort_values("cv_MAPE")
    return XgbBest(
        params={k: best[k] for k in keys},
        val_mape=best["MAPE"], val_mae=best["MAE"],
        val_rmse=best["RMSE"], val_r2=best["R2"],
    ), trace


# Legacy single-split selector kept for backward compatibility; do NOT use for HPO
def grid_search(X_train, y_train, X_val, y_val, grid=None, seed=42, verbose=False):
    """DEPRECATED: HPO on val. Kept only for legacy callers; use grid_search_cv()."""
    grid = grid or GRID
    rows = []
    keys = list(grid.keys())
    best = None
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        model = XGBRegressor(**params, objective="reg:squarederror",
                              random_state=seed, verbosity=0, n_jobs=-1)
        model.fit(X_train.values, y_train.values)
        yhat = model.predict(X_val.values)
        scores = _from_metrics(y_val.values, yhat)
        rows.append({**params, **scores})
        if best is None or scores["MAPE"] < best["MAPE"]:
            best = {**params, **scores}
    trace = pd.DataFrame(rows).sort_values("MAPE")
    return XgbBest(
        params={k: best[k] for k in keys},
        val_mape=best["MAPE"], val_mae=best["MAE"],
        val_rmse=best["RMSE"], val_r2=best["R2"],
    ), trace


def rolling_forecast(
    X_full: pd.DataFrame, y_full: pd.Series,
    block_index: pd.DatetimeIndex,
    params: dict,
    step_days: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Rolling-origin weekly refit — thin wrapper over RollingForecaster."""
    from src.forecasting.rolling import RollingForecaster, make_xgboost_factory

    rf = RollingForecaster(
        model_factory=make_xgboost_factory(params, seed=seed),
        step_days=step_days, horizon_days=step_days, min_train_days=1,
    )
    out = rf.fit_predict(X=X_full, y=y_full, eval_index=block_index)
    return out.reset_index().rename(columns={"yhat": "predicted"})[["date", "predicted"]]


def shap_summary(X_train: pd.DataFrame, y_train: pd.Series, params: dict,
                  seed: int = 42) -> pd.DataFrame:
    """Mean |SHAP| per feature on the training fold (for Figure 6.6)."""
    import shap
    model = XGBRegressor(
        **params, objective="reg:squarederror",
        random_state=seed, verbosity=0, n_jobs=-1,
    )
    model.fit(X_train.values, y_train.values)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train.values)
    importance = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
