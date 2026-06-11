"""Task 2 forecast helper — uniform interface across specialties + aliases.

Handles the daily vs weekly resolution split automatically:
  - Daily specialties accept horizons: 1d / 7d / monthly / yearly
  - Weekly specialties (Maternity, Psychiatry) accept: 1week / 4weeks / yearly

Usage:
    from task2_specialties.inference.load import load_model
    from task2_specialties.inference.forecast import forecast

    bundle = load_model("Medicine", "Stat 1")
    result = forecast(bundle, horizon="7d", start_date="2026-06-12")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

DAILY_HORIZONS = {"1d": 1, "7d": 7, "monthly": 30, "yearly": 365}
WEEKLY_HORIZONS = {"1week": 1, "4weeks": 4, "yearly": 52}


def forecast(
    bundle: dict[str, Any],
    horizon: str,
    start_date: str | pd.Timestamp,
    exog_future: pd.DataFrame | None = None,
    feature_future: pd.DataFrame | None = None,
) -> dict:
    """Produce a Task 2 forecast.

    Parameters
    ----------
    bundle : dict from `load_model(specialty, alias)`.
    horizon : str
        - For daily specialties: "1d" | "7d" | "monthly" | "yearly"
        - For weekly specialties: "1week" | "4weeks" | "yearly"
    start_date : first forecast date.
    exog_future : Required for Stat 2 (only used by Maternity / Psychiatry).
    feature_future : Required for ML 1 / ML 2 — the specialty's task2 exogenous matrix.

    Returns
    -------
    {
      "alias": str,
      "specialty": str,
      "resolution": "daily" or "weekly",
      "horizon": str,
      "forecast_dates": [...],
      "point_forecasts": [...],
      "aggregated_horizon_total": float or None,
      "badge": str,
      "warning": str or None,
    }
    """
    resolution = bundle["resolution"]
    horizons = WEEKLY_HORIZONS if resolution == "weekly" else DAILY_HORIZONS
    if horizon not in horizons:
        raise ValueError(
            f"Bad horizon {horizon!r} for {resolution} resolution. "
            f"Use one of: {list(horizons)}"
        )
    n = horizons[horizon]

    fitted = bundle["fitted"]
    card = bundle["card"]
    alias = bundle["alias"]
    specialty = bundle["specialty"]
    badge_id = card["badge"]

    start_ts = pd.Timestamp(start_date)
    if resolution == "weekly":
        dates = pd.date_range(start_ts, periods=n, freq="W-MON")
    else:
        dates = pd.date_range(start_ts, periods=n, freq="D")

    # Dispatch
    family = card["family"]
    if family == "Statistical":
        yhat = _predict_statistical(fitted, alias, n, exog_future)
    elif family == "ML":
        yhat = _predict_ml(fitted, alias, n, feature_future)
    else:
        raise RuntimeError(f"Unknown family: {family!r}")

    yhat = np.asarray(yhat, dtype=float).ravel()[:n]
    # Counts cannot be negative; clip at 0
    yhat = np.maximum(yhat, 0.0)
    point_forecasts = [round(float(y), 2) for y in yhat]
    forecast_dates = [d.date().isoformat() for d in dates]

    aggregate = None
    if horizon in ("monthly", "yearly", "4weeks"):
        aggregate = round(float(yhat.sum()), 1)

    warning = None
    if badge_id == "research":
        warning = ("Research preview only — do NOT base operational "
                   "decisions on this model.")
    elif badge_id == "planning":
        warning = ("Suitable for week-ahead / monthly planning. "
                   "Not for daily staffing decisions.")

    return {
        "alias": alias,
        "specialty": specialty,
        "resolution": resolution,
        "horizon": horizon,
        "forecast_dates": forecast_dates,
        "point_forecasts": point_forecasts,
        "aggregated_horizon_total": aggregate,
        "badge": badge_id,
        "warning": warning,
    }


def _predict_statistical(fitted, alias, n, exog_future):
    """ARIMA (Stat 1) or SARIMAX-weekly (Stat 2)."""
    inner = fitted.get("fitted") if isinstance(fitted, dict) else fitted
    if inner is None:
        raise RuntimeError(f"{alias} bundle missing 'fitted' model")
    if alias == "Stat 2":
        if exog_future is None:
            return inner.predict(n_periods=n)
        return inner.predict(n_periods=n, X=exog_future.values[:n])
    return inner.predict(n_periods=n)


def _predict_ml(fitted, alias, n, feature_future):
    """ML 1 (XGBoost) or ML 2 (ANN)."""
    if feature_future is None:
        raise ValueError(f"{alias} requires feature_future")
    X = feature_future.values[:n]

    if alias == "ML 1":
        inner = fitted.get("fitted") if isinstance(fitted, dict) else fitted
        return inner.predict(X)

    # ML 2 — ANN; rebuild torch net from state_dict
    import torch
    import torch.nn as nn
    arch = fitted["model_arch"]
    state = {k: torch.from_numpy(v) for k, v in fitted["model_state_dict"].items()}
    layers = []; last = arch["in_dim"]
    for _ in range(arch["hidden_layers"]):
        layers += [nn.Linear(last, arch["units"]), nn.ReLU(),
                   nn.Dropout(arch["dropout"])]
        last = arch["units"]
    layers.append(nn.Linear(last, 1))
    net = nn.Sequential(*layers)
    net.load_state_dict(state)
    net.eval()

    # Apply feature scaler
    fs = fitted.get("feature_scaler")
    if fs is not None:
        mean = np.array([fs["mean"][c] for c in fitted["feature_names"]])
        std = np.array([fs["std"][c] for c in fitted["feature_names"]])
        Xs = (X - mean) / np.maximum(std, 1e-9)
    else:
        Xs = X
    with torch.no_grad():
        yn = net(torch.from_numpy(Xs.astype(np.float32))).squeeze(-1).numpy()

    ts = fitted.get("target_scaler")
    if ts:
        return yn * ts["std"] + ts["mean"]
    return yn
