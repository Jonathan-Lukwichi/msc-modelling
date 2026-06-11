"""Task 1 forecast helper — uniform interface across all 6 aliases.

Hides the model-family differences (ARIMA univariate vs SARIMAX with exog vs
XGBoost / ANN tabular vs hybrid) behind a single function.

Usage:
    from task1_daily_arrivals.inference.load import load_model
    from task1_daily_arrivals.inference.forecast import forecast

    bundle = load_model("Hybrid 1")
    result = forecast(bundle, horizon="7d", start_date="2026-06-12",
                      exog_future=my_exog_df,
                      feature_future=my_features_df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


HORIZON_TO_PERIODS = {"1d": 1, "7d": 7, "monthly": 30, "yearly": 365}


def forecast(
    bundle: dict[str, Any],
    horizon: str,
    start_date: str | pd.Timestamp,
    exog_future: pd.DataFrame | None = None,
    feature_future: pd.DataFrame | None = None,
) -> dict:
    """Produce a forecast for the requested horizon.

    Parameters
    ----------
    bundle : dict from `load_model()`.
    horizon : one of {"1d", "7d", "monthly", "yearly"}.
    start_date : first date of the forecast window.
    exog_future : Required for Stat 2 / Hybrid 1 / Hybrid 2 (SARIMAX-based).
                  Must be the §5.2.5 raw-10 exog matrix indexed by future
                  dates. The bundle's scaler is applied internally.
    feature_future : Required for ML 1 / ML 2 (XGBoost / ANN).
                     The engineered+consensus feature matrix indexed by
                     future dates.

    Returns
    -------
    {
      "alias": str,
      "horizon": str,
      "forecast_dates": [...ISO strings...],
      "point_forecasts": [...numbers...],
      "aggregated_horizon_total": float or None,
      "badge": str ("operational" / "planning" / "research"),
      "warning": str or None,
    }
    """
    if horizon not in HORIZON_TO_PERIODS:
        raise ValueError(f"Unknown horizon: {horizon!r}. "
                         f"Use one of {list(HORIZON_TO_PERIODS)}")
    n = HORIZON_TO_PERIODS[horizon]
    alias = bundle["alias"]
    card = bundle["card"]
    fitted = bundle["fitted"]
    badge_id = card["badge"]

    start_ts = pd.Timestamp(start_date)
    dates = pd.date_range(start_ts, periods=n, freq="D")

    # Dispatch by family
    family = card["family"]
    if family == "Statistical":
        # Stat 1 = ARIMA (univariate, no exog) or Stat 2 = SARIMAX (with exog)
        sci = card["internal_only"]["scientific_name"]
        if sci.startswith("ARIMA"):
            yhat = _predict_arima(fitted, n_periods=n)
        else:
            yhat = _predict_sarimax(fitted, n_periods=n,
                                    exog_future=exog_future)
    elif family == "ML":
        yhat = _predict_ml(fitted, alias, feature_future=feature_future,
                            n_periods=n)
    elif family == "Hybrid":
        yhat = _predict_hybrid(fitted, alias, n_periods=n,
                                exog_future=exog_future,
                                feature_future=feature_future)
    else:
        raise RuntimeError(f"Unknown family: {family!r}")

    yhat = np.asarray(yhat, dtype=float).ravel()[:n]
    point_forecasts = [round(float(y), 2) for y in yhat]
    forecast_dates = [d.date().isoformat() for d in dates]

    aggregate = None
    if horizon in ("monthly", "yearly"):
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
        "horizon": horizon,
        "forecast_dates": forecast_dates,
        "point_forecasts": point_forecasts,
        "aggregated_horizon_total": aggregate,
        "badge": badge_id,
        "warning": warning,
    }


# -----------------------------------------------------------------------
# Family-specific predict helpers
# -----------------------------------------------------------------------
def _predict_arima(fitted, n_periods):
    # pmdarima ARIMA
    try:
        return fitted.predict(n_periods=n_periods)
    except Exception:
        # Bundle may be a dict — fall back to inner model
        if isinstance(fitted, dict) and "fitted" in fitted:
            return fitted["fitted"].predict(n_periods=n_periods)
        raise


def _predict_sarimax(fitted, n_periods, exog_future):
    if exog_future is None:
        raise ValueError("Stat 2 (SARIMAX) requires exog_future")
    if hasattr(fitted, "predict"):
        return fitted.predict(n_periods=n_periods,
                              X=exog_future.values[:n_periods])
    if isinstance(fitted, dict) and "fitted" in fitted:
        return fitted["fitted"].predict(n_periods=n_periods,
                                          X=exog_future.values[:n_periods])
    raise RuntimeError("Cannot dispatch SARIMAX predict")


def _predict_ml(fitted, alias, feature_future, n_periods):
    if feature_future is None:
        raise ValueError(f"{alias} requires feature_future "
                         f"(engineered+consensus feature matrix)")
    X = feature_future.values[:n_periods]
    if hasattr(fitted, "predict"):
        return fitted.predict(X)
    if isinstance(fitted, dict):
        # bundle may contain {'fitted': model, 'feature_names': [...]}
        inner = fitted.get("fitted") or fitted.get("model")
        if inner is not None and hasattr(inner, "predict"):
            return inner.predict(X)
    raise RuntimeError(f"Cannot dispatch ML predict for {alias}")


def _predict_hybrid(fitted, alias, n_periods, exog_future, feature_future):
    # Hybrid bundle conventionally contains {'base': SARIMAX, 'refiner': ML,
    #  'feature_names': [...], ...}
    if not isinstance(fitted, dict):
        raise RuntimeError(f"Hybrid {alias} expected a dict bundle")
    base = fitted.get("base") or fitted.get("fitted")
    refiner = fitted.get("refiner")
    if base is None:
        raise RuntimeError(f"Hybrid {alias} missing 'base' component")
    if exog_future is None:
        raise ValueError(f"{alias} requires exog_future for the SARIMAX base")
    base_pred = base.predict(n_periods=n_periods,
                              X=exog_future.values[:n_periods])
    if refiner is None or feature_future is None:
        # No refiner correction — return base alone
        return base_pred
    refine_pred = refiner.predict(feature_future.values[:n_periods])
    return np.asarray(base_pred).ravel() + np.asarray(refine_pred).ravel()
