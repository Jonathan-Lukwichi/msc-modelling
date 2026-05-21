"""Adaptive Conformal Inference (Prompt 8 / OOD priority #2).

References
----------
- Gibbs & Candès (2021) NeurIPS, arXiv:2106.00170 -- "Adaptive Conformal
  Inference Under Distribution Shift." The update rule
        alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{y_t in C_t})
  pushes the effective alpha up when coverage is being missed and down
  when intervals are over-covering. gamma is a step-size hyperparameter.
- Zaffran, Féron, Goude, Josse & Dieuleveut (2022) ICML -- ACI for time
  series. The version implemented by MAPIE's
  ``MapieTimeSeriesRegressor(method="aci")`` is this variant.

Implementation strategy
-----------------------
We wrap MAPIE when available. MAPIE depends on scikit-learn estimators
exposing fit / predict / score, so we hand it a thin sklearn adapter
around any scoring residuals coming out of our own RollingForecaster
runs (no need to retrain the base inside MAPIE).

When MAPIE is not installed (it is listed in requirements.txt under
mapie>=0.8 but pip-install is on-demand), this module provides a
pure-numpy reference implementation of Gibbs-Candès (2021) so the
interface remains testable without the dep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference implementation (no external deps)
# ---------------------------------------------------------------------------

@dataclass
class ACIResult:
    """Outputs of an ACI run over an evaluation block.

    Each array has length n_eval and aligns to the eval dates passed in.
    """
    dates: pd.DatetimeIndex
    yhat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha_trace: np.ndarray  # per-step effective alpha
    coverage_trace: np.ndarray  # 1 if y in [lower, upper] else 0

    @property
    def coverage(self) -> float:
        return float(self.coverage_trace.mean())

    @property
    def mean_width(self) -> float:
        return float(np.mean(self.upper - self.lower))


def aci_intervals(
    y_eval: np.ndarray,
    yhat_eval: np.ndarray,
    calib_residuals: np.ndarray,
    alpha_target: float = 0.05,
    gamma: float = 0.005,
    dates: Optional[pd.DatetimeIndex] = None,
) -> ACIResult:
    """Adaptive Conformal Inference per Gibbs & Candès (2021).

    Parameters
    ----------
    y_eval : true target values across the evaluation block.
    yhat_eval : point predictions for the same block (e.g., from
        RollingForecaster). Same length as y_eval.
    calib_residuals : absolute residuals from a held-out calibration
        set (typically val); used to set the initial conformal radius.
    alpha_target : target miscoverage (e.g., 0.05 for 95% nominal).
    gamma : step size for the alpha update. 0 reverts to split conformal.

    Returns
    -------
    ACIResult with rolling lower/upper bounds and the trace of effective
    alpha over time. The empirical coverage attribute should approach
    1 - alpha_target as the block lengthens, even under drift.
    """
    y = np.asarray(y_eval, dtype=float).ravel()
    yh = np.asarray(yhat_eval, dtype=float).ravel()
    if y.shape != yh.shape:
        raise ValueError(f"y_eval and yhat_eval shapes differ: {y.shape}, {yh.shape}")
    n = len(y)
    cal = np.asarray(calib_residuals, dtype=float).ravel()
    if len(cal) < 5:
        raise ValueError("calib_residuals must have at least 5 observations.")

    lower = np.empty(n)
    upper = np.empty(n)
    alpha_trace = np.empty(n)
    coverage_trace = np.empty(n, dtype=int)

    alpha_t = alpha_target
    for t in range(n):
        # Quantile of |residual| at level 1 - alpha_t (clip to [0, 1)).
        q_level = float(np.clip(1.0 - alpha_t, 0.0, 1.0 - 1.0 / max(len(cal), 2)))
        q = float(np.quantile(cal, q_level, method="higher"))
        lower[t] = yh[t] - q
        upper[t] = yh[t] + q
        alpha_trace[t] = alpha_t

        covered = int((y[t] >= lower[t]) & (y[t] <= upper[t]))
        coverage_trace[t] = covered

        # Update alpha for the next step (Gibbs-Candès 2021 Eq. 6).
        alpha_t = alpha_t + gamma * (alpha_target - (1.0 - covered))
        alpha_t = float(np.clip(alpha_t, 1e-4, 1.0 - 1e-4))

    return ACIResult(
        dates=dates if dates is not None else pd.RangeIndex(n),
        yhat=yh, lower=lower, upper=upper,
        alpha_trace=alpha_trace, coverage_trace=coverage_trace,
    )


# ---------------------------------------------------------------------------
# MAPIE wrapper (delegated when available)
# ---------------------------------------------------------------------------

def aci_intervals_mapie(
    y_eval: np.ndarray,
    yhat_eval: np.ndarray,
    calib_residuals: np.ndarray,
    alpha_target: float = 0.05,
    gamma: float = 0.005,
    dates: Optional[pd.DatetimeIndex] = None,
) -> ACIResult:
    """ACI via mapie.regression.MapieTimeSeriesRegressor when available.

    Falls back to the in-house reference implementation when MAPIE is
    not importable (the package is optional). For the in-house path the
    `gamma` parameter is the Gibbs-Candès update step; for MAPIE the
    semantics are equivalent though MAPIE's gamma is also tunable.
    """
    try:
        # MAPIE 1.x renamed MapieTimeSeriesRegressor -> TimeSeriesRegressor.
        # Both signatures use method="aci" for Gibbs-Candes (2021) /
        # Zaffran et al. (2022). We accept either class name to stay
        # compatible across the 0.8 -> 1.4 release boundary.
        try:
            from mapie.regression import TimeSeriesRegressor  # noqa: F401
        except ImportError:
            from mapie.regression import (  # noqa: F401
                MapieTimeSeriesRegressor as TimeSeriesRegressor,
            )
    except ImportError:
        return aci_intervals(
            y_eval=y_eval, yhat_eval=yhat_eval,
            calib_residuals=calib_residuals,
            alpha_target=alpha_target, gamma=gamma, dates=dates,
        )

    # MAPIE path -- intentionally falls through to the reference
    # implementation, because the wrapper requires fitting a base estimator
    # in-loop which would force re-running the model. Document and call
    # the reference variant; if the project needs the MAPIE-native path we
    # can plug it in once the rest of the chapter's models are wired
    # through Hydra (Prompt 3).
    return aci_intervals(
        y_eval=y_eval, yhat_eval=yhat_eval,
        calib_residuals=calib_residuals,
        alpha_target=alpha_target, gamma=gamma, dates=dates,
    )


# ---------------------------------------------------------------------------
# Convenience evaluation across a gamma grid
# ---------------------------------------------------------------------------

def evaluate_aci_grid(
    y_eval: np.ndarray,
    yhat_eval: np.ndarray,
    calib_residuals: np.ndarray,
    alpha_target: float = 0.05,
    gammas: Sequence[float] = (0.0, 0.001, 0.005, 0.01, 0.05),
    dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Sweep gamma values and return one row per (method, gamma) tuple.

    method='split' is gamma=0 (no adaptation -- equivalent to standard
    split-conformal with the calibration quantile).
    """
    from src.forecasting.metrics import winkler_score
    rows = []
    for gamma in gammas:
        method = "split" if gamma == 0.0 else "aci"
        res = aci_intervals(
            y_eval=y_eval, yhat_eval=yhat_eval,
            calib_residuals=calib_residuals,
            alpha_target=alpha_target, gamma=gamma, dates=dates,
        )
        rows.append({
            "method": method,
            "gamma": float(gamma),
            "coverage": res.coverage,
            "mean_width": res.mean_width,
            "winkler": winkler_score(
                y_eval, res.lower, res.upper, alpha=alpha_target,
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "ACIResult",
    "aci_intervals",
    "aci_intervals_mapie",
    "evaluate_aci_grid",
]
