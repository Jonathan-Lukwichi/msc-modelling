"""Evaluation metrics (Ch3 §3.6.2).

MAPE is the primary ranking metric. MAE and RMSE complete the metric block.
R^2 is reported for goodness of fit. MAPE < 10% is conventionally regarded as
excellent in healthcare forecasting (Ch3 §3.6.2).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _to_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape}, predicted {p.shape}")
    return a, p


def mae(actual, predicted) -> float:
    a, p = _to_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _to_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted, eps: float | None = None) -> float:
    """Mean Absolute Percentage Error in %.

    Healthcare convention: drop rows where actual < eps (these are usually
    zero-arrival days that survived filtering and inflate MAPE pathologically).

    When eps is None (default), it is auto-set to 1% of the maximum actual
    value, so the same function works for both daily counts (mean ~60) and
    share targets in [0, 1] (mean ~0.5). Pass eps=0 to disable filtering.
    """
    a, p = _to_arrays(actual, predicted)
    if eps is None:
        # Auto: 1% of max |actual| -- adapts to count vs share scales
        eps = max(1e-9, 0.01 * float(np.nanmax(np.abs(a))))
    mask = a >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def r2(actual, predicted) -> float:
    a, p = _to_arrays(actual, predicted)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def score(actual, predicted) -> dict[str, float]:
    """Compute the full MAPE / MAE / RMSE / R² block in one call."""
    return {
        "MAPE": mape(actual, predicted),
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "R2": r2(actual, predicted),
    }


# ---------------------------------------------------------------------------
# Added by Prompt 2: scale-free + prediction-interval metrics
# ---------------------------------------------------------------------------

def mase(actual, predicted, y_train, seasonality: int = 7) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006, IJF 22(4):679-688).

    MASE = mean(|y - yhat|) / mean(|y_t - y_{t-m}|) computed on the training
    set. By construction MASE = 1 means parity with the seasonal naive
    forecast; <1 beats naive, >1 is worse than naive.

    The training-series denominator is the scale-free trick that makes MASE
    well-defined even when the actual series contains zeros (unlike MAPE).
    """
    a, p = _to_arrays(actual, predicted)
    y_tr = np.asarray(y_train, dtype=float).ravel()
    if len(y_tr) <= seasonality:
        raise ValueError(
            f"y_train has length {len(y_tr)} but seasonality={seasonality}; "
            "need at least seasonality + 1 train points."
        )
    diffs = np.abs(y_tr[seasonality:] - y_tr[:-seasonality])
    denom = float(np.mean(diffs))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(a - p)) / denom)


def coverage(actual, lower, upper) -> float:
    """Empirical interval coverage: fraction of actuals inside [lower, upper]."""
    a = np.asarray(actual, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError(
            f"shape mismatch: actual {a.shape}, lower {lo.shape}, upper {hi.shape}"
        )
    return float(np.mean((a >= lo) & (a <= hi)))


def winkler_score(actual, lower, upper, alpha: float) -> float:
    """Winkler (1972) score for prediction intervals at nominal level (1 - alpha).

    Defined per observation as:
        width                              if lower <= y <= upper
        width + (2/alpha) * (lower - y)    if y < lower
        width + (2/alpha) * (y - upper)    if y > upper
    Lower is better. Penalises both wide intervals and uncovered points.
    Returned value is the mean across observations.

    Reference: Winkler, R. L. (1972). "A decision-theoretic approach to
    interval estimation." JASA 67(337):187-191. Re-derived in the M4
    forecasting competition (Makridakis, Spiliotis & Assimakopoulos 2020).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    a = np.asarray(actual, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError(
            f"shape mismatch: actual {a.shape}, lower {lo.shape}, upper {hi.shape}"
        )
    width = hi - lo
    penalty = np.where(
        a < lo, (2.0 / alpha) * (lo - a),
        np.where(a > hi, (2.0 / alpha) * (a - hi), 0.0),
    )
    return float(np.mean(width + penalty))


def per_horizon_metrics(df: pd.DataFrame,
                          horizon_col: str = "horizon") -> pd.DataFrame:
    """Compute MAPE/MAE/RMSE/R2 for each value in df[horizon_col].

    Expects columns 'actual' and 'predicted' in df.
    """
    if not {"actual", "predicted", horizon_col}.issubset(df.columns):
        raise ValueError(
            f"df must have columns 'actual', 'predicted', {horizon_col!r}; "
            f"got {list(df.columns)}"
        )
    rows = []
    for h, sub in df.groupby(horizon_col):
        s = score(sub["actual"], sub["predicted"])
        rows.append({"horizon": int(h), "n": len(sub), **s})
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def score_per_horizon(
    actual: pd.Series,
    predicted: pd.Series,
    horizons: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
) -> pd.DataFrame:
    """Per-horizon metrics, expects actual and predicted aligned to a fold's test_idx."""
    rows = []
    for h in horizons:
        a = actual.iloc[h - 1 : h]
        p = predicted.iloc[h - 1 : h]
        rows.append({"horizon_day": h, **score(a, p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    actual = rng.poisson(60, size=100)
    predicted = actual + rng.normal(0, 5, size=100)
    print("Smoke test on synthetic data:")
    for k, v in score(actual, predicted).items():
        print(f"  {k:>4s}: {v:.4f}")
