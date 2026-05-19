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


def mape(actual, predicted, eps: float = 1.0) -> float:
    """Mean Absolute Percentage Error in %.

    Healthcare convention: drop rows where actual < eps (these are usually
    zero-arrival days that survived filtering and inflate MAPE pathologically).
    Default eps=1 keeps every non-zero day.
    """
    a, p = _to_arrays(actual, predicted)
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
