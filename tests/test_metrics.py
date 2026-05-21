"""Smoke tests for src/forecasting/metrics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.metrics import (
    mape, mae, rmse, r2, score,
    mase, coverage, winkler_score, per_horizon_metrics,
)


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_constant_prediction_at_mean_gives_r2_zero():
    y = np.array([10.0, 20.0, 30.0])
    yhat = np.full_like(y, y.mean())
    assert r2(y, yhat) == pytest.approx(0.0, abs=1e-9)


def test_mape_drops_below_eps():
    """eps filter prevents division-by-near-zero blowups."""
    y = np.array([0.5, 50.0, 60.0])   # first row < eps=1
    yhat = np.array([5.0, 50.0, 60.0])
    # Only the last two rows should count
    assert mape(y, yhat, eps=1.0) == 0.0


def test_score_returns_all_four_metrics():
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 4.0, 5.2])
    s = score(y, yhat)
    assert set(s.keys()) == {"MAPE", "MAE", "RMSE", "R2"}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# Prompt 2 — MASE, Winkler, coverage, per-horizon
# ---------------------------------------------------------------------------

def test_mase_perfect_prediction_is_zero():
    y_train = np.array([10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0,
                         14.0, 16.0])
    y_test = np.array([15.0, 14.0, 16.0])
    yhat = y_test.copy()
    assert mase(y_test, yhat, y_train, seasonality=7) == 0.0


def test_mase_for_seasonal_naive_is_one():
    """If we forecast y_{t} = y_{t-7}, MASE should be ~1 on a series whose
    drift is dominated by week-on-week noise.
    """
    rng = np.random.default_rng(0)
    n = 200
    seasonal = np.tile(np.array([10, 8, 12, 9, 11, 7, 13], dtype=float), n // 7 + 1)[:n]
    noise = rng.normal(0, 1, size=n)
    y = seasonal + noise

    train = y[:150]
    test = y[150:]
    # Forecast: yhat_t = y_{t-7}
    yhat_seasonal_naive = y[150 - 7 : -7]
    yhat_seasonal_naive = yhat_seasonal_naive[: len(test)]
    m = mase(test, yhat_seasonal_naive, train, seasonality=7)
    assert 0.7 <= m <= 1.5, f"Seasonal-naive MASE on its native baseline should be ~1, got {m}"


def test_winkler_penalises_uncovered():
    actual = np.array([10.0])
    lower = np.array([20.0])
    upper = np.array([25.0])
    # Uncovered: width=5, penalty=(2/0.05)*(20-10)=400 -> 405
    w = winkler_score(actual, lower, upper, alpha=0.05)
    assert w == pytest.approx(5.0 + (2.0 / 0.05) * 10.0, rel=1e-9)


def test_winkler_just_width_when_covered():
    actual = np.array([12.0, 14.0])
    lower = np.array([10.0, 13.0])
    upper = np.array([15.0, 16.0])
    # Both covered: widths 5 + 3 = 8 / 2 = 4.0
    w = winkler_score(actual, lower, upper, alpha=0.05)
    assert w == pytest.approx(4.0, rel=1e-9)


def test_coverage_matches_numpy_definition():
    actual = np.array([1, 2, 3, 4, 5], dtype=float)
    lower = np.array([0, 1, 2, 5, 6], dtype=float)
    upper = np.array([5, 5, 5, 10, 10], dtype=float)
    expected = float(np.mean((actual >= lower) & (actual <= upper)))
    assert coverage(actual, lower, upper) == pytest.approx(expected, rel=1e-9)


def test_per_horizon_metrics_groups_by_horizon():
    df = pd.DataFrame({
        "horizon": [1, 1, 2, 2, 3, 3],
        "actual": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        "predicted": [10.0, 20.0, 11.0, 21.0, 13.0, 23.0],
    })
    out = per_horizon_metrics(df)
    assert set(out.columns) >= {"horizon", "n", "MAPE", "MAE", "RMSE", "R2"}
    assert list(out["horizon"]) == [1, 2, 3]
    assert out.loc[out["horizon"] == 1, "MAE"].iloc[0] == 0.0
    assert out.loc[out["horizon"] == 3, "MAE"].iloc[0] == 3.0
