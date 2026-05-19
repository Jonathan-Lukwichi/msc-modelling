"""Smoke tests for src/forecasting/metrics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.metrics import mape, mae, rmse, r2, score


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
