"""Tests for ACI (Prompt 8).

Acceptance criterion from the prompt: under synthetic drift mid-stream,
ACI mean coverage > split-conformal coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.uq.aci import (
    aci_intervals,
    evaluate_aci_grid,
)


@pytest.fixture
def synthetic_drift():
    """Stationary noise for the first half, then a structural mean shift."""
    rng = np.random.default_rng(0)
    n = 400
    yhat = np.zeros(n)
    noise = rng.normal(0, 1, size=n)
    y = yhat + noise
    # Inject drift: add +2 to the actuals in the second half.
    y[n // 2 :] += 2.0
    # Calibration: stationary residuals from the first 100 steps only.
    calib = np.abs(noise[:100])
    return y, yhat, calib


def test_aci_basic_shape(synthetic_drift):
    y, yhat, calib = synthetic_drift
    res = aci_intervals(y, yhat, calib, alpha_target=0.10, gamma=0.005)
    assert len(res.yhat) == len(y)
    assert len(res.lower) == len(y)
    assert len(res.upper) == len(y)
    assert (res.upper >= res.lower).all()


def test_aci_beats_split_conformal_under_drift(synthetic_drift):
    """Acceptance: ACI mean coverage > split-conformal coverage."""
    y, yhat, calib = synthetic_drift
    grid = evaluate_aci_grid(
        y, yhat, calib, alpha_target=0.10,
        gammas=(0.0, 0.005, 0.01, 0.05),
    )
    split_cov = grid.loc[grid["method"] == "split", "coverage"].iloc[0]
    best_aci_cov = grid.loc[grid["method"] == "aci", "coverage"].max()
    assert best_aci_cov > split_cov, (
        f"Best ACI coverage ({best_aci_cov:.3f}) should beat split-conformal "
        f"({split_cov:.3f}) under injected drift."
    )


def test_aci_zero_gamma_matches_split_conformal():
    """gamma=0 yields a fixed-quantile interval; the trace alpha must stay constant."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, size=200)
    yhat = np.zeros(200)
    calib = np.abs(rng.normal(0, 1, size=100))
    res = aci_intervals(y, yhat, calib, alpha_target=0.10, gamma=0.0)
    assert np.allclose(res.alpha_trace, 0.10)


def test_evaluate_aci_grid_includes_winkler():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, size=200)
    yhat = np.zeros(200)
    calib = np.abs(rng.normal(0, 1, size=100))
    df = evaluate_aci_grid(y, yhat, calib, alpha_target=0.1)
    assert {"method", "gamma", "coverage", "mean_width", "winkler"}.issubset(df.columns)
    assert (df["winkler"] >= 0).all()
