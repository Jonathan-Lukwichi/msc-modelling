"""Smoke tests for src/forecasting/features.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import (
    build_task1_exogenous, build_task2_exogenous, StandardScaler,
)
from conftest import REAL_DATA_AVAILABLE


@pytest.fixture(scope="module")
def train_df():
    if not REAL_DATA_AVAILABLE:
        pytest.skip(
            "Requires the raw G1-G4 hospital CSVs referenced by "
            "configs/paths.local.yaml; not available in this environment. "
            "See README 'Data access'."
        )
    splits = Splits.from_config()
    return splits.slice(load_g1(), "train")


@pytest.fixture(scope="module")
def val_df():
    if not REAL_DATA_AVAILABLE:
        pytest.skip(
            "Requires the raw G1-G4 hospital CSVs referenced by "
            "configs/paths.local.yaml; not available in this environment. "
            "See README 'Data access'."
        )
    splits = Splits.from_config()
    return splits.slice(load_g1(), "val")


def test_task1_exogenous_has_15_columns(train_df):
    """6 DoW dummies + 2 continuous (scaled) + 7 calendar binaries = 15."""
    X, _ = build_task1_exogenous(train_df, fit_scaler=True)
    assert X.shape[1] == 15, f"Expected 15 cols, got {X.shape[1]}"


def test_task1_exogenous_contains_523_5_inventory(train_df):
    """Every §5.2.5 raw 10 feature is represented."""
    X, _ = build_task1_exogenous(train_df, fit_scaler=True)
    required_binaries = [
        "is_weekend", "is_long_weekend", "is_public_holiday",
        "is_school_holiday", "is_festive_season", "is_winter_holiday",
        "is_near_holiday",
    ]
    for b in required_binaries:
        assert b in X.columns, f"Missing §5.2.5 binary: {b}"
    assert "temp_mean_C" in X.columns
    assert "wind_max_kmh" in X.columns
    dow_dummies = [c for c in X.columns if c.startswith("dow_")]
    assert len(dow_dummies) == 6, f"Expected 6 DoW dummies, got {len(dow_dummies)}"


def test_scaler_fit_on_train_no_leak(train_df, val_df):
    """Scaler is fit on train; val mean should NOT be exactly 0."""
    X_train, scaler = build_task1_exogenous(train_df, fit_scaler=True)
    X_val, _ = build_task1_exogenous(val_df, scaler=scaler)
    # Train scaled cols should be near 0 mean
    assert abs(X_train["temp_mean_C"].mean()) < 1e-6
    # Val mean reflects drift; non-zero
    assert abs(X_val["temp_mean_C"].mean()) > 0.01


def test_task2_surgery_has_sign_reversal_interactions(train_df):
    """Surgery exog block per §5.3.3 must include the three sign-reversal columns."""
    X, _ = build_task2_exogenous(train_df, "Surgery", fit_scaler=True)
    interaction_cols = [c for c in X.columns if c.startswith("surgery_")]
    assert "surgery_is_weekend" in X.columns
    assert "surgery_is_long_weekend" in X.columns
    assert "surgery_is_public_holiday" in X.columns
    assert len(interaction_cols) == 3


def test_task2_surgery_is_weather_flat(train_df):
    """Surgery is weather-flat per §5.3.3 — neither temp nor wind."""
    X, _ = build_task2_exogenous(train_df, "Surgery", fit_scaler=True)
    assert "temp_mean_C" not in X.columns
    assert "wind_max_kmh" not in X.columns
