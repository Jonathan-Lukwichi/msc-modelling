"""Smoke tests for src/forecasting/cv.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.cv import rolling_origin, count_folds


@pytest.fixture
def two_year_index():
    return pd.date_range("2022-03-01", periods=2 * 365, freq="D")


def test_count_matches_iterator(two_year_index):
    n = count_folds(two_year_index, horizon_days=7, step_days=7, min_train_days=365)
    actual = sum(1 for _ in rolling_origin(two_year_index, horizon_days=7,
                                            step_days=7, min_train_days=365))
    assert n == actual


def test_folds_are_expanding(two_year_index):
    sizes = [
        len(f.train_idx) for f in rolling_origin(
            two_year_index, horizon_days=7, step_days=7, min_train_days=365
        )
    ]
    for prev, curr in zip(sizes, sizes[1:]):
        assert curr > prev, "Training window must expand monotonically"


def test_train_test_no_overlap(two_year_index):
    for f in rolling_origin(two_year_index, horizon_days=7, step_days=7,
                             min_train_days=365):
        overlap = set(f.train_idx) & set(f.test_idx)
        assert not overlap, f"Fold {f.fold_id} has train/test overlap"


def test_horizon_is_seven(two_year_index):
    for f in rolling_origin(two_year_index, horizon_days=7, step_days=7,
                             min_train_days=365):
        assert len(f.test_idx) == 7


def test_raises_on_short_series():
    short = pd.date_range("2024-01-01", periods=100, freq="D")
    with pytest.raises(ValueError):
        list(rolling_origin(short, horizon_days=7, step_days=7, min_train_days=365))
