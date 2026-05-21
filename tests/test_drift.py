"""Tests for sliding-window CV and importance weighting (Prompt 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.drift.importance_weights import kmm_weights
from src.forecasting.drift.sliding_cv import (
    make_iw_sample_weight_fn, sliding_forecaster,
)
from src.forecasting.rolling import FoldPrediction


# ---------------------------------------------------------------------------
# KMM
# ---------------------------------------------------------------------------

def test_kmm_weights_non_negative_and_normalised():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(300, 4))
    X_recent = rng.normal(0.5, 1, size=(60, 4))  # shifted distribution
    w = kmm_weights(X_train, X_recent)
    assert len(w) == 300
    assert (w >= 0).all()
    # Normalisation: mean(w) ~= 1 (sum ~= n_train)
    assert abs(w.mean() - 1.0) < 0.05


def test_kmm_weights_concentrate_on_shifted_subset():
    """When the recent set is concentrated near the END of the train
    sample, KMM weights should be larger on the trailing observations.
    """
    n = 200
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(n, 3))
    # Inject a sharp shift onto the trailing 40 rows
    X[-40:] += 3.0
    X_recent = X[-40:]   # the "target" distribution
    w = kmm_weights(X, X_recent)
    mean_first = float(w[:160].mean())
    mean_last = float(w[-40:].mean())
    assert mean_last > mean_first, (
        f"KMM should give larger weights to the shifted region "
        f"({mean_last:.3f} vs {mean_first:.3f})"
    )


# ---------------------------------------------------------------------------
# sliding_forecaster
# ---------------------------------------------------------------------------

def _mean_factory():
    def factory(X_train, y_train, sample_weight=None):
        if sample_weight is not None:
            mean = float(np.average(y_train.values, weights=sample_weight))
        else:
            mean = float(y_train.mean())

        class _Fitted:
            def predict(self, X_future, h):
                return FoldPrediction(yhat=np.full(h, mean))

        return _Fitted()
    return factory


def test_sliding_window_truncates_train_size():
    n = 800
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    y = pd.Series(np.arange(n, dtype=float), index=dates)
    X = pd.DataFrame({"x": np.arange(n, dtype=float)}, index=dates)
    eval_idx = dates[500:600]

    seen = []

    def factory(X_train, y_train, sample_weight=None):
        seen.append(len(y_train))
        return _mean_factory()(X_train, y_train, sample_weight)

    rf = sliding_forecaster(
        factory, window_days=180, weight_method=None,
        step_days=7, horizon_days=7, min_train_days=180,
    )
    rf.fit_predict(X, y, eval_idx)
    assert max(seen) == 180


def test_iw_sample_weight_fn_returns_uniform_when_too_few_recent():
    """When the train window is barely larger than recent_days, the
    fallback path returns uniform weights.
    """
    fn = make_iw_sample_weight_fn(method="kmm", recent_days=90)
    X = pd.DataFrame({"x": np.arange(100, dtype=float)})
    y = pd.Series(np.arange(100, dtype=float))
    w = fn(X, y)
    assert np.allclose(w, 1.0)
