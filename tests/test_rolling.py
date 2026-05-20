"""Tests for the unified RollingForecaster (Prompt 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.rolling import (
    FoldPrediction,
    RollingForecaster,
    make_arima_factory,
    make_xgboost_factory,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_series(n: int, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """A two-year-ish daily series with weekly cycle + noise + linear trend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = (
        60.0 + 0.01 * t
        + 8.0 * np.sin(2 * np.pi * t / 7.0)
        + rng.normal(0, 4, size=n)
    )
    X = pd.DataFrame({
        "x1": np.cos(2 * np.pi * t / 7.0),
        "x2": rng.normal(0, 1, size=n),
    }, index=dates)
    return pd.Series(y, index=dates, name="y"), X


def _identity_factory(X_train, y_train, sample_weight=None):
    """Toy model: predicts the train mean. Lets us check iteration logic."""
    mean = float(y_train.mean())

    class _Fitted:
        def predict(self, X_future, h):
            return FoldPrediction(yhat=np.full(h, mean))

    return _Fitted()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_expanding_vs_sliding_train_lengths():
    """Sliding=N gives at most N train days; expanding grows monotonically."""
    y, X = _make_series(900)
    eval_idx = y.index[400:600]  # 200-day eval block

    seen_train_lengths_expanding = []
    seen_train_lengths_sliding = []

    def expanding_factory(X_train, y_train, sample_weight=None):
        seen_train_lengths_expanding.append(len(y_train))
        return _identity_factory(X_train, y_train, sample_weight)

    def sliding_factory(X_train, y_train, sample_weight=None):
        seen_train_lengths_sliding.append(len(y_train))
        return _identity_factory(X_train, y_train, sample_weight)

    rf_exp = RollingForecaster(
        model_factory=expanding_factory, min_train_days=1,
    )
    _ = rf_exp.fit_predict(X, y, eval_idx)

    rf_slide = RollingForecaster(
        model_factory=sliding_factory, window_days=180, min_train_days=180,
    )
    _ = rf_slide.fit_predict(X, y, eval_idx)

    # Expanding train sizes grow monotonically.
    assert all(
        b >= a for a, b in zip(
            seen_train_lengths_expanding, seen_train_lengths_expanding[1:]
        )
    ), "Expanding window should monotonically grow"
    # Sliding train sizes never exceed window_days.
    assert max(seen_train_lengths_sliding) <= 180


def test_57_folds_over_396_day_block_h7_step7():
    """The Jan-2025 ~ Jan-2026 test block has 396 days; horizon 7 step 7 = 57 folds.

    (Actually ceil(396/7) = 57; final fold's horizon is truncated to 4 days.)
    """
    y, X = _make_series(900)
    test_block = y.index[500:500 + 396]
    assert len(test_block) == 396

    rf = RollingForecaster(
        model_factory=_identity_factory,
        step_days=7, horizon_days=7, min_train_days=1,
    )
    preds = rf.fit_predict(X, y, test_block)
    assert len(preds) == 396, f"Expected 396 predictions, got {len(preds)}"
    # Exactly 57 fold_ids
    assert preds["fold_id"].nunique() == 57, (
        f"Expected 57 folds, got {preds['fold_id'].nunique()}"
    )
    # Last fold has 4 days (396 mod 7).
    last_fold_size = (preds["fold_id"] == preds["fold_id"].max()).sum()
    assert last_fold_size == 4


def test_no_train_eval_overlap():
    """Per fold, the model factory must never see eval dates in y_train."""
    y, X = _make_series(900)
    eval_idx = y.index[500:700]

    overlap_seen = []

    def picky_factory(X_train, y_train, sample_weight=None):
        # Compare the train window's date range to the upcoming eval slot.
        # The current train must end BEFORE the next predicted day.
        train_max = y_train.index.max()
        overlap_seen.append(train_max)
        return _identity_factory(X_train, y_train, sample_weight)

    rf = RollingForecaster(model_factory=picky_factory, min_train_days=1)
    preds = rf.fit_predict(X, y, eval_idx)

    # For each fold's first predicted date, its train_max must equal that date - 1.
    grouped = preds.reset_index().groupby("fold_id")["date"].min().sort_index()
    for fold_id, first_pred_date in grouped.items():
        train_max = overlap_seen[fold_id]
        assert first_pred_date == train_max + pd.Timedelta(days=1), (
            f"Fold {fold_id}: train ends {train_max}, "
            f"first pred {first_pred_date}; expected adjacency"
        )


def test_sample_weight_fn_is_applied():
    """sample_weight_fn output reaches the factory."""
    y, X = _make_series(600)
    eval_idx = y.index[300:400]
    captured: list = []

    def weights(X_train, y_train):
        n = len(y_train)
        return np.linspace(0.1, 1.0, n)

    def capture_factory(X_train, y_train, sample_weight=None):
        captured.append(sample_weight)
        return _identity_factory(X_train, y_train, sample_weight)

    rf = RollingForecaster(
        model_factory=capture_factory,
        sample_weight_fn=weights, min_train_days=1,
    )
    _ = rf.fit_predict(X, y, eval_idx)

    # At least one call must have received non-None weights.
    assert any(w is not None for w in captured)
    # All captured weights match the synthetic linspace.
    for w in captured:
        if w is not None:
            assert np.isclose(w[0], 0.1) and np.isclose(w[-1], 1.0)


def test_arima_wrapper_byte_identical_to_legacy():
    """The new ARIMA wrapper produces the same predictions as the legacy loop."""
    from pmdarima import ARIMA as PmARIMA

    y, _ = _make_series(400)
    eval_idx = y.index[200:240]  # 40-day eval

    # Legacy loop (copied from the pre-refactor function).
    rows_legacy = []
    block_start, block_end = eval_idx[0], eval_idx[-1]
    origin_pos = y.index.get_loc(block_start) - 1
    while origin_pos < y.index.get_loc(block_end):
        train_through = y.iloc[: origin_pos + 1]
        n_remaining = y.index.get_loc(block_end) - origin_pos
        h = int(min(7, n_remaining))
        model = PmARIMA(order=(1, 1, 1), suppress_warnings=True)
        model.fit(train_through.values)
        yhat, conf = model.predict(n_periods=h, return_conf_int=True, alpha=0.05)
        dates = y.index[origin_pos + 1 : origin_pos + 1 + h]
        for date, y_pred, lo, hi in zip(dates, yhat, conf[:, 0], conf[:, 1]):
            rows_legacy.append({"date": date, "predicted": float(y_pred),
                                  "lower_95": float(lo), "upper_95": float(hi)})
        origin_pos += 7
    legacy = pd.DataFrame(rows_legacy)

    # New wrapper.
    from src.forecasting.models.arima import rolling_forecast
    new = rolling_forecast(y, eval_idx, order=(1, 1, 1), step_days=7, alpha=0.05)

    assert len(new) == len(legacy)
    np.testing.assert_array_equal(
        legacy["date"].values, new["date"].values
    )
    np.testing.assert_allclose(
        legacy["predicted"].values, new["predicted"].values, rtol=1e-9,
    )
    np.testing.assert_allclose(
        legacy["lower_95"].values, new["lower_95"].values, rtol=1e-9,
    )


def test_xgboost_wrapper_byte_identical_to_legacy():
    """The new XGBoost wrapper produces the same predictions as the legacy loop."""
    from xgboost import XGBRegressor

    y, X = _make_series(500)
    eval_idx = y.index[300:340]

    params = {
        "n_estimators": 50, "max_depth": 3,
        "learning_rate": 0.1, "subsample": 1.0,
    }

    rows_legacy = []
    block_start, block_end = eval_idx[0], eval_idx[-1]
    origin_pos = y.index.get_loc(block_start) - 1
    while origin_pos < y.index.get_loc(block_end):
        Xt = X.iloc[: origin_pos + 1]
        yt = y.iloc[: origin_pos + 1]
        n_remaining = y.index.get_loc(block_end) - origin_pos
        h = int(min(7, n_remaining))
        Xf = X.iloc[origin_pos + 1 : origin_pos + 1 + h]
        m = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=42, verbosity=0, n_jobs=-1)
        m.fit(Xt.values, yt.values)
        yhat = m.predict(Xf.values)
        for date, yp in zip(Xf.index, yhat):
            rows_legacy.append({"date": date, "predicted": float(yp)})
        origin_pos += 7
    legacy = pd.DataFrame(rows_legacy)

    from src.forecasting.models.xgboost_m import rolling_forecast
    new = rolling_forecast(X, y, eval_idx, params=params, step_days=7, seed=42)

    assert len(new) == len(legacy)
    np.testing.assert_array_equal(
        legacy["date"].values, new["date"].values
    )
    np.testing.assert_allclose(
        legacy["predicted"].values, new["predicted"].values, rtol=1e-9,
    )


def test_constructor_validates_inputs():
    with pytest.raises(ValueError):
        RollingForecaster(
            model_factory=_identity_factory, step_days=0,
        )
    with pytest.raises(ValueError):
        RollingForecaster(
            model_factory=_identity_factory, refit_every=0,
        )
    with pytest.raises(ValueError):
        RollingForecaster(
            model_factory=_identity_factory,
            window_days=100, min_train_days=200,
        )
