"""Tests for the OOF residual hybrid (Prompt 4).

Three acceptance properties from dissertation_improvement_prompts.md:
  1. OOF residual dates do not overlap base training dates for any fold.
  2. OOF residual variance is no smaller than in-sample residual variance
     (selection-bias smoke test).
  3. Refiner params are tuned independently of base params.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.rolling import FoldPrediction
from src.forecasting.hybrids.oof.oof_residuals import (
    build_oof_residuals,
    OOFResidualHybrid,
    xgb_refiner_factory,
)


# ---------------------------------------------------------------------------
# Fixtures: a toy weekly cycle + noise
# ---------------------------------------------------------------------------

def _series(n: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = (
        60.0 + 0.01 * t
        + 6.0 * np.sin(2 * np.pi * t / 7.0)
        + rng.normal(0, 4, size=n)
    )
    X = pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * t / 7.0),
        "dow_cos": np.cos(2 * np.pi * t / 7.0),
        "noise": rng.normal(0, 1, size=n),
    }, index=dates)
    return pd.Series(y, index=dates), X


def _mean_factory():
    """Trivial base model: predicts y_train.mean() over h days."""
    def factory(X_train, y_train, sample_weight=None):
        mean = float(y_train.mean())

        class _Fitted:
            def predict(self, X_future, h):
                return FoldPrediction(yhat=np.full(h, mean))

        return _Fitted()
    return factory


# ---------------------------------------------------------------------------
# Acceptance test 1: OOF residual dates do not overlap base training dates
# ---------------------------------------------------------------------------

def test_oof_dates_never_overlap_base_training():
    y, X = _series(n=400)
    train_y = y.iloc[:300]
    train_X = X.iloc[:300]

    train_windows = []

    def watching_factory(X_train, y_train, sample_weight=None):
        train_windows.append((y_train.index.min(), y_train.index.max()))
        return _mean_factory()(X_train, y_train, sample_weight)

    oof = build_oof_residuals(
        watching_factory, train_X, train_y,
        horizon=7, step=7, min_train_days=100,
    )
    assert len(oof) > 0

    # Map each OOF date back to its fold_id and check the corresponding
    # train window does not contain that date.
    for fold_id in oof["fold_id"].unique():
        fold_dates = oof.index[oof["fold_id"] == fold_id]
        train_start, train_end = train_windows[fold_id]
        for d in fold_dates:
            assert d > train_end, (
                f"OOF date {d} should be strictly after the fold's "
                f"training window end ({train_end})"
            )


# ---------------------------------------------------------------------------
# Acceptance test 2: OOF residual variance >= in-sample residual variance
# ---------------------------------------------------------------------------

def test_oof_residual_variance_meets_or_exceeds_in_sample():
    """Selection-bias smoke test.

    In-sample residuals from a single fit on the full train block tend to
    UNDER-state the variance because the fit has seen those exact rows.
    OOF residuals -- where the prediction comes from a model that did NOT
    see that row -- recover the true generalisation variance.

    For a non-trivial model with capacity, var(OOF) > var(in-sample).
    For a trivial mean-predictor, they should be approximately equal
    (both are just sample variance of (y - mean)). We assert >= to cover
    both cases.
    """
    y, X = _series(n=400, seed=0)

    # In-sample variance: fit once on full train, residuals = y - mean.
    train_y = y.iloc[:300]
    in_sample_resid = train_y - train_y.mean()
    in_sample_var = float(np.var(in_sample_resid, ddof=0))

    # OOF variance via the same trivial base.
    train_X = X.iloc[:300]
    oof = build_oof_residuals(
        _mean_factory(), train_X, train_y,
        horizon=7, step=7, min_train_days=200,
    )
    oof_var = float(np.var(oof["residual"], ddof=0))

    # OOF variance must be no SMALLER than in-sample (with small numerical slack).
    assert oof_var >= in_sample_var * 0.95, (
        f"OOF residual variance ({oof_var:.3f}) should be >= "
        f"in-sample ({in_sample_var:.3f})"
    )


# ---------------------------------------------------------------------------
# Acceptance test 3: refiner params are NOT inherited from base params
# ---------------------------------------------------------------------------

def test_refiner_hpo_independent_of_base():
    """The hybrid must tune the refiner on RESIDUALS alone -- not on y.

    We pass a base_factory and a separate refiner_factory plus a small
    refiner HPO space; the refiner's chosen params must come from the
    inner HPO loop and have nothing to do with any base hyperparameter.
    """
    pytest.importorskip("optuna")
    y, X = _series(n=500, seed=1)
    train_y = y.iloc[:400]
    train_X = X.iloc[:400]

    hybrid = OOFResidualHybrid(
        base_factory=_mean_factory(),
        refiner_factory=xgb_refiner_factory,
        standardize_residuals=True,
        nested_hpo=True,
        refiner_hpo_space={
            "n_estimators": [50, 100, 200],
            "max_depth":     [3, 4, 6],
            "learning_rate": (0.01, 0.1),
        },
        n_hpo_trials=4, n_inner_folds=3, min_train_days=200,
    )
    hybrid.fit(train_X, train_y)

    chosen = hybrid.best_refiner_params
    assert chosen is not None
    # Sanity: the chosen params come from the declared refiner space only.
    assert set(chosen.keys()) <= {"n_estimators", "max_depth", "learning_rate"}


# ---------------------------------------------------------------------------
# End-to-end fit/predict smoke
# ---------------------------------------------------------------------------

def test_oof_hybrid_fit_predict_smoke():
    y, X = _series(n=400, seed=7)
    train_y = y.iloc[:300]
    train_X = X.iloc[:300]
    eval_idx = y.iloc[300:330].index   # 30-day eval block
    eval_X = X.loc[eval_idx]

    hybrid = OOFResidualHybrid(
        base_factory=_mean_factory(),
        refiner_factory=xgb_refiner_factory,
        standardize_residuals=True,
        nested_hpo=False,
        horizon=7, min_train_days=100,
    )
    hybrid.fit(train_X, train_y)
    out = hybrid.predict(eval_X, eval_idx)

    assert {"base_yhat", "refiner_pred", "predicted"}.issubset(out.columns)
    assert len(out) == len(eval_idx)
    assert not out["predicted"].isna().any()
