"""Sliding-window + sample-weighted helpers built on RollingForecaster.

The actual sliding-window machinery already lives inside
``RollingForecaster`` via the ``window_days`` parameter (Prompt 1). This
module supplies the user-facing helpers and a default
``recent_window_picker`` for KMM/RuLSIF weight estimation -- the typical
"compare last 90 days of train to all train days" pattern from Sugiyama
et al. (2007).
"""
from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from src.forecasting.rolling import (
    RollingForecaster, ModelFactory, SampleWeightFn,
)


def make_iw_sample_weight_fn(
    method: Literal["kmm", "rulsif"] = "rulsif",
    recent_days: int = 90,
    alpha: float = 0.1,
) -> SampleWeightFn:
    """Build a sample_weight_fn that compares the last ``recent_days`` of
    the current training fold to the rest, using either KMM or RuLSIF.

    Returns a callable ``(X_train, y_train) -> np.ndarray`` suitable for
    ``RollingForecaster(sample_weight_fn=...)``.
    """
    from .importance_weights import kmm_weights, rulsif_weights

    def fn(X_train, y_train):
        n = len(X_train) if X_train is not None else len(y_train)
        if n <= recent_days * 2:
            return np.ones(n)
        # X for distance comparison: prefer X_train if available, else
        # a univariate matrix of y.
        if X_train is not None:
            X = X_train.values.astype(float)
        else:
            X = y_train.values.reshape(-1, 1).astype(float)
        recent = X[-recent_days:]
        if method == "rulsif":
            return rulsif_weights(X, recent, alpha=alpha)
        return kmm_weights(X, recent)

    return fn


def sliding_forecaster(
    model_factory: ModelFactory,
    window_days: int,
    *,
    weight_method: Optional[Literal["kmm", "rulsif"]] = None,
    recent_days: int = 90,
    step_days: int = 7,
    horizon_days: int = 7,
    min_train_days: int = 365,
) -> RollingForecaster:
    """Convenience constructor for a sliding-window forecaster optionally
    importance-weighted by KMM or RuLSIF.
    """
    swfn = (
        make_iw_sample_weight_fn(method=weight_method, recent_days=recent_days)
        if weight_method is not None else None
    )
    return RollingForecaster(
        model_factory=model_factory,
        step_days=step_days, horizon_days=horizon_days,
        min_train_days=min_train_days, window_days=window_days,
        sample_weight_fn=swfn,
    )


__all__ = ["make_iw_sample_weight_fn", "sliding_forecaster"]
