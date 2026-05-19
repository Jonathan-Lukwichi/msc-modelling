"""Three zero/light-fit reference-floor baselines (plan §6).

  naive_yest:     y_hat(t) = y(t-1)
  naive_seasonal: y_hat(t) = y(t-7)
  dow_mean:       y_hat(t) = mean of same weekday on train fold

Any later model that fails to beat these on validation is suspect.

Grounding (§5.2.2 / §5.2.3):
  - lag-1 ACF = 0.538, lag-7 ACF = 0.490 -> shift baselines have real signal
  - Kruskal-Wallis on day-of-week p = 3.3e-30 -> DoW mean is a calendar floor
"""
from __future__ import annotations

import pandas as pd


def predict_naive_yest(series: pd.Series) -> pd.Series:
    """y_hat(t) = y(t-1). Returns NaN for the first index."""
    return series.shift(1)


def predict_naive_seasonal(series: pd.Series, lag: int = 7) -> pd.Series:
    """y_hat(t) = y(t-7). Returns NaN for the first `lag` indices."""
    return series.shift(lag)


def predict_dow_mean(train_series: pd.Series, full_index: pd.DatetimeIndex) -> pd.Series:
    """y_hat(t) = mean of same weekday on the training fold.

    Maps the seven train-weekday means onto every date in full_index.
    """
    if not isinstance(train_series.index, pd.DatetimeIndex):
        raise TypeError("train_series must have a DatetimeIndex")
    if not isinstance(full_index, pd.DatetimeIndex):
        raise TypeError("full_index must be a DatetimeIndex")
    weekday_means = train_series.groupby(train_series.index.dayofweek).mean()
    return pd.Series(full_index.dayofweek.map(weekday_means).to_numpy(),
                     index=full_index,
                     name="dow_mean")
