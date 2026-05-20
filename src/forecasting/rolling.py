"""Unified rolling-origin forecaster (Prompt 1 of Ch6 refactor).

Replaces the five copy-paste rolling_forecast() implementations in
models/{arima,sarimax,negbin,xgboost_m,ann,lstm}.py (and the duplicate
copies in scripts/17_final_test.py and scripts/21_uncertainty_quantification.py)
with a single class that parameterises the loop.

The model-specific code is reduced to a "model factory" closure that
takes (X_train, y_train, sample_weight) and returns a FittedModel which
exposes `predict(X_future, h) -> FoldPrediction`. Each model module
keeps its public `rolling_forecast()` function as a 5-10 line wrapper
that builds such a factory and delegates to RollingForecaster.

Design references
-----------------
- Hyndman & Athanasopoulos (2021), FPP3 §5.10  — rolling-origin evaluation.
- Bontempi, Ben Taieb & Le Borgne (2013), Springer LNBIP 138:62-77
  — recursive vs direct multi-step strategies (this class implements
    recursive single-fit-per-origin; direct strategy is Prompt 6).
- Sugiyama, Krauledat & Müller (2007), JMLR 8:985-1005  — IWCV motivates
  the `sample_weight_fn` hook (used by Prompt 7's sliding+RuLSIF).

The class is intentionally model-agnostic. It does not know about
ARIMA, XGBoost, or LSTM — only about windows and dates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldPrediction:
    """Output of a single fold's predict() call.

    All arrays are length-h (the horizon for this fold). Intervals are
    optional; pass None when the backing model doesn't expose them.
    """
    yhat: np.ndarray
    lower_80: Optional[np.ndarray] = None
    upper_80: Optional[np.ndarray] = None
    lower_95: Optional[np.ndarray] = None
    upper_95: Optional[np.ndarray] = None


@runtime_checkable
class FittedModel(Protocol):
    """A model that has been fit on a training window and can predict ahead."""

    def predict(self, X_future: Optional[pd.DataFrame], h: int) -> FoldPrediction:
        ...


ModelFactory = Callable[
    [Optional[pd.DataFrame], pd.Series, Optional[np.ndarray]],
    FittedModel,
]
"""Signature: (X_train, y_train, sample_weight) -> FittedModel."""

SampleWeightFn = Callable[[Optional[pd.DataFrame], pd.Series], np.ndarray]


# ---------------------------------------------------------------------------
# The forecaster itself
# ---------------------------------------------------------------------------

class RollingForecaster:
    """Rolling-origin weekly refit driver.

    Parameters
    ----------
    model_factory : Callable
        Function `(X_train, y_train, sample_weight) -> FittedModel`. The
        returned object must expose `predict(X_future, h) -> FoldPrediction`.
        For univariate models (ARIMA), X_train and X_future may be None.
    step_days : int, default 7
        How many days to advance the origin between refits.
    horizon_days : int, default 7
        How many days to forecast at each origin.
    min_train_days : int, default 365
        Smallest acceptable training window. Folds that would shrink the
        train below this are skipped (currently only relevant for
        sliding windows).
    window_days : int or None, default None
        None  -> expanding window (use all data up to origin).
        int N -> sliding window of last N days.
    refit_every : int, default 1
        Refit the base model every k folds; reuse the cached model otherwise.
        1 is the canonical setting used throughout the chapter.
    sample_weight_fn : Callable, optional
        `(X_train, y_train) -> array[n_train]` of per-sample weights, e.g.
        KMM or RuLSIF importance weights (Prompt 7). When None, no weights
        are passed to the factory.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        step_days: int = 7,
        horizon_days: int = 7,
        min_train_days: int = 365,
        window_days: Optional[int] = None,
        refit_every: int = 1,
        sample_weight_fn: Optional[SampleWeightFn] = None,
    ):
        if step_days <= 0 or horizon_days <= 0:
            raise ValueError("step_days and horizon_days must be positive.")
        if refit_every < 1:
            raise ValueError("refit_every must be at least 1.")
        if window_days is not None and window_days < min_train_days:
            raise ValueError(
                f"window_days={window_days} < min_train_days={min_train_days}; "
                "the sliding window cannot be smaller than the train floor."
            )
        self.model_factory = model_factory
        self.step_days = step_days
        self.horizon_days = horizon_days
        self.min_train_days = min_train_days
        self.window_days = window_days
        self.refit_every = refit_every
        self.sample_weight_fn = sample_weight_fn

    # -- internals ----------------------------------------------------------

    def _training_window(self, origin_pos: int) -> tuple[int, int]:
        """Return [start_pos, end_pos] inclusive for the train slice."""
        end_pos = origin_pos
        if self.window_days is None:
            start_pos = 0
        else:
            start_pos = max(0, end_pos - self.window_days + 1)
        return start_pos, end_pos

    # -- main entry point ---------------------------------------------------

    def fit_predict(
        self,
        X: Optional[pd.DataFrame],
        y: pd.Series,
        eval_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Roll the model across the evaluation block and return predictions.

        Parameters
        ----------
        X : DataFrame, indexed by date, columns are features.
            For univariate models pass None.
        y : Series, indexed by date.
        eval_index : DatetimeIndex covering the dates we want predicted.
            Predictions are returned ONLY for dates in this index that the
            model actually produced (the last fold may be truncated to
            horizon_days <= eval block end).

        Returns
        -------
        DataFrame indexed by date with columns
            yhat, lower_80, upper_80, lower_95, upper_95, fold_id
        Intervals are NaN for models that don't expose them.
        """
        if len(eval_index) == 0:
            return pd.DataFrame(
                columns=["yhat", "lower_80", "upper_80",
                          "lower_95", "upper_95", "fold_id"]
            )

        block_start = eval_index[0]
        block_end = eval_index[-1]

        start_loc = y.index.get_loc(block_start)
        end_loc = y.index.get_loc(block_end)
        if not isinstance(start_loc, int) or not isinstance(end_loc, int):
            raise ValueError("eval_index dates must each appear exactly once in y.")

        origin_pos = start_loc - 1
        if origin_pos < 0:
            raise ValueError(
                "First eval date is the first row of y; need at least one "
                "training observation before the eval block begins."
            )

        rows: list[dict] = []
        fitted_model: Optional[FittedModel] = None
        folds_since_refit = self.refit_every  # force first-iteration refit
        fold_id = 0

        while origin_pos < end_loc:
            train_start, train_end = self._training_window(origin_pos)
            n_train = train_end - train_start + 1
            if n_train < self.min_train_days:
                # Slide forward but emit no prediction for this slot.
                origin_pos += self.step_days
                continue

            n_remaining = end_loc - origin_pos
            h = int(min(self.horizon_days, n_remaining))

            X_train = (
                X.iloc[train_start : train_end + 1] if X is not None else None
            )
            y_train = y.iloc[train_start : train_end + 1]
            X_future = (
                X.iloc[origin_pos + 1 : origin_pos + 1 + h]
                if X is not None
                else None
            )
            future_dates = y.index[origin_pos + 1 : origin_pos + 1 + h]

            if folds_since_refit >= self.refit_every or fitted_model is None:
                sample_weight = (
                    self.sample_weight_fn(X_train, y_train)
                    if self.sample_weight_fn is not None
                    else None
                )
                fitted_model = self.model_factory(X_train, y_train, sample_weight)
                folds_since_refit = 0

            prediction = fitted_model.predict(X_future, h)
            self._validate_prediction(prediction, h)

            for i, date in enumerate(future_dates):
                rows.append({
                    "date": date,
                    "yhat": float(prediction.yhat[i]),
                    "lower_80": _safe_get(prediction.lower_80, i),
                    "upper_80": _safe_get(prediction.upper_80, i),
                    "lower_95": _safe_get(prediction.lower_95, i),
                    "upper_95": _safe_get(prediction.upper_95, i),
                    "fold_id": fold_id,
                })

            fold_id += 1
            folds_since_refit += 1
            origin_pos += self.step_days

        df = pd.DataFrame(rows)
        if df.empty:
            return df.set_index(pd.DatetimeIndex([], name="date"))
        return df.set_index("date")

    @staticmethod
    def _validate_prediction(p: FoldPrediction, h: int) -> None:
        if p.yhat is None:
            raise ValueError("FoldPrediction.yhat must not be None.")
        if len(p.yhat) != h:
            raise ValueError(
                f"FoldPrediction.yhat has length {len(p.yhat)} but the fold "
                f"asked for {h} predictions."
            )

    def count_folds(self, y: pd.Series, eval_index: pd.DatetimeIndex) -> int:
        """Diagnostic helper: how many folds will fit_predict produce?"""
        if len(eval_index) == 0:
            return 0
        start_loc = y.index.get_loc(eval_index[0])
        end_loc = y.index.get_loc(eval_index[-1])
        origin_pos = start_loc - 1
        n = 0
        while origin_pos < end_loc:
            train_start, train_end = self._training_window(origin_pos)
            if (train_end - train_start + 1) >= self.min_train_days:
                n += 1
            origin_pos += self.step_days
        return n


def _safe_get(arr: Optional[np.ndarray], i: int) -> float:
    if arr is None:
        return float("nan")
    return float(arr[i])


# ---------------------------------------------------------------------------
# Convenience factory builders for the simple model families.
#
# These are pure helpers so the per-model wrappers stay 5-10 lines.
# Heavier models (ANN, LSTM) build their own factory closures inline
# because they need standardisation + sequence machinery.
# ---------------------------------------------------------------------------

def make_arima_factory(order: tuple[int, int, int], alpha: float = 0.05):
    """Factory for pmdarima.ARIMA univariate models (no exogenous)."""
    from pmdarima import ARIMA as PmARIMA

    def factory(X_train, y_train, sample_weight=None):
        if sample_weight is not None:
            # pmdarima ARIMA does not support sample weights; warn-by-ignore.
            pass
        model = PmARIMA(order=order, suppress_warnings=True)
        model.fit(y_train.values)

        class _Fitted:
            def predict(self, X_future, h):
                yhat, conf = model.predict(
                    n_periods=h, return_conf_int=True, alpha=alpha,
                )
                return FoldPrediction(
                    yhat=np.asarray(yhat, dtype=float),
                    lower_95=np.asarray(conf[:, 0], dtype=float),
                    upper_95=np.asarray(conf[:, 1], dtype=float),
                )

        return _Fitted()

    return factory


def make_sarimax_factory(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    alpha: float = 0.05,
):
    """Factory for pmdarima.ARIMA SARIMAX (with exogenous)."""
    from pmdarima import ARIMA as PmARIMA

    def factory(X_train, y_train, sample_weight=None):
        if X_train is None:
            raise ValueError("SARIMAX factory requires X_train (exogenous).")
        model = PmARIMA(
            order=order, seasonal_order=seasonal_order, suppress_warnings=True,
        )
        model.fit(y_train.values, X=X_train.values)

        class _Fitted:
            def predict(self, X_future, h):
                if X_future is None:
                    raise ValueError("SARIMAX prediction needs X_future.")
                yhat, conf = model.predict(
                    n_periods=h, X=X_future.values,
                    return_conf_int=True, alpha=alpha,
                )
                return FoldPrediction(
                    yhat=np.asarray(yhat, dtype=float),
                    lower_95=np.asarray(conf[:, 0], dtype=float),
                    upper_95=np.asarray(conf[:, 1], dtype=float),
                )

        return _Fitted()

    return factory


def make_xgboost_factory(params: dict, seed: int = 42):
    """Factory for XGBoost regression."""
    from xgboost import XGBRegressor

    def factory(X_train, y_train, sample_weight=None):
        if X_train is None:
            raise ValueError("XGBoost factory requires X_train.")
        model = XGBRegressor(
            **params, objective="reg:squarederror",
            random_state=seed, verbosity=0, n_jobs=-1,
        )
        model.fit(
            X_train.values, y_train.values,
            sample_weight=(sample_weight if sample_weight is not None else None),
        )

        class _Fitted:
            def predict(self, X_future, h):
                yhat = model.predict(X_future.values)
                return FoldPrediction(yhat=np.asarray(yhat, dtype=float))

        return _Fitted()

    return factory


__all__ = [
    "FoldPrediction",
    "FittedModel",
    "ModelFactory",
    "SampleWeightFn",
    "RollingForecaster",
    "make_arima_factory",
    "make_sarimax_factory",
    "make_xgboost_factory",
]
