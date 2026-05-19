"""SARIMAX baseline per Ch3 §3.5.2 + Ch5 §5.2.2 + §5.7.

SARIMAX = Seasonal ARIMA with eXogenous regressors. Order template
(p, 1, q)(P, 1, Q)_7 with p, q, P, Q in {0, 1, 2}, d=D=1 fixed, AIC-selected.
Gaussian likelihood — pairs with the NB GLM (§8.4) as the two parallel
parametric baselines mandated by Ch5 §5.7.

Exogenous block: §5.2.5 raw 10 (built in features.build_task1_exogenous).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pmdarima import ARIMA as PmARIMA, auto_arima


@dataclass
class SarimaxResult:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float
    fitted_train: object   # pmdarima ARIMA fitted on initial train (with exog)


def pick_order(
    train_series: pd.Series,
    train_exog: pd.DataFrame,
    max_p: int = 2, max_q: int = 2,
    max_P: int = 2, max_Q: int = 2,
    m: int = 7,
    seed: int = 42,
) -> SarimaxResult:
    """Pick (p, 1, q)(P, 1, Q)_m via stepwise AIC, d=D=1 fixed (§5.2.2)."""
    np.random.seed(seed)
    model = auto_arima(
        train_series.values,
        X=train_exog.values,
        start_p=0, start_q=0,
        max_p=max_p, max_q=max_q,
        d=1,
        seasonal=True, m=m,
        start_P=0, start_Q=0,
        max_P=max_P, max_Q=max_Q,
        D=1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
        trace=False,
        random_state=seed,
    )
    return SarimaxResult(
        order=model.order,
        seasonal_order=model.seasonal_order,
        aic=float(model.aic()),
        fitted_train=model,
    )


def fit_with_order(
    train_series: pd.Series,
    train_exog: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> SarimaxResult:
    """Fit a SARIMAX with a known order (skips the auto_arima search)."""
    model = PmARIMA(order=order, seasonal_order=seasonal_order,
                    suppress_warnings=True)
    model.fit(train_series.values, X=train_exog.values)
    return SarimaxResult(
        order=order,
        seasonal_order=seasonal_order,
        aic=float(model.aic()),
        fitted_train=model,
    )


def rolling_forecast(
    full_series: pd.Series,
    full_exog: pd.DataFrame,
    block_index: pd.DatetimeIndex,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    step_days: int = 7,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rolling-origin weekly refit forecast with exogenous regressors."""
    rows = []
    block_start = block_index[0]
    block_end = block_index[-1]

    origin_pos = full_series.index.get_loc(block_start) - 1
    while origin_pos < full_series.index.get_loc(block_end):
        y_train = full_series.iloc[: origin_pos + 1].values
        X_train = full_exog.iloc[: origin_pos + 1].values
        n_remaining = full_series.index.get_loc(block_end) - origin_pos
        h = int(min(step_days, n_remaining))
        X_future = full_exog.iloc[origin_pos + 1 : origin_pos + 1 + h].values

        model = PmARIMA(order=order, seasonal_order=seasonal_order,
                        suppress_warnings=True)
        model.fit(y_train, X=X_train)
        yhat, conf = model.predict(n_periods=h, X=X_future,
                                    return_conf_int=True, alpha=alpha)

        dates = full_series.index[origin_pos + 1 : origin_pos + 1 + h]
        for date, y_pred, lo, hi in zip(dates, yhat, conf[:, 0], conf[:, 1]):
            rows.append({
                "date": date,
                "predicted": float(y_pred),
                "lower_95": float(lo),
                "upper_95": float(hi),
            })
        origin_pos += step_days

    return pd.DataFrame(rows)


def extract_coefficients(fitted_train, exog_cols: list[str]) -> pd.DataFrame:
    """Pull exogenous coefficients + std errors + p-values.

    pmdarima wraps statsmodels SARIMAX. When X is passed as a numpy array,
    statsmodels names exog params 'x1', 'x2', ... in order. We map those
    back to exog_cols. AR/MA/seasonal/sigma2 params are skipped.
    """
    sm_res = fitted_train.arima_res_
    param_names = list(sm_res.param_names)
    params = np.asarray(sm_res.params)
    bse = np.asarray(sm_res.bse)
    pvals = np.asarray(sm_res.pvalues)

    rows = []
    for i, name in enumerate(param_names):
        if name.startswith("x") and name[1:].isdigit():
            try:
                feature = exog_cols[int(name[1:]) - 1]
            except (ValueError, IndexError):
                feature = name
            rows.append({
                "feature": feature,
                "coef": float(params[i]),
                "std_err": float(bse[i]),
                "p_value": float(pvals[i]),
            })
        elif name in exog_cols:
            rows.append({
                "feature": name,
                "coef": float(params[i]),
                "std_err": float(bse[i]),
                "p_value": float(pvals[i]),
            })
    return pd.DataFrame(rows)
