"""ARIMA baseline per §3.5.2 Algorithm 2.

No exogenous regressors (those go in SARIMAX / NB GLM). Order picked by stepwise
auto_arima with d=1 fixed (per §5.2.2 ADF/KPSS). Forecast strategy: rolling-origin
weekly refit — pick order once on the initial train fit, then re-estimate
parameters of that fixed order at each weekly origin and forecast 7 days.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from pmdarima import ARIMA, auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


@dataclass
class ArimaResult:
    order: tuple[int, int, int]
    aic: float
    fitted_train: "auto_arima"  # the pmdarima ARIMA fitted on initial train


def pick_order(train_series: pd.Series, max_p: int = 3, max_q: int = 3,
               d: int = 1, seed: int = 42) -> ArimaResult:
    """Pick (p, d, q) via stepwise AIC, d fixed by §5.2.2."""
    np.random.seed(seed)
    model = auto_arima(
        train_series.values,
        start_p=0, start_q=0,
        max_p=max_p, max_q=max_q,
        d=d,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
        trace=False,
        random_state=seed,
    )
    return ArimaResult(order=model.order, aic=float(model.aic()), fitted_train=model)


def residual_diagnostics(residuals: np.ndarray) -> dict:
    """Ljung-Box at 7/14/21 and Jarque-Bera (§7.4)."""
    lb = acorr_ljungbox(residuals, lags=[7, 14, 21], return_df=True)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
    return {
        "lb_stat_lag7": float(lb.loc[7, "lb_stat"]),
        "lb_pvalue_lag7": float(lb.loc[7, "lb_pvalue"]),
        "lb_stat_lag14": float(lb.loc[14, "lb_stat"]),
        "lb_pvalue_lag14": float(lb.loc[14, "lb_pvalue"]),
        "lb_stat_lag21": float(lb.loc[21, "lb_stat"]),
        "lb_pvalue_lag21": float(lb.loc[21, "lb_pvalue"]),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_p),
        "jb_skew": float(jb_skew),
        "jb_kurtosis": float(jb_kurt),
    }


def rolling_forecast(
    full_series: pd.Series,
    block_index: pd.DatetimeIndex,
    order: tuple[int, int, int],
    step_days: int = 7,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rolling-origin weekly refit forecast.

    For each origin t in block_index spaced by step_days, refit ARIMA(order)
    on full_series up to t-1 and forecast the next step_days days.
    """
    rows = []
    block_start = block_index[0]
    block_end = block_index[-1]

    # Origin = the last training day; forecast covers origin+1 .. origin+step.
    # First origin is the day before block_start.
    origin_pos = full_series.index.get_loc(block_start) - 1
    while origin_pos < full_series.index.get_loc(block_end):
        train_through = full_series.iloc[: origin_pos + 1]
        n_remaining = full_series.index.get_loc(block_end) - origin_pos
        h = int(min(step_days, n_remaining))
        model = ARIMA(order=order, suppress_warnings=True)
        model.fit(train_through.values)
        yhat, conf = model.predict(n_periods=h, return_conf_int=True, alpha=alpha)
        dates = full_series.index[origin_pos + 1 : origin_pos + 1 + h]
        for date, y_pred, lo, hi in zip(dates, yhat, conf[:, 0], conf[:, 1]):
            rows.append({
                "date": date,
                "predicted": float(y_pred),
                "lower_95": float(lo),
                "upper_95": float(hi),
            })
        origin_pos += step_days

    df = pd.DataFrame(rows)
    return df
