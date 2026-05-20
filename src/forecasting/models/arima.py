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
    """Rolling-origin weekly refit forecast — thin wrapper over RollingForecaster.

    Backward-compatible signature and output schema. The shared iteration
    logic lives in src.forecasting.rolling.RollingForecaster (Prompt 1).
    """
    from src.forecasting.rolling import RollingForecaster, make_arima_factory

    rf = RollingForecaster(
        model_factory=make_arima_factory(order=order, alpha=alpha),
        step_days=step_days, horizon_days=step_days,
        min_train_days=1,  # ARIMA only needs one prior observation
    )
    out = rf.fit_predict(X=None, y=full_series, eval_index=block_index)
    return out.reset_index().rename(columns={"yhat": "predicted"})[
        ["date", "predicted", "lower_95", "upper_95"]
    ]
