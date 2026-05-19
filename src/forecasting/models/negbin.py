"""Negative Binomial GLM regression baseline per Ch5 §5.7 + §5.2.1.

Headline parametric likelihood. Pairs with SARIMAX (§8) as the two parallel
parametric baselines mandated by §5.7 ("a Negative Binomial regression
alongside a SARIMA baseline" — verbatim from §5.7; the SARIMA there is what
we call SARIMAX because it carries the §5.2.5 exogenous block).

Design:
  - Log-link NB GLM with §5.2.5 exogenous block + y_{t-7} autoregressive control.
    The lag-7 control is motivated by §5.2.2 (lag-7 ACF = 0.490) and keeps the
    model independent from the SARIMAX AR/MA dynamics so the two baselines
    contribute distinct information.
  - Dispersion alpha estimated via Poisson pre-fit + Pearson chi^2 / df_resid.
  - Rolling-origin weekly refit, h=7 (lag-7 from observed history at each origin).
  - 95% prediction intervals via the NB pmf (overdispersion-aware).

Normal-likelihood GLM is fit alongside as the §5.2.1 sensitivity (AIC delta).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class NbGlmResult:
    alpha: float
    aic_nb: float
    aic_normal: Optional[float]
    fitted_train: sm.regression.linear_model.RegressionResults


def estimate_alpha(y: np.ndarray, X_with_const: np.ndarray) -> float:
    """Estimate NB dispersion alpha via Poisson pre-fit + Pearson chi^2.

    Standard procedure: fit Poisson, compute Pearson residuals
    (y - mu) / sqrt(mu), then alpha = chi^2 / df_resid - 1, clipped to >= 0.05.
    """
    poisson_fit = sm.GLM(y, X_with_const, family=sm.families.Poisson()).fit(
        method="lbfgs", maxiter=100, disp=0,
    )
    mu = poisson_fit.predict()
    pearson = (y - mu) / np.sqrt(np.maximum(mu, 1e-6))
    chi2 = float(np.sum(pearson ** 2))
    df = max(len(y) - X_with_const.shape[1], 1)
    alpha = chi2 / df - 1.0
    return float(max(alpha, 0.05))


def fit_with_lag7(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    fit_normal_sensitivity: bool = True,
) -> NbGlmResult:
    """Fit NB GLM and (optionally) Normal GLM on the training fold.

    y_train is expected to be aligned to X_train. Lag-7 column is added inside.
    """
    df = pd.concat([y_train.rename("y"), X_train], axis=1)
    df["y_lag7"] = df["y"].shift(7)
    df = df.dropna()
    y = df["y"].values
    X = df.drop(columns=["y"]).values
    X_const = sm.add_constant(X)

    alpha = estimate_alpha(y, X_const)
    nb_fit = sm.GLM(y, X_const, family=sm.families.NegativeBinomial(alpha=alpha)).fit(
        method="lbfgs", maxiter=200, disp=0,
    )

    normal_fit = None
    if fit_normal_sensitivity:
        normal_fit = sm.GLM(y, X_const, family=sm.families.Gaussian()).fit(disp=0)

    return NbGlmResult(
        alpha=alpha,
        aic_nb=float(nb_fit.aic),
        aic_normal=float(normal_fit.aic) if normal_fit is not None else None,
        fitted_train=nb_fit,
    )


def rolling_forecast(
    full_series: pd.Series,
    full_exog: pd.DataFrame,
    block_index: pd.DatetimeIndex,
    step_days: int = 7,
    alpha_floor: float = 0.05,
) -> pd.DataFrame:
    """Rolling-origin weekly refit, h=7 (lag-7 stays in observed history)."""
    rows = []
    block_start = block_index[0]
    block_end = block_index[-1]

    full_with_lag = pd.concat(
        [full_series.rename("y"),
         full_exog,
         full_series.shift(7).rename("y_lag7")],
        axis=1,
    )

    origin_pos = full_series.index.get_loc(block_start) - 1
    while origin_pos < full_series.index.get_loc(block_end):
        train_df = full_with_lag.iloc[: origin_pos + 1].dropna()
        y_train = train_df["y"].values
        X_train = train_df.drop(columns=["y"]).values
        X_train_c = sm.add_constant(X_train)

        alpha = estimate_alpha(y_train, X_train_c)
        nb_fit = sm.GLM(
            y_train, X_train_c, family=sm.families.NegativeBinomial(alpha=alpha)
        ).fit(method="lbfgs", maxiter=200, disp=0)

        n_remaining = full_series.index.get_loc(block_end) - origin_pos
        h = int(min(step_days, n_remaining))
        future_idx = full_series.index[origin_pos + 1 : origin_pos + 1 + h]
        X_future = full_with_lag.loc[future_idx].drop(columns=["y"]).values
        X_future_c = sm.add_constant(X_future, has_constant="add")
        # Some folds end up with no constant added when the input is rank-deficient;
        # patch by prepending ones.
        if X_future_c.shape[1] != X_train_c.shape[1]:
            X_future_c = np.column_stack([np.ones(len(X_future)), X_future])

        mu = nb_fit.predict(X_future_c)
        # NB(mean=mu, alpha) -> variance = mu + alpha * mu^2
        # Use scipy.stats.nbinom with reparam: n = 1/alpha, p = n/(n+mu)
        n = 1.0 / max(alpha, alpha_floor)
        p_param = n / (n + mu)
        lo = stats.nbinom.ppf(0.025, n, p_param)
        hi = stats.nbinom.ppf(0.975, n, p_param)

        for date, y_pred, lo_i, hi_i in zip(future_idx, mu, lo, hi):
            rows.append({
                "date": date,
                "predicted": float(y_pred),
                "lower_95": float(lo_i),
                "upper_95": float(hi_i),
                "alpha": float(alpha),
            })
        origin_pos += step_days

    return pd.DataFrame(rows)


def extract_coefficients(fitted_train: sm.regression.linear_model.RegressionResults,
                          feature_names: list[str]) -> pd.DataFrame:
    """Return coef / std_err / p_value / IRR per exogenous feature."""
    params = fitted_train.params
    bse = fitted_train.bse
    pvals = fitted_train.pvalues
    # The first param is the const; rest map to feature_names in order
    rows = [{
        "feature": "const",
        "coef": float(params[0]),
        "std_err": float(bse[0]),
        "p_value": float(pvals[0]),
        "IRR": float(np.exp(params[0])),
    }]
    for i, name in enumerate(feature_names, start=1):
        rows.append({
            "feature": name,
            "coef": float(params[i]),
            "std_err": float(bse[i]),
            "p_value": float(pvals[i]),
            "IRR": float(np.exp(params[i])),
        })
    return pd.DataFrame(rows)
