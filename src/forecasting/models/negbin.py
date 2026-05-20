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
    """Rolling-origin weekly refit, h<=7 (lag-7 stays in observed history).

    Thin wrapper over RollingForecaster. The lag-7 regressor is built
    inside the per-fold factory closure so it sees the train slice that
    RollingForecaster is currently exposing.
    """
    from src.forecasting.rolling import RollingForecaster, FoldPrediction

    # The lag-7 column must be aligned to full dates so the eval slice
    # references the correct shifted values during prediction.
    full_lag7 = full_series.shift(7).rename("y_lag7")
    full_exog_with_lag = pd.concat([full_exog, full_lag7], axis=1)

    def factory(X_train, y_train, sample_weight=None):
        # Drop rows where lag-7 isn't yet defined (first 7 train days).
        df = pd.concat([y_train.rename("y"), X_train], axis=1).dropna()
        y = df["y"].values
        X = df.drop(columns=["y"]).values
        X_c = sm.add_constant(X)
        alpha = estimate_alpha(y, X_c)
        nb_fit = sm.GLM(
            y, X_c, family=sm.families.NegativeBinomial(alpha=alpha)
        ).fit(method="lbfgs", maxiter=200, disp=0)
        n_train_cols = X_c.shape[1]

        class _Fitted:
            def predict(self, X_future, h):
                X_f = X_future.values
                X_f_c = sm.add_constant(X_f, has_constant="add")
                if X_f_c.shape[1] != n_train_cols:
                    X_f_c = np.column_stack([np.ones(len(X_f)), X_f])
                mu = nb_fit.predict(X_f_c)
                n = 1.0 / max(alpha, alpha_floor)
                p_param = n / (n + mu)
                lo = stats.nbinom.ppf(0.025, n, p_param)
                hi = stats.nbinom.ppf(0.975, n, p_param)
                self.alpha = alpha
                return FoldPrediction(
                    yhat=np.asarray(mu, dtype=float),
                    lower_95=np.asarray(lo, dtype=float),
                    upper_95=np.asarray(hi, dtype=float),
                )

        fitted = _Fitted()
        fitted.alpha = alpha
        return fitted

    rf = RollingForecaster(
        model_factory=factory,
        step_days=step_days, horizon_days=step_days, min_train_days=8,
    )
    out = rf.fit_predict(X=full_exog_with_lag, y=full_series,
                          eval_index=block_index)
    df = out.reset_index().rename(columns={"yhat": "predicted"})
    # Alpha is re-estimated per fold; for the legacy column we recompute via
    # a quick scan (each unique fold_id had a single alpha emitted by the
    # closure). The factory mutates a side-effect _Fitted.alpha, but we
    # don't have access to it post hoc here; the alpha column in the legacy
    # output is informational, not used downstream. Leave NaN for now.
    df["alpha"] = np.nan
    return df[["date", "predicted", "lower_95", "upper_95", "alpha"]]


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
