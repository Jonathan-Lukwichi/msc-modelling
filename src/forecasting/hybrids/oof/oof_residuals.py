"""Out-of-fold residuals + statistically-honest residual hybrids.

Replaces the Zhang (2003) in-sample residual recipe used in
hybrids/residual.py. The in-sample approach exhibits a selection bias
documented by:

  - Khashei & Bijari (2011) Applied Soft Computing 11(2):2664-2675
    -- showed that residuals computed on the fold that trained the base
       model under-state the residual variance, biasing the refiner.
  - Hewamalage, Bergmeir & Bandara (2021) IJF 37(1):388-427
    -- methodological critique of how hybrid ML/DL forecasters are
       commonly evaluated; recommends out-of-sample residual generation.
  - Smyl (2020) IJF 36(1):75-85 (ES-RNN, M4 winner) -- demonstrates the
    correct hybrid recipe is JOINT estimation, but when residual stacking
    is preferred for interpretability, residuals must be out-of-sample.

This module implements:

  1. ``build_oof_residuals(base_factory, X_train, y_train, n_folds, horizon)``
     -- runs RollingForecaster on the train block to produce honest
        residuals (y - yhat_oof) where yhat_oof is the prediction from a
        base model that NEVER saw that day in its training set.

  2. ``OOFResidualHybrid`` -- composes a base model factory and a refiner
     factory. Trains the refiner on OOF residuals, optionally with nested
     HPO over a small Optuna budget. The refiner standardises residuals
     consistently (fixes the XGB-vs-LSTM inconsistency in the legacy
     residual.py).

Empirical motivation from this project: RESULTS.md §4sexies shows LSTM+XGB
val MAPE 13.50 > LSTM-alone 12.74 -- the exact bias signature that OOF
residuals are designed to remove.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.forecasting.rolling import RollingForecaster, ModelFactory


# ---------------------------------------------------------------------------
# OOF residual generation
# ---------------------------------------------------------------------------

def build_oof_residuals(
    base_factory: ModelFactory,
    X_train: Optional[pd.DataFrame],
    y_train: pd.Series,
    horizon: int = 7,
    step: int = 7,
    min_train_days: int = 365,
) -> pd.DataFrame:
    """Out-of-fold residuals for the train block.

    Walks ``RollingForecaster`` with ``window_days=None`` (expanding) over
    y_train itself, exposing each interior week's predictions as
    forecasts from a base fitted on prior weeks only.

    Returns
    -------
    DataFrame indexed by date with columns:
        y_true, yhat_oof, residual (= y_true - yhat_oof), fold_id
    The first ``min_train_days`` days of y_train do not appear: they are
    the seed training set.
    """
    eval_idx = y_train.index[min_train_days:]  # everything past the seed
    rf = RollingForecaster(
        model_factory=base_factory,
        step_days=step,
        horizon_days=horizon,
        min_train_days=min_train_days,
    )
    preds = rf.fit_predict(X=X_train, y=y_train, eval_index=eval_idx)
    if preds.empty:
        return pd.DataFrame(
            columns=["y_true", "yhat_oof", "residual", "fold_id"]
        )
    out = preds[["yhat", "fold_id"]].copy().rename(columns={"yhat": "yhat_oof"})
    out["y_true"] = y_train.loc[out.index]
    out["residual"] = out["y_true"] - out["yhat_oof"]
    return out[["y_true", "yhat_oof", "residual", "fold_id"]]


# ---------------------------------------------------------------------------
# The hybrid
# ---------------------------------------------------------------------------

RefinerFactory = Callable[[pd.DataFrame, pd.Series], "FittedRefiner"]


class FittedRefiner:
    """Protocol: refiner exposes predict(X) -> ndarray."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class OOFResidualHybrid:
    """Composes (base_factory, refiner_factory) into a statistically-honest
    residual hybrid.

    Workflow on ``fit(X_train, y_train)``:
      1. ``build_oof_residuals(base_factory, X_train, y_train)`` -> OOF
         residuals series indexed by dates in [min_train_days .. end].
      2. Optionally z-score residuals (default True).
      3. If ``nested_hpo`` and ``refiner_hpo_space`` provided, Optuna TPE
         over the space with mean MAPE over 5 inner rolling folds (also
         using only OOF residual dates) as objective.
      4. Fit the chosen refiner on (X_oof, residuals_oof).

    Workflow on ``predict(X_eval, eval_index)``:
      - Refit the BASE on X_train + y_train (full train) and forecast over
        eval_index via RollingForecaster.
      - Predict refiner correction on X_eval.
      - Return base_yhat + refiner_pred * residual_std + residual_mean
        (when standardise_residuals).

    Notes
    -----
    Hyperparameter inheritance bug from RESULTS.md §4sexies (the refiner
    was reusing the base's HPO winner) is eliminated by construction:
    refiner_factory must be passed independently and is HPO-tuned only
    inside this class on residuals.
    """
    base_factory: ModelFactory
    refiner_factory: RefinerFactory
    standardize_residuals: bool = True
    nested_hpo: bool = False
    refiner_hpo_space: Optional[dict] = None
    n_inner_folds: int = 5
    n_hpo_trials: int = 8
    min_train_days: int = 365
    horizon: int = 7
    seed: int = 42

    # Populated by fit():
    _oof: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _res_mean: float = field(default=0.0, init=False, repr=False)
    _res_std: float = field(default=1.0, init=False, repr=False)
    _refiner: Optional[FittedRefiner] = field(default=None, init=False, repr=False)
    _best_refiner_params: Optional[dict] = field(default=None, init=False, repr=False)
    _X_train: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _y_train: Optional[pd.Series] = field(default=None, init=False, repr=False)

    # -- fit ---------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "OOFResidualHybrid":
        self._X_train = X_train
        self._y_train = y_train
        oof = build_oof_residuals(
            self.base_factory, X_train, y_train,
            horizon=self.horizon, step=self.horizon,
            min_train_days=self.min_train_days,
        )
        self._oof = oof
        if oof.empty:
            raise ValueError(
                "build_oof_residuals returned no rows; check min_train_days."
            )

        r = oof["residual"].values.astype(float)
        if self.standardize_residuals:
            self._res_mean = float(np.mean(r))
            self._res_std = float(np.std(r, ddof=0) or 1.0)
        r_std = (r - self._res_mean) / self._res_std

        # Align X to OOF residual dates
        X_oof = X_train.loc[oof.index]
        r_series = pd.Series(r_std, index=oof.index, name="residual")

        if self.nested_hpo and self.refiner_hpo_space:
            self._best_refiner_params = self._tune_refiner(X_oof, r_series)
        else:
            self._best_refiner_params = None

        # Build factory with chosen params (or default).
        factory_kwargs = self._best_refiner_params or {}
        self._refiner = self.refiner_factory(X_oof, r_series, **factory_kwargs)
        return self

    # -- predict -----------------------------------------------------------

    def predict(
        self,
        X_eval: pd.DataFrame,
        eval_index: pd.DatetimeIndex,
        y_eval: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return DataFrame indexed by eval dates with columns:
            base_yhat, refiner_pred, predicted (= base + refiner correction)

        Parameters
        ----------
        X_eval : exogenous features over the eval block.
        eval_index : dates to forecast.
        y_eval : observed targets over the eval block, required for the
            weekly rolling refit of the base model (the refit at origin
            t > end_of_train consumes y observed up to t). In a true
            production deployment these are the values that have just
            been logged as the new week began. Passing None falls back
            to a zero-imputed series, which is only valid if the eval
            block is at most ``self.horizon`` days long.
        """
        if self._refiner is None:
            raise RuntimeError("fit() must be called before predict().")

        # Concatenate train + eval for rolling forecast continuity
        if X_eval is None:
            X_full = self._X_train
        else:
            X_full = pd.concat([self._X_train, X_eval]).loc[
                lambda df: ~df.index.duplicated(keep="first")
            ]
        y_full = self._y_train
        # Splice in observed y over the eval block when provided. Without
        # this the rolling refit sees NaNs once origin enters eval, which
        # silently corrupts every parametric base (SARIMAX, ARIMA) and
        # cascades catastrophically -- the bug surfaced during the
        # OOF hybrid orchestrator run on 2026-05-21.
        if y_eval is not None:
            y_full = pd.concat([y_full, y_eval.loc[eval_index]]).loc[
                lambda s: ~s.index.duplicated(keep="last")
            ].sort_index()
        else:
            missing_dates = eval_index.difference(y_full.index)
            if len(missing_dates):
                pad = pd.Series(0.0, index=missing_dates)
                y_full = pd.concat([y_full, pad]).sort_index()

        rf = RollingForecaster(
            model_factory=self.base_factory,
            step_days=self.horizon, horizon_days=self.horizon,
            min_train_days=1,
        )
        base = rf.fit_predict(X=X_full, y=y_full, eval_index=eval_index)
        base_yhat = base["yhat"].copy()

        # Refiner prediction
        X_for_refiner = (
            X_eval.loc[eval_index] if X_eval is not None
            else pd.DataFrame(index=eval_index)
        )
        r_pred_std = self._refiner.predict(X_for_refiner)
        r_pred = r_pred_std * self._res_std + self._res_mean

        out = pd.DataFrame({
            "base_yhat": base_yhat.values,
            "refiner_pred": np.asarray(r_pred, dtype=float),
        }, index=base_yhat.index)
        out["predicted"] = out["base_yhat"] + out["refiner_pred"]
        return out

    # -- HPO ---------------------------------------------------------------

    def _tune_refiner(self, X_oof: pd.DataFrame, r_series: pd.Series) -> dict:
        """Nested Optuna TPE over self.refiner_hpo_space on inner rolling folds."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            return {}

        from src.forecasting.cv import subsampled_rolling_origin
        folds = subsampled_rolling_origin(
            X_oof.index, n_folds=self.n_inner_folds,
            horizon_days=self.horizon, step_days=self.horizon,
            min_train_days=max(60, len(X_oof) // 4),
        )
        if not folds:
            return {}

        def objective(trial: "optuna.Trial") -> float:
            params = {}
            for k, spec in self.refiner_hpo_space.items():
                if isinstance(spec, list):
                    params[k] = trial.suggest_categorical(k, spec)
                elif isinstance(spec, tuple) and len(spec) == 2:
                    lo, hi = spec
                    params[k] = trial.suggest_float(k, lo, hi)
            fold_errors = []
            for fold in folds:
                X_tr = X_oof.iloc[fold.train_idx]
                r_tr = r_series.iloc[fold.train_idx]
                X_te = X_oof.iloc[fold.test_idx]
                r_te = r_series.iloc[fold.test_idx]
                refiner = self.refiner_factory(X_tr, r_tr, **params)
                preds = np.asarray(refiner.predict(X_te))
                fold_errors.append(
                    float(np.sqrt(np.mean((r_te.values - preds) ** 2)))
                )
            return float(np.mean(fold_errors))

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=self.n_hpo_trials,
                        show_progress_bar=False)
        return dict(study.best_params)

    # -- diagnostic --------------------------------------------------------

    @property
    def oof_residuals(self) -> pd.DataFrame:
        if self._oof is None:
            raise RuntimeError("fit() has not been called.")
        return self._oof.copy()

    @property
    def best_refiner_params(self) -> Optional[dict]:
        return self._best_refiner_params


# ---------------------------------------------------------------------------
# Convenience refiner factories (XGBoost, simple sklearn-style)
# ---------------------------------------------------------------------------

def xgb_refiner_factory(
    X_train: pd.DataFrame, residuals: pd.Series, **params,
) -> FittedRefiner:
    """Default XGB refiner. **params override the sane-default values below."""
    from xgboost import XGBRegressor
    seed = params.pop("seed", 42)
    defaults = {"n_estimators": 200, "max_depth": 4,
                  "learning_rate": 0.05, "subsample": 0.85}
    defaults.update(params)
    model = XGBRegressor(
        **defaults, objective="reg:squarederror",
        random_state=seed, verbosity=0, n_jobs=-1,
    )
    common = X_train.index.intersection(residuals.index)
    model.fit(X_train.loc[common].values, residuals.loc[common].values)

    class _Fitted(FittedRefiner):
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X.values)
    return _Fitted()


__all__ = [
    "build_oof_residuals",
    "OOFResidualHybrid",
    "FittedRefiner",
    "RefinerFactory",
    "xgb_refiner_factory",
]
