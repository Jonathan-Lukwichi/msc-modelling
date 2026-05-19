"""Deployment package: save/load fitted models for cloud-app reuse.

Each model is saved as a single pickle (.pkl) that bundles everything a cloud
app needs to make a prediction:

  - The fitted model object (or PyTorch state-dict + class info for ANN/LSTM)
  - Feature column names (in fitting order)
  - Feature scaler (mean, std) if the model needs scaled input
  - Target scaler (mean, std) for neural-net models
  - Lookback length (for sequence models)
  - Best hyperparameters
  - Metadata: trained-on dates, val metrics, package version

The cloud app loads with `load_package(path)` and predicts via the returned
Predictor object's `.predict(X)` method. The Predictor handles all
preprocessing internally; the app only needs to supply a raw feature
DataFrame with the same columns the model was trained on.

PyTorch caveat: the cloud environment MUST have `src/forecasting/models/*.py`
on its PYTHONPATH so the `_MLP` and `_LSTMNet` class definitions are
importable. The pickle records the import path; missing modules raise a
clean ImportError at load time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable
import json
import pickle

import numpy as np
import pandas as pd


PACKAGE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Package container
# ---------------------------------------------------------------------------

@dataclass
class ModelPackage:
    """Everything a cloud app needs to make predictions with a trained model."""
    name: str
    family: str
    fitted: Any                              # the trained model object / state
    feature_names: list[str] = field(default_factory=list)
    feature_scaler: Optional[dict] = None    # {'mean': pd.Series, 'std': pd.Series}
    target_scaler: Optional[dict] = None     # {'mean': float, 'std': float}
    lookback: Optional[int] = None
    best_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "family": self.family,
            "fitted": self.fitted,
            "feature_names": self.feature_names,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "lookback": self.lookback,
            "best_params": self.best_params,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_package(pkg: ModelPackage, path: str | Path) -> None:
    """Pickle a ModelPackage to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pkg.metadata.setdefault("package_version", PACKAGE_VERSION)
    pkg.metadata.setdefault("saved_at", datetime.utcnow().isoformat())
    with open(path, "wb") as f:
        pickle.dump(pkg.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_package(path: str | Path) -> ModelPackage:
    """Load a ModelPackage from disk."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return ModelPackage(
        name=d["name"], family=d["family"], fitted=d["fitted"],
        feature_names=d.get("feature_names", []),
        feature_scaler=d.get("feature_scaler"),
        target_scaler=d.get("target_scaler"),
        lookback=d.get("lookback"),
        best_params=d.get("best_params", {}),
        metadata=d.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Predictors — wrap a loaded package and expose a uniform .predict() interface
# ---------------------------------------------------------------------------

class Predictor:
    """Base class. Subclasses know how to call the wrapped model.

    Standard interface for the cloud app:
        predictor = make_predictor(load_package('arima.pkl'))
        forecast = predictor.predict(X_future)
    """
    def __init__(self, pkg: ModelPackage):
        self.pkg = pkg

    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        raise NotImplementedError


class NaivePredictor(Predictor):
    """Naive yest / seasonal / DoW-mean — needs a history series, not features."""
    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        name = self.pkg.name
        if history is None:
            raise ValueError("NaivePredictor needs the recent history series")
        if name == "naive_yest":
            return pd.Series([float(history.iloc[-1])] * len(X), index=X.index)
        if name == "naive_seasonal":
            tail = history.iloc[-7:].values
            n_repeats = (len(X) + 6) // 7
            tiled = np.tile(tail, n_repeats)[: len(X)]
            return pd.Series(tiled, index=X.index)
        if name == "dow_mean":
            weekday_means = self.pkg.fitted   # dict {0..6: mean}
            yhat = pd.Series(
                [weekday_means[int(d.dayofweek)] for d in X.index],
                index=X.index,
            )
            return yhat
        raise ValueError(f"Unknown naive sub-name: {name}")


class ArimaPredictor(Predictor):
    """Plain ARIMA (no exogenous) — supplies h-step forecast."""
    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        model = self.pkg.fitted
        n = len(X)
        yhat = model.predict(n_periods=n)
        return pd.Series(np.asarray(yhat), index=X.index)


class SarimaxPredictor(Predictor):
    """SARIMAX with §5.2.5 exogenous block. X must contain those columns."""
    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pkg.feature_scaler is None:
            return X
        out = X.copy()
        mean = self.pkg.feature_scaler["mean"]
        std = self.pkg.feature_scaler["std"]
        for c in mean.index:
            if c in out.columns:
                out[c] = (out[c] - mean[c]) / std[c]
        return out

    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        model = self.pkg.fitted
        X = self._apply_scaler(X)
        X = X[self.pkg.feature_names]
        yhat = model.predict(n_periods=len(X), X=X.values)
        return pd.Series(np.asarray(yhat), index=X.index)


class NbGlmPredictor(Predictor):
    """NB GLM with exogenous + y_lag7. X must contain the exog cols; history needed for lag."""
    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        import statsmodels.api as sm
        if history is None or len(history) < 7:
            raise ValueError("NbGlmPredictor needs >= 7 days of history for y_lag7")
        model = self.pkg.fitted
        scaler = self.pkg.feature_scaler
        Xs = X.copy()
        if scaler is not None:
            for c in scaler["mean"].index:
                if c in Xs.columns:
                    Xs[c] = (Xs[c] - scaler["mean"][c]) / scaler["std"][c]
        # Build the design matrix: features + y_lag7
        lags = []
        for i, date in enumerate(X.index):
            target_date_minus_7 = date - pd.Timedelta(days=7)
            if target_date_minus_7 in history.index:
                lags.append(float(history.loc[target_date_minus_7]))
            elif i >= 7:
                # Use predicted from earlier in this call (recursive)
                lags.append(float(history.iloc[-1]))
            else:
                lags.append(float(history.iloc[-(7 - i)]))
        Xs = Xs[self.pkg.feature_names[:-1]]  # exclude y_lag7 from list
        Xs["y_lag7"] = lags
        X_with_const = sm.add_constant(Xs.values, has_constant="add")
        if X_with_const.shape[1] != len(model.params):
            X_with_const = np.column_stack([np.ones(len(Xs)), Xs.values])
        mu = model.predict(X_with_const)
        return pd.Series(mu, index=X.index)


class XgboostPredictor(Predictor):
    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        model = self.pkg.fitted
        X_ord = X[self.pkg.feature_names]
        yhat = model.predict(X_ord.values)
        return pd.Series(yhat, index=X.index)


class NeuralNetPredictor(Predictor):
    """ANN or LSTM. Handles state_dict reconstruction and scaling."""
    def _build_model(self):
        import torch
        info = self.pkg.fitted
        # info is a dict: {'class_module': str, 'class_name': str,
        #                  'init_kwargs': {...}, 'state_dict': {...}}
        import importlib
        mod = importlib.import_module(info["class_module"])
        cls = getattr(mod, info["class_name"])
        model = cls(**info["init_kwargs"])
        model.load_state_dict(info["state_dict"])
        model.eval()
        return model

    def _scale_X(self, X: pd.DataFrame) -> np.ndarray:
        X_ord = X[self.pkg.feature_names]
        mean = self.pkg.feature_scaler["mean"]
        std = self.pkg.feature_scaler["std"]
        return ((X_ord - mean) / std).astype(np.float32).values

    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        import torch
        model = self._build_model()
        Xs = self._scale_X(X)
        ts = self.pkg.target_scaler or {"mean": 0.0, "std": 1.0}

        if self.pkg.lookback is None:
            # ANN-style: direct forward pass
            with torch.no_grad():
                yhat_norm = model(torch.from_numpy(Xs)).numpy()
        else:
            # LSTM-style: needs lookback rows of history concatenated with X
            if history is None:
                raise ValueError("LSTMPredictor needs history for the lookback window")
            # Pull history features from the package's history-scaling expectation —
            # the cloud app should supply a feature DataFrame `X_history` instead;
            # this simple path assumes history contains the SAME feature columns.
            raise NotImplementedError(
                "Sequence-based LSTM predict requires a feature-history DataFrame; "
                "use predict_sequence(X_history, X_future) instead."
            )
        yhat = yhat_norm * ts["std"] + ts["mean"]
        return pd.Series(yhat, index=X.index)

    def predict_sequence(self, X_history: pd.DataFrame, X_future: pd.DataFrame) -> pd.Series:
        """Sliding-window LSTM forecast. X_history must contain at least `lookback` rows."""
        import torch
        if self.pkg.lookback is None:
            return self.predict(X_future)
        model = self._build_model()
        lb = self.pkg.lookback
        cols = self.pkg.feature_names
        full = pd.concat([X_history[cols], X_future[cols]])
        mean, std = self.pkg.feature_scaler["mean"], self.pkg.feature_scaler["std"]
        full_scaled = ((full - mean) / std).astype(np.float32).values
        ts = self.pkg.target_scaler or {"mean": 0.0, "std": 1.0}

        rows = []
        n_hist = len(X_history)
        for i in range(len(X_future)):
            pos = n_hist + i
            window = full_scaled[pos - lb : pos]
            with torch.no_grad():
                yhat_norm = float(model(torch.from_numpy(window[None, :, :])).item())
            rows.append(yhat_norm * ts["std"] + ts["mean"])
        return pd.Series(rows, index=X_future.index)


class HybridPredictor(Predictor):
    """Residual hybrid: y = base.predict(X) + refiner.predict(X).

    fitted = {'base': ModelPackage, 'refiner': ModelPackage}
    """
    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        d = self.pkg.fitted
        base_pred = make_predictor(d["base"]).predict(X, history=history)
        refiner_pred = make_predictor(d["refiner"]).predict(X, history=history)
        return base_pred + refiner_pred


class STLHybridPredictor(Predictor):
    """STL hybrid: y = trend_extrapolation + seasonal_tile + refiner.predict(X).

    fitted = {'trend_tail': pd.Series (last 30 train trend),
              'seasonal_tail': pd.Series (last 7 seasonal),
              'refiner': ModelPackage}
    """
    def predict(self, X: pd.DataFrame, *, history: Optional[pd.Series] = None) -> pd.Series:
        d = self.pkg.fitted
        # Trend: damped-linear from tail
        from src.forecasting.hybrids.stl_hybrid import forecast_trend, forecast_seasonal
        trend_fc = forecast_trend(d["trend_tail"], X.index, method="damped_linear")
        seasonal_fc = forecast_seasonal(d["seasonal_tail"], X.index, period=7)
        refiner_pred = make_predictor(d["refiner"]).predict(X, history=history)
        return trend_fc + seasonal_fc + refiner_pred.reindex(X.index).fillna(0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FAMILY_MAP = {
    "naive": NaivePredictor,
    "classical_arima": ArimaPredictor,
    "classical_sarimax": SarimaxPredictor,
    "parametric_glm": NbGlmPredictor,
    "ml_xgboost": XgboostPredictor,
    "dl_ann": NeuralNetPredictor,
    "dl_lstm": NeuralNetPredictor,
    "hybrid_residual": HybridPredictor,
    "hybrid_stl": STLHybridPredictor,
}


def make_predictor(pkg: ModelPackage) -> Predictor:
    cls = _FAMILY_MAP.get(pkg.family)
    if cls is None:
        raise ValueError(f"No predictor for family '{pkg.family}'. "
                          f"Known: {list(_FAMILY_MAP)}")
    return cls(pkg)


# ---------------------------------------------------------------------------
# Public load function for cloud apps
# ---------------------------------------------------------------------------

def load_model(path: str | Path) -> Predictor:
    """Single entry point a cloud app should use.

        from src.forecasting.deploy import load_model
        predictor = load_model('artefacts/models/deploy/sarimax.pkl')
        forecast = predictor.predict(X_future, history=y_history)
    """
    return make_predictor(load_package(path))
