"""Save every trained model as a .pkl ready for a cloud-based app.

Refits each model on the §5.5.2 training fold (post-COVID, 848 days) using the
best hyperparameters discovered during HPO, wraps the fitted state plus
preprocessing metadata in a ModelPackage, and pickles to
artefacts/models/deploy/.

The packaged pickles are self-contained — a cloud app calls
src.forecasting.deploy.load_model(path) and gets a Predictor with a
.predict(X, history=...) method.

Run this AFTER the CV runs deposit their best_params.json files:
  scripts/06_xgboost.py
  scripts/07_ann.py
  scripts/08_lstm.py
  scripts/09_hybrids.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score
from src.forecasting.deploy import ModelPackage, save_package


OUT = ROOT / "artefacts" / "models" / "deploy"
OUT.mkdir(parents=True, exist_ok=True)


def _common_metadata(name: str, family: str, n_train: int,
                      val_metrics: dict | None = None) -> dict:
    return {
        "model_name": name,
        "family": family,
        "n_train_days": n_train,
        "trained_on": "2022-03-01 to 2024-06-30 (post-COVID, §5.5.2)",
        "split_source": "Ch5 §5.5.2",
        "val_metrics": val_metrics or {},
    }


def _load_val_metrics(name: str) -> dict | None:
    p = ROOT / "artefacts" / "metrics" / f"{name}_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "block" in df.columns:
        row = df[df["block"] == "val"].iloc[0]
    else:
        row = df.iloc[0]
    return {k: float(row[k]) for k in ("MAPE", "MAE", "RMSE", "R2") if k in row}


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------

def save_naive_baselines(g1: pd.DataFrame, train_idx: pd.DatetimeIndex) -> None:
    target = g1["total_daily_arrivals"]
    train = target.loc[train_idx]

    # DoW mean
    weekday_means = train.groupby(train.index.dayofweek).mean().to_dict()
    pkg = ModelPackage(
        name="dow_mean", family="naive",
        fitted=weekday_means,
        metadata=_common_metadata("dow_mean", "naive", len(train),
                                    _load_val_metrics("reference_floor")),
    )
    save_package(pkg, OUT / "dow_mean.pkl")
    print(f"  saved dow_mean.pkl")

    # Naive yest / seasonal — no model state needed (rule based)
    for nm in ("naive_yest", "naive_seasonal"):
        pkg = ModelPackage(
            name=nm, family="naive",
            fitted=None,
            metadata=_common_metadata(nm, "naive", len(train), None),
        )
        save_package(pkg, OUT / f"{nm}.pkl")
        print(f"  saved {nm}.pkl")


# ---------------------------------------------------------------------------
# ARIMA / SARIMAX / NB GLM
# ---------------------------------------------------------------------------

def save_arima(g1, train_idx):
    from src.forecasting.models.arima import pick_order
    target = g1["total_daily_arrivals"].loc[train_idx]
    order_file = ROOT / "artefacts" / "models" / "arima_order.txt"
    if order_file.exists():
        cfg = dict(line.partition("=")[::2] for line in
                    order_file.read_text().strip().splitlines())
        order = eval(cfg["order"].strip())
        from pmdarima import ARIMA as PmARIMA
        model = PmARIMA(order=order, suppress_warnings=True)
        model.fit(target.values)
    else:
        model = pick_order(target).fitted_train
    pkg = ModelPackage(
        name="arima", family="classical_arima",
        fitted=model,
        best_params={"order": tuple(model.order)},
        metadata=_common_metadata("arima", "classical_arima", len(target),
                                    _load_val_metrics("arima")),
    )
    save_package(pkg, OUT / "arima.pkl")
    print(f"  saved arima.pkl  (order={model.order})")


def save_sarimax(g1, train_idx):
    target = g1["total_daily_arrivals"].loc[train_idx]
    X_train, scaler = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    order_file = ROOT / "artefacts" / "models" / "sarimax_order.txt"
    cfg = dict(line.partition("=")[::2] for line in
                order_file.read_text().strip().splitlines())
    order = eval(cfg["order"].strip())
    seasonal_order = eval(cfg["seasonal_order"].strip())
    from pmdarima import ARIMA as PmARIMA
    model = PmARIMA(order=order, seasonal_order=seasonal_order,
                    suppress_warnings=True)
    model.fit(target.values, X=X_train.values)
    pkg = ModelPackage(
        name="sarimax", family="classical_sarimax",
        fitted=model,
        feature_names=list(X_train.columns),
        feature_scaler={"mean": scaler.mean_, "std": scaler.std_},
        best_params={"order": tuple(order), "seasonal_order": tuple(seasonal_order)},
        metadata=_common_metadata("sarimax", "classical_sarimax",
                                    len(target),
                                    _load_val_metrics("sarimax")),
    )
    save_package(pkg, OUT / "sarimax.pkl")
    print(f"  saved sarimax.pkl  (order={order} x {seasonal_order})")


def save_nbglm(g1, train_idx):
    from src.forecasting.models.negbin import fit_with_lag7
    target = g1["total_daily_arrivals"].loc[train_idx]
    X_train, scaler = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    fit_result = fit_with_lag7(target, X_train, fit_normal_sensitivity=False)
    pkg = ModelPackage(
        name="nbglm", family="parametric_glm",
        fitted=fit_result.fitted_train,
        feature_names=list(X_train.columns) + ["y_lag7"],
        feature_scaler={"mean": scaler.mean_, "std": scaler.std_},
        best_params={"alpha": fit_result.alpha},
        metadata={**_common_metadata("nbglm", "parametric_glm",
                                       len(target),
                                       _load_val_metrics("nbglm")),
                   "dispersion_alpha": fit_result.alpha,
                   "aic_nb": fit_result.aic_nb},
    )
    save_package(pkg, OUT / "nbglm.pkl")
    print(f"  saved nbglm.pkl  (alpha={fit_result.alpha:.3f})")


# ---------------------------------------------------------------------------
# ML / DL standalone
# ---------------------------------------------------------------------------

def _consensus_inputs(g1, train_idx):
    eng = load_engineered()
    X_all = build_selected_X(eng)
    target = g1["total_daily_arrivals"]
    df = pd.concat([target.rename("y"), X_all], axis=1, join="inner").dropna()
    train_idx = train_idx.intersection(df.index)
    return df.loc[train_idx, "y"], df.loc[train_idx].drop(columns=["y"]), df


def save_xgboost(g1, train_idx):
    from xgboost import XGBRegressor
    y_train, X_train, _ = _consensus_inputs(g1, train_idx)
    params_path = ROOT / "artefacts" / "models" / "xgboost_best_params.json"
    params = (json.loads(params_path.read_text()) if params_path.exists()
              else {"n_estimators": 200, "max_depth": 4,
                    "learning_rate": 0.05, "subsample": 0.85})
    model = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train.values, y_train.values)
    pkg = ModelPackage(
        name="xgboost", family="ml_xgboost",
        fitted=model,
        feature_names=list(X_train.columns),
        feature_scaler=None,  # XGBoost handles unscaled features directly
        best_params=params,
        metadata=_common_metadata("xgboost", "ml_xgboost", len(y_train),
                                    _load_val_metrics("xgboost")),
    )
    save_package(pkg, OUT / "xgboost.pkl")
    print(f"  saved xgboost.pkl  (params={params})")


def _train_ann_and_package(g1, train_idx, family_name="dl_ann"):
    import torch
    from src.forecasting.models.ann import _MLP, _train_one, _seed_everything
    y_train, X_train, _ = _consensus_inputs(g1, train_idx)
    params_path = ROOT / "artefacts" / "models" / "ann_best_params.json"
    params = (json.loads(params_path.read_text()) if params_path.exists()
              else {"hidden_layers": 1, "units": 128, "dropout": 0.2,
                    "learning_rate": 0.001, "batch_size": 32, "seed": 42})
    _seed_everything(params.get("seed", 42))

    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32)
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32)

    n_es = max(28, len(Xtr) // 6)
    Xtr_t = torch.from_numpy(Xtr.values[:-n_es])
    ytr_t = torch.from_numpy(ytr.values[:-n_es])
    Xes_t = torch.from_numpy(Xtr.values[-n_es:])
    yes_t = torch.from_numpy(ytr.values[-n_es:])

    n_hidden = [params["units"]] * params["hidden_layers"]
    model, _, _ = _train_one(Xtr_t, ytr_t, Xes_t, yes_t, params, max_epochs=120)

    info = {
        "class_module": "src.forecasting.models.ann",
        "class_name": "_MLP",
        "init_kwargs": {
            "n_in": Xtr.shape[1],
            "n_hidden": n_hidden,
            "dropout": params["dropout"],
        },
        "state_dict": model.state_dict(),
    }
    pkg = ModelPackage(
        name="ann", family=family_name,
        fitted=info,
        feature_names=list(X_train.columns),
        feature_scaler={"mean": mean, "std": std},
        target_scaler={"mean": y_mean, "std": y_std},
        best_params=params,
        metadata=_common_metadata("ann", family_name, len(y_train),
                                    _load_val_metrics("ann")),
    )
    save_package(pkg, OUT / "ann.pkl")
    print(f"  saved ann.pkl  (params={params})")


def _train_lstm_and_package(g1, train_idx):
    import torch
    from src.forecasting.models.lstm import (
        _LSTMNet, _build_sequences, _train_one, _seed_everything,
    )
    y_train, X_train, _ = _consensus_inputs(g1, train_idx)
    params_path = ROOT / "artefacts" / "models" / "lstm_best_params.json"
    params = (json.loads(params_path.read_text()) if params_path.exists()
              else {"lookback": 21, "units": 128, "dropout": 0.2,
                    "learning_rate": 0.001, "batch_size": 32, "seed": 42})
    _seed_everything(params.get("seed", 42))

    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values
    lookback = params["lookback"]
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
    n_es = max(28, lookback + 7)
    model, _ = _train_one(Xtr_seq[:-n_es], ytr_seq[:-n_es],
                            Xtr_seq[-n_es:], ytr_seq[-n_es:],
                            params, max_epochs=60)

    info = {
        "class_module": "src.forecasting.models.lstm",
        "class_name": "_LSTMNet",
        "init_kwargs": {
            "n_features": Xtr.shape[1],
            "units": params["units"],
            "dropout": params["dropout"],
        },
        "state_dict": model.state_dict(),
    }
    pkg = ModelPackage(
        name="lstm", family="dl_lstm",
        fitted=info,
        feature_names=list(X_train.columns),
        feature_scaler={"mean": mean, "std": std},
        target_scaler={"mean": y_mean, "std": y_std},
        lookback=lookback,
        best_params=params,
        metadata=_common_metadata("lstm", "dl_lstm", len(y_train),
                                    _load_val_metrics("lstm")),
    )
    save_package(pkg, OUT / "lstm.pkl")
    print(f"  saved lstm.pkl  (params={params})")


# ---------------------------------------------------------------------------
# Hybrids
# ---------------------------------------------------------------------------

def _load_package(name: str) -> ModelPackage | None:
    from src.forecasting.deploy import load_package
    p = OUT / f"{name}.pkl"
    return load_package(p) if p.exists() else None


def save_residual_hybrids(g1, train_idx):
    """Save SARIMAX+XGB, SARIMAX+LSTM, LSTM+XGB as HybridPredictor packages."""
    from src.forecasting.hybrids import residual as R
    from src.forecasting.features import build_task1_exogenous

    target = g1["total_daily_arrivals"]
    y_train, X_train_consensus, df_consensus = _consensus_inputs(g1, train_idx)

    # 1) SARIMAX + XGB / + LSTM share the SARIMAX residual signal
    order_file = ROOT / "artefacts" / "models" / "sarimax_order.txt"
    cfg = dict(line.partition("=")[::2] for line in
                order_file.read_text().strip().splitlines())
    order = eval(cfg["order"].strip())
    seasonal_order = eval(cfg["seasonal_order"].strip())
    X_sarimax_train, _ = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    train_resid = R.sarimax_train_residuals(target, X_sarimax_train,
                                              order, seasonal_order)

    # Refit XGBoost refiner on residuals
    xgb_refiner = R.fit_xgb_refiner(X_train_consensus.loc[train_resid.index],
                                      train_resid)
    pkg_xgb_refiner = ModelPackage(
        name="hybrid_sarimax_xgb_refiner", family="ml_xgboost",
        fitted=xgb_refiner,
        feature_names=list(X_train_consensus.columns),
        feature_scaler=None,
        metadata={"role": "refiner_on_sarimax_residuals"},
    )
    base_pkg = _load_package("sarimax")
    if base_pkg is None:
        print("  (sarimax base not packaged yet; skipping hybrids)")
        return
    hybrid_pkg = ModelPackage(
        name="hybrid_sarimax_xgb", family="hybrid_residual",
        fitted={"base": base_pkg, "refiner": pkg_xgb_refiner},
        metadata=_common_metadata("hybrid_sarimax_xgb", "hybrid_residual",
                                    len(y_train),
                                    _load_val_metrics("hybrid_sarimax_xgb")),
    )
    save_package(hybrid_pkg, OUT / "hybrid_sarimax_xgb.pkl")
    print(f"  saved hybrid_sarimax_xgb.pkl")

    # LSTM refiner on the same SARIMAX residuals -> SARIMAX+LSTM
    lstm_refiner = R.fit_lstm_refiner(X_train_consensus.loc[train_resid.index],
                                        train_resid)
    # Wrap LSTM refiner in a ModelPackage we can later reconstruct
    refiner_model, mean, std, r_mean, r_std, lookback = lstm_refiner
    info = {
        "class_module": "src.forecasting.models.lstm",
        "class_name": "_LSTMNet",
        "init_kwargs": {
            "n_features": X_train_consensus.shape[1],
            "units": 64, "dropout": 0.2,
        },
        "state_dict": refiner_model.state_dict(),
    }
    pkg_lstm_refiner = ModelPackage(
        name="hybrid_sarimax_lstm_refiner", family="dl_lstm",
        fitted=info,
        feature_names=list(X_train_consensus.columns),
        feature_scaler={"mean": mean, "std": std},
        target_scaler={"mean": r_mean, "std": r_std},
        lookback=lookback,
        metadata={"role": "refiner_on_sarimax_residuals"},
    )
    hybrid_pkg = ModelPackage(
        name="hybrid_sarimax_lstm", family="hybrid_residual",
        fitted={"base": base_pkg, "refiner": pkg_lstm_refiner},
        metadata=_common_metadata("hybrid_sarimax_lstm", "hybrid_residual",
                                    len(y_train),
                                    _load_val_metrics("hybrid_sarimax_lstm")),
    )
    save_package(hybrid_pkg, OUT / "hybrid_sarimax_lstm.pkl")
    print(f"  saved hybrid_sarimax_lstm.pkl")

    # 2) LSTM + XGBoost residual hybrid (needs standalone LSTM packaged first)
    lstm_pkg = _load_package("lstm")
    if lstm_pkg is None:
        print("  (lstm base not packaged yet; skipping LSTM+XGB hybrid)")
    else:
        # Use the standalone LSTM's in-sample residuals to train an XGB refiner
        lstm_params = lstm_pkg.best_params
        in_sample, _ = R.lstm_train_in_sample(target, X_sarimax_train, lstm_params)
        lxg_resid = target.loc[in_sample.index] - in_sample
        sigma = lxg_resid.std()
        lxg_resid = lxg_resid[lxg_resid.abs() <= 5 * sigma]
        xgb_refiner_lxg = R.fit_xgb_refiner(
            X_train_consensus.loc[lxg_resid.index.intersection(X_train_consensus.index)],
            lxg_resid,
        )
        pkg_lxg_refiner = ModelPackage(
            name="hybrid_lstm_xgb_refiner", family="ml_xgboost",
            fitted=xgb_refiner_lxg,
            feature_names=list(X_train_consensus.columns),
            metadata={"role": "refiner_on_lstm_residuals"},
        )
        hybrid_pkg = ModelPackage(
            name="hybrid_lstm_xgb", family="hybrid_residual",
            fitted={"base": lstm_pkg, "refiner": pkg_lxg_refiner},
            metadata=_common_metadata("hybrid_lstm_xgb", "hybrid_residual",
                                        len(y_train),
                                        _load_val_metrics("hybrid_lstm_xgb")),
        )
        save_package(hybrid_pkg, OUT / "hybrid_lstm_xgb.pkl")
        print(f"  saved hybrid_lstm_xgb.pkl")


def save_stl_hybrids(g1, train_idx):
    """Save STL+XGB, STL+ANN, STL+LSTM as STLHybridPredictor packages."""
    from src.forecasting.hybrids import stl_hybrid as S
    y_train, X_train, _ = _consensus_inputs(g1, train_idx)
    decomp = S.decompose_train(y_train, period=7)

    # Refiners
    refiners = {}
    refiners["xgb"] = S.fit_xgb_refiner_on_residual(X_train, decomp.residual)
    refiners["ann"] = S.fit_ann_refiner_on_residual(X_train, decomp.residual)
    refiners["lstm"] = S.fit_lstm_refiner_on_residual(X_train, decomp.residual)

    for kind in ("xgb", "ann", "lstm"):
        name = f"hybrid_stl_{kind}"
        ref = refiners[kind]
        if kind == "xgb":
            refiner_pkg = ModelPackage(
                name=f"{name}_refiner", family="ml_xgboost",
                fitted=ref,
                feature_names=list(X_train.columns),
                metadata={"role": "refiner_on_stl_residual"},
            )
        elif kind == "ann":
            model, mean, std, r_mean, r_std = ref
            info = {
                "class_module": "src.forecasting.models.ann",
                "class_name": "_MLP",
                "init_kwargs": {
                    "n_in": X_train.shape[1],
                    "n_hidden": [128, 128],
                    "dropout": 0.2,
                },
                "state_dict": model.state_dict(),
            }
            refiner_pkg = ModelPackage(
                name=f"{name}_refiner", family="dl_ann",
                fitted=info,
                feature_names=list(X_train.columns),
                feature_scaler={"mean": mean, "std": std},
                target_scaler={"mean": r_mean, "std": r_std},
                metadata={"role": "refiner_on_stl_residual"},
            )
        else:  # lstm
            model, mean, std, r_mean, r_std, lookback = ref
            info = {
                "class_module": "src.forecasting.models.lstm",
                "class_name": "_LSTMNet",
                "init_kwargs": {
                    "n_features": X_train.shape[1],
                    "units": 64, "dropout": 0.2,
                },
                "state_dict": model.state_dict(),
            }
            refiner_pkg = ModelPackage(
                name=f"{name}_refiner", family="dl_lstm",
                fitted=info,
                feature_names=list(X_train.columns),
                feature_scaler={"mean": mean, "std": std},
                target_scaler={"mean": r_mean, "std": r_std},
                lookback=lookback,
                metadata={"role": "refiner_on_stl_residual"},
            )

        hybrid_pkg = ModelPackage(
            name=name, family="hybrid_stl",
            fitted={
                "trend_tail": decomp.trend.iloc[-30:],
                "seasonal_tail": decomp.seasonal.iloc[-7:],
                "refiner": refiner_pkg,
            },
            metadata=_common_metadata(name, "hybrid_stl", len(y_train),
                                        _load_val_metrics(name)),
        )
        save_package(hybrid_pkg, OUT / f"{name}.pkl")
        print(f"  saved {name}.pkl")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest():
    """Catalogue every saved pickle for the cloud app."""
    pkls = sorted(OUT.glob("*.pkl"))
    manifest = []
    from src.forecasting.deploy import load_package
    for p in pkls:
        try:
            pkg = load_package(p)
            row = {
                "filename": p.name,
                "model_name": pkg.name,
                "family": pkg.family,
                "feature_count": len(pkg.feature_names),
                "lookback": pkg.lookback,
                "best_params": pkg.best_params,
                "val_metrics": pkg.metadata.get("val_metrics", {}),
                "saved_at": pkg.metadata.get("saved_at"),
                "package_version": pkg.metadata.get("package_version"),
                "trained_on": pkg.metadata.get("trained_on"),
            }
            manifest.append(row)
        except Exception as exc:
            manifest.append({"filename": p.name, "load_error": str(exc)})
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nManifest written: {manifest_path.relative_to(ROOT)}")
    print(f"Total packages: {len(manifest)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    train_idx = splits.slice(g1, "train").index

    print("=" * 70)
    print("Saving every trained model as a cloud-ready .pkl")
    print(f"Output directory: {OUT.relative_to(ROOT)}")
    print(f"Training fold: {len(train_idx)} days (post-COVID, §5.5.2)")
    print("=" * 70)

    t0 = time.time()
    print("\n[1/5] Naive baselines")
    save_naive_baselines(g1, train_idx)

    print("\n[2/5] Classical / parametric")
    save_arima(g1, train_idx)
    save_sarimax(g1, train_idx)
    save_nbglm(g1, train_idx)

    print("\n[3/5] ML / DL standalone")
    save_xgboost(g1, train_idx)
    _train_ann_and_package(g1, train_idx)
    _train_lstm_and_package(g1, train_idx)

    print("\n[4/5] Residual hybrids")
    save_residual_hybrids(g1, train_idx)

    print("\n[5/5] STL hybrids")
    save_stl_hybrids(g1, train_idx)

    write_manifest()
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("\nLoad in a cloud app via:")
    print("    from src.forecasting.deploy import load_model")
    print("    predictor = load_model('artefacts/models/deploy/sarimax.pkl')")
    print("    forecast = predictor.predict(X_future, history=y_history)")


if __name__ == "__main__":
    main()
