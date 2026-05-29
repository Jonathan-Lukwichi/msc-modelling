"""Phase 2 — HPO targeting minimum RMSE.

For each model, runs HPO with three algorithms:
  Grid search    (deterministic, exhaustive over discrete grid)
  Random search  (uniform sampling in continuous bounds)
  Optuna TPE     (Bayesian, informed sampling)

Fair protocol: 10 trials per algorithm × 5 folds inner rolling-origin CV.
Selection metric: cv_RMSE (lower = better).

Scope (Option B per user):
  Standalone (XGB, ANN, LSTM) -> all 3 algorithms
  Hybrids (STL+*, SARIMAX+*, LSTM+XGB) -> Optuna only

Outputs:
  artefacts/phase2_hpo/
    trace_{model}_{algo}.csv            full HPO trace (per-trial cv metrics)
    winner_{model}_{algo}.json          winning params
    val_preds_{model}_{algo}.csv        rolling-forecast on val + actual
    daily_{model}_{algo}.csv            daily table with per-row metrics
    weekly_{model}_{algo}.csv           weekly aggregated metrics
    monthly_{model}_{algo}.csv          monthly aggregated metrics
    yearly_{model}_{algo}.csv           yearly aggregated metrics
    summary_phase2.csv                  one-row-per-(model,algo) headlines
    figures/hpo_comparison_{model}.png  Grid vs Random vs Optuna for that model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Re-use Phase 1 fit_predict + aggregation helpers
from scripts.phase1_defaults_all_models import (
    xgb_fit_predict, ann_fit_predict, lstm_fit_predict,
    arima_fit_predict, sarimax_fit_predict,
    stl_hybrid_fit_predict, sarimax_hybrid_fit_predict,
    lstm_xgb_hybrid_fit_predict,
    rolling_forecast_generic, aggregate_metrics, load_data,
)
from src.forecasting.metrics import score
from src.forecasting.cv import subsampled_rolling_origin


OUT = ROOT / "artefacts" / "phase2_hpo"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Search spaces (Chapter 5 ranges)
# -------------------------------------------------------------------------
SPACES = {
    "xgboost": {
        "grid": {
            "n_estimators": [100, 300, 500],
            "max_depth":    [3, 5],
            "learning_rate":[0.01, 0.1],
            "subsample":    [0.85],
        },
        "random_bounds": {
            "n_estimators": (100, 500, "int_choice", [100, 200, 300, 500]),
            "max_depth":    (3, 8, "int"),
            "learning_rate":(0.01, 0.3, "loguniform"),
            "subsample":    (0.7, 1.0, "uniform"),
        },
    },
    "ann": {
        "grid": {
            "hidden_layers":[1, 2],
            "units":        [64, 128, 192],
            "dropout":      [0.2, 0.3],
            "learning_rate":[0.001, 0.003],
        },
        "random_bounds": {
            "hidden_layers":(1, 2, "int_choice", [1, 2]),
            "units":        (64, 256, "int_choice", [64, 128, 192, 256]),
            "dropout":      (0.1, 0.4, "uniform"),
            "learning_rate":(1e-4, 1e-2, "loguniform"),
        },
    },
    "lstm": {
        "grid": {
            "lookback":     [7, 14],
            "units":        [64, 128],
            "dropout":      [0.2],
            "learning_rate":[0.001, 0.003],
        },
        "random_bounds": {
            "lookback":     (7, 21, "int_choice", [7, 14, 21]),
            "units":        (64, 192, "int_choice", [64, 96, 128, 192]),
            "dropout":      (0.1, 0.3, "uniform"),
            "learning_rate":(1e-4, 5e-3, "loguniform"),
        },
    },
}


# -------------------------------------------------------------------------
# HPO algorithms — return list of (params_dict, cv_RMSE, cv_MAPE, cv_MAE)
# -------------------------------------------------------------------------
def _cv_eval(params, X, y, fit_predict_fn, n_folds=5):
    """Average metrics across n_folds rolling-origin folds."""
    folds = subsampled_rolling_origin(X.index, n_folds=n_folds,
                                      horizon_days=7, step_days=7,
                                      min_train_days=365)
    rmses, mapes, maes = [], [], []
    for f in folds:
        Xtr = X.iloc[f.train_idx]; ytr = y.iloc[f.train_idx]
        Xte = X.iloc[f.test_idx];  yte = y.iloc[f.test_idx]
        yhat = fit_predict_fn(Xtr, ytr, Xte, params=params) \
               if "params" in fit_predict_fn.__code__.co_varnames \
               else fit_predict_fn(Xtr, ytr, Xte)
        m = score(yte.values, np.asarray(yhat).ravel())
        rmses.append(m["RMSE"]); mapes.append(m["MAPE"]); maes.append(m["MAE"])
    return float(np.mean(rmses)), float(np.mean(mapes)), float(np.mean(maes))


def hpo_grid(model: str, X, y, fit_predict_fn, n_trials=10, n_folds=5):
    grid = SPACES[model]["grid"]
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    # Cap to n_trials (pick evenly spaced)
    if len(combos) > n_trials:
        idx = np.linspace(0, len(combos)-1, n_trials).round().astype(int)
        combos = [combos[i] for i in sorted(set(idx))]
    trace = []
    for c in combos:
        params = dict(zip(keys, c))
        # Add seed for stochastic models
        if model in ("ann", "lstm"):
            params.setdefault("batch_size", 32); params.setdefault("seed", 42)
        rmse, mape, mae = _cv_eval(params, X, y, fit_predict_fn, n_folds)
        trace.append({**params, "cv_RMSE": rmse, "cv_MAPE": mape, "cv_MAE": mae})
    return pd.DataFrame(trace)


def hpo_random(model: str, X, y, fit_predict_fn, n_trials=10, n_folds=5,
               seed=42):
    rng = np.random.default_rng(seed)
    bounds = SPACES[model]["random_bounds"]
    trace = []
    for t in range(n_trials):
        params = {}
        for k, spec in bounds.items():
            if spec[2] == "int_choice":
                params[k] = int(rng.choice(spec[3]))
            elif spec[2] == "int":
                params[k] = int(rng.integers(spec[0], spec[1]+1))
            elif spec[2] == "uniform":
                params[k] = float(rng.uniform(spec[0], spec[1]))
            elif spec[2] == "loguniform":
                params[k] = float(np.exp(rng.uniform(np.log(spec[0]),
                                                      np.log(spec[1]))))
        if model in ("ann", "lstm"):
            params.setdefault("batch_size", 32); params.setdefault("seed", 42)
        rmse, mape, mae = _cv_eval(params, X, y, fit_predict_fn, n_folds)
        trace.append({**params, "cv_RMSE": rmse, "cv_MAPE": mape, "cv_MAE": mae})
    return pd.DataFrame(trace)


def hpo_auto_arima(model: str, X, y, n_folds=5):
    """auto_arima HPO for ARIMA / SARIMAX (Chapter 5 §5.2.2 template, AIC).

    Picks the order on the full training fold (AIC) and reports cv_RMSE/MAPE
    via 5-fold rolling-origin CV using the picked order. This matches the
    Chapter 5 protocol — AIC is the principled order-selection criterion for
    these models; we report RMSE for comparability with the ML side.
    """
    from pmdarima import auto_arima
    from src.forecasting.features import build_task1_exogenous

    y_tr = y
    print(f"  Running auto_arima ({model}) on full train fold...")
    if model == "arima":
        m = auto_arima(
            y_tr.values, start_p=0, start_q=0, max_p=3, max_q=3, d=1,
            seasonal=False, stepwise=True, suppress_warnings=True,
            error_action="ignore", information_criterion="aic", trace=False,
            random_state=42,
        )
        order = m.order
        seasonal_order = (0, 0, 0, 0)
        aic = float(m.aic())
        print(f"  picked order: {order}, AIC = {aic:.2f}")
    elif model == "sarimax":
        from src.forecasting.io import load_g1
        g1 = load_g1()
        Xexog, _ = build_task1_exogenous(g1.loc[y_tr.index], fit_scaler=True)
        m = auto_arima(
            y_tr.values, X=Xexog.values,
            start_p=0, start_q=0, max_p=2, max_q=2, d=1,
            seasonal=True, m=7,
            start_P=0, start_Q=0, max_P=2, max_Q=2, D=1,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", information_criterion="aic", trace=False,
            random_state=42,
        )
        order = m.order
        seasonal_order = m.seasonal_order
        aic = float(m.aic())
        print(f"  picked order: {order} x {seasonal_order}, AIC = {aic:.2f}")
    else:
        raise ValueError(model)

    # Build fit_predict with the picked order, run rolling-origin CV
    if model == "arima":
        def fp(Xtr, ytr, Xte, params=None):
            return arima_fit_predict(Xtr, ytr, Xte,
                                     params={"order": order})
    else:
        def fp(Xtr, ytr, Xte, params=None):
            return sarimax_fit_predict(Xtr, ytr, Xte,
                                       params={"order": order,
                                               "seasonal_order": seasonal_order})

    rmse, mape, mae = _cv_eval(None, X, y, fp, n_folds=n_folds)
    return pd.DataFrame([{
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "aic": aic,
        "cv_RMSE": rmse, "cv_MAPE": mape, "cv_MAE": mae,
    }])


def hpo_optuna(model: str, X, y, fit_predict_fn, n_trials=10, n_folds=5,
               seed=42):
    import optuna
    bounds = SPACES[model]["random_bounds"]
    trace = []

    def objective(trial):
        params = {}
        for k, spec in bounds.items():
            if spec[2] == "int_choice":
                params[k] = trial.suggest_categorical(k, spec[3])
            elif spec[2] == "int":
                params[k] = trial.suggest_int(k, spec[0], spec[1])
            elif spec[2] == "uniform":
                params[k] = trial.suggest_float(k, spec[0], spec[1])
            elif spec[2] == "loguniform":
                params[k] = trial.suggest_float(k, spec[0], spec[1], log=True)
        if model in ("ann", "lstm"):
            params.setdefault("batch_size", 32); params.setdefault("seed", 42)
        rmse, mape, mae = _cv_eval(params, X, y, fit_predict_fn, n_folds)
        trace.append({**params, "cv_RMSE": rmse, "cv_MAPE": mape, "cv_MAE": mae})
        return rmse

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    return pd.DataFrame(trace)


# -------------------------------------------------------------------------
# Hybrid fit_predict wrappers (refiner takes params; outer components fixed)
# -------------------------------------------------------------------------
def make_hybrid_fp(hybrid_name: str):
    """Return a fit_predict that takes a 'params' kwarg to tune the refiner."""
    if hybrid_name == "stl_xgb":
        def fp(Xtr, ytr, Xte, params=None):
            from src.forecasting.hybrids.stl_hybrid import (
                decompose_train, forecast_trend, forecast_seasonal,
            )
            d = decompose_train(ytr); val_idx = Xte.index
            return (forecast_trend(d.trend, val_idx).values
                    + forecast_seasonal(d.seasonal, val_idx).values
                    + xgb_fit_predict(Xtr, d.residual, Xte, params=params))
        return fp
    if hybrid_name == "stl_ann":
        def fp(Xtr, ytr, Xte, params=None):
            from src.forecasting.hybrids.stl_hybrid import (
                decompose_train, forecast_trend, forecast_seasonal,
            )
            d = decompose_train(ytr); val_idx = Xte.index
            return (forecast_trend(d.trend, val_idx).values
                    + forecast_seasonal(d.seasonal, val_idx).values
                    + ann_fit_predict(Xtr, d.residual, Xte, params=params))
        return fp
    if hybrid_name == "stl_lstm":
        def fp(Xtr, ytr, Xte, params=None):
            from src.forecasting.hybrids.stl_hybrid import (
                decompose_train, forecast_trend, forecast_seasonal,
            )
            d = decompose_train(ytr); val_idx = Xte.index
            return (forecast_trend(d.trend, val_idx).values
                    + forecast_seasonal(d.seasonal, val_idx).values
                    + lstm_fit_predict(Xtr, d.residual, Xte, params=params))
        return fp
    if hybrid_name == "sarimax_xgb":
        # SARIMAX order fixed (Ch5 AIC); tune the XGB refiner via params
        def fp(Xtr, ytr, Xte, params=None):
            from src.forecasting.io import load_g1
            from src.forecasting.features import build_task1_exogenous
            from pmdarima import ARIMA as PmARIMA
            order = (1, 1, 1); seas = (0, 1, 1, 7)
            try:
                cfg = {}
                for line in (ROOT / "artefacts" / "models" /
                             "sarimax_order.txt").read_text().strip().splitlines():
                    k, _, v = line.partition("="); cfg[k.strip()] = v.strip()
                order = eval(cfg["order"]); seas = eval(cfg["seasonal_order"])
            except Exception:
                pass
            g1 = load_g1()
            Xexog_train, scaler = build_task1_exogenous(g1.loc[Xtr.index],
                                                         fit_scaler=True)
            Xexog_test, _ = build_task1_exogenous(g1.loc[Xte.index],
                                                   scaler=scaler)
            m = PmARIMA(order=order, seasonal_order=seas,
                        suppress_warnings=True)
            m.fit(ytr.values, X=Xexog_train.values)
            base_pred = m.predict(n_periods=len(Xte), X=Xexog_test.values)
            in_sample = m.predict_in_sample(X=Xexog_train.values)
            resid = pd.Series(ytr.values - in_sample, index=Xtr.index)
            rp = xgb_fit_predict(Xtr, resid, Xte, params=params)
            return np.asarray(base_pred).ravel() + np.asarray(rp).ravel()
        return fp
    if hybrid_name == "sarimax_lstm":
        # SARIMAX order fixed (Ch5 AIC); tune the LSTM refiner via params
        def fp(Xtr, ytr, Xte, params=None):
            from src.forecasting.io import load_g1
            from src.forecasting.features import build_task1_exogenous
            from pmdarima import ARIMA as PmARIMA
            order = (1, 1, 1); seas = (0, 1, 1, 7)
            try:
                cfg = {}
                for line in (ROOT / "artefacts" / "models" /
                             "sarimax_order.txt").read_text().strip().splitlines():
                    k, _, v = line.partition("="); cfg[k.strip()] = v.strip()
                order = eval(cfg["order"]); seas = eval(cfg["seasonal_order"])
            except Exception:
                pass
            g1 = load_g1()
            Xexog_train, scaler = build_task1_exogenous(g1.loc[Xtr.index],
                                                         fit_scaler=True)
            Xexog_test, _ = build_task1_exogenous(g1.loc[Xte.index],
                                                   scaler=scaler)
            m = PmARIMA(order=order, seasonal_order=seas,
                        suppress_warnings=True)
            m.fit(ytr.values, X=Xexog_train.values)
            base_pred = m.predict(n_periods=len(Xte), X=Xexog_test.values)
            in_sample = m.predict_in_sample(X=Xexog_train.values)
            resid = pd.Series(ytr.values - in_sample, index=Xtr.index)
            rp = lstm_fit_predict(Xtr, resid, Xte, params=params)
            return np.asarray(base_pred).ravel() + np.asarray(rp).ravel()
        return fp
    if hybrid_name == "lstm_xgb":
        # LSTM at Ch5 default (base, fixed); tune the XGB refiner via params
        def fp(Xtr, ytr, Xte, params=None):
            from scripts.phase1_defaults_all_models import DEFAULTS
            lstm_default = DEFAULTS["lstm"]
            lstm_tr_pred = lstm_fit_predict(Xtr, ytr, Xtr, params=lstm_default)
            lstm_te_pred = lstm_fit_predict(Xtr, ytr, Xte, params=lstm_default)
            L = lstm_default["lookback"]
            ytr_aligned = ytr.iloc[L:] if len(lstm_tr_pred) != len(ytr) else ytr
            Xtr_aligned = Xtr.iloc[L:] if len(lstm_tr_pred) != len(ytr) else Xtr
            n = min(len(lstm_tr_pred), len(ytr_aligned))
            resid = pd.Series(
                ytr_aligned.values[:n] - np.asarray(lstm_tr_pred).ravel()[:n],
                index=Xtr_aligned.index[:n])
            rp = xgb_fit_predict(Xtr_aligned.iloc[:n], resid, Xte,
                                  params=params)
            return np.asarray(lstm_te_pred).ravel() + np.asarray(rp).ravel()
        return fp
    raise ValueError(hybrid_name)


# -------------------------------------------------------------------------
# Refiner-tunable hybrids share the standalone model's search space
# -------------------------------------------------------------------------
def base_model_of(hybrid: str) -> str:
    if hybrid.endswith("_xgb") or hybrid.startswith("sarimax_xgb"):
        return "xgboost"
    if hybrid.endswith("_ann"):
        return "ann"
    if hybrid.endswith("_lstm") or hybrid == "lstm_xgb":
        return "lstm"
    raise ValueError(hybrid)


# -------------------------------------------------------------------------
# Per-(model, algo) pipeline
# -------------------------------------------------------------------------
def run_one(model: str, algo: str, X, y, train_idx, val_idx,
            n_trials=10, n_folds=5):
    print(f"\n[{model} / {algo}] starting...")
    t0 = time.time()

    # Special handling for ARIMA / SARIMAX — only one HPO algorithm
    # (auto_arima with AIC, the Chapter 5 protocol)
    if model in ("arima", "sarimax"):
        if algo != "auto_arima":
            print(f"  [skip] ARIMA/SARIMAX only supports auto_arima HPO")
            return None
        X_tr = X.loc[train_idx]; y_tr = y.loc[train_idx]
        trace = hpo_auto_arima(model, X_tr, y_tr, n_folds=n_folds)
        trace.to_csv(OUT / f"trace_{model}_{algo}.csv", index=False)
        winner_row = trace.iloc[0]
        winner = {"order": eval(winner_row["order"]),
                  "seasonal_order": eval(winner_row["seasonal_order"])
                  if winner_row["seasonal_order"] != "(0, 0, 0, 0)" else None}
        (OUT / f"winner_{model}_{algo}.json").write_text(
            json.dumps({k: list(v) if v else None for k, v in winner.items()},
                       indent=2))

        # Build fp + rolling forecast
        if model == "arima":
            fp = lambda Xtr, ytr, Xte: arima_fit_predict(
                Xtr, ytr, Xte, params={"order": winner["order"]})
        else:
            fp = lambda Xtr, ytr, Xte: sarimax_fit_predict(
                Xtr, ytr, Xte,
                params={"order": winner["order"],
                        "seasonal_order": winner["seasonal_order"]})
        # Fall through to common rolling-forecast block below
        algo_for_label = algo
    elif model in ("xgboost", "ann", "lstm"):
        fp = {"xgboost": xgb_fit_predict,
              "ann":     ann_fit_predict,
              "lstm":    lstm_fit_predict}[model]
        space_model = model
    else:
        fp = make_hybrid_fp(model)
        space_model = base_model_of(model)

    # Search the standalone-equivalent space (skipped for ARIMA/SARIMAX which
    # handled it above)
    if model not in ("arima", "sarimax"):
        X_tr = X.loc[train_idx]; y_tr = y.loc[train_idx]
        if algo == "grid":
            trace = hpo_grid(space_model, X_tr, y_tr, fp, n_trials, n_folds)
        elif algo == "random":
            trace = hpo_random(space_model, X_tr, y_tr, fp, n_trials, n_folds)
        elif algo == "optuna":
            trace = hpo_optuna(space_model, X_tr, y_tr, fp, n_trials, n_folds)
        else:
            raise ValueError(algo)

        trace.to_csv(OUT / f"trace_{model}_{algo}.csv", index=False)

        # Winner by min cv_RMSE
        # NOTE: pandas/numpy coerce ints to float64 when stored in DataFrame;
        # cast known-integer hyperparameters back to int for the model.
        INT_PARAMS = {"n_estimators", "max_depth", "hidden_layers",
                      "units", "batch_size", "lookback", "seed"}
        winner_row = trace.loc[trace["cv_RMSE"].idxmin()]
        winner = {}
        for k, v in winner_row.items():
            if k in ("cv_RMSE", "cv_MAPE", "cv_MAE"):
                continue
            if k in INT_PARAMS:
                winner[k] = int(round(float(v)))
            elif isinstance(v, (np.integer,)):
                winner[k] = int(v)
            elif isinstance(v, (np.floating,)):
                winner[k] = float(v)
            else:
                winner[k] = v
        (OUT / f"winner_{model}_{algo}.json").write_text(json.dumps(winner, indent=2))
        print(f"  winner: cv_RMSE={winner_row['cv_RMSE']:.3f}  "
              f"cv_MAPE={winner_row['cv_MAPE']:.2f}%")

        def fp_with_winner(Xtr, ytr, Xte):
            return fp(Xtr, ytr, Xte, params=winner)
    else:
        # ARIMA / SARIMAX already prepared trace + winner + fp above
        winner_row = trace.iloc[0]
        fp_with_winner = fp

    pred = rolling_forecast_generic(fp_with_winner, X, y, val_idx, step_days=7)
    pred["actual"] = y.loc[pred["date"]].values
    pred.to_csv(OUT / f"val_preds_{model}_{algo}.csv", index=False)

    # Daily + aggregated tables
    daily_rows = [{
        "date": r["date"].date().isoformat(),
        "actual": float(r["actual"]),
        "predicted": float(r["predicted"]),
        "abs_error": abs(r["actual"] - r["predicted"]),
        "pct_error": abs(r["actual"] - r["predicted"]) / max(r["actual"], 1e-9) * 100,
    } for _, r in pred.iterrows()]
    daily_df = pd.DataFrame(daily_rows)
    avg = {"date": "AVG",
           "actual": daily_df["actual"].mean(),
           "predicted": daily_df["predicted"].mean(),
           "abs_error": daily_df["abs_error"].mean(),
           "pct_error": daily_df["pct_error"].mean()}
    pd.concat([daily_df, pd.DataFrame([avg])], ignore_index=True
              ).to_csv(OUT / f"daily_{model}_{algo}.csv", index=False)

    weekly  = aggregate_metrics(pred.copy(), "W")
    monthly = aggregate_metrics(pred.copy(), "ME")
    yearly  = aggregate_metrics(pred.copy(), "YE")
    weekly.to_csv(OUT  / f"weekly_{model}_{algo}.csv", index=False)
    monthly.to_csv(OUT / f"monthly_{model}_{algo}.csv", index=False)
    yearly.to_csv(OUT  / f"yearly_{model}_{algo}.csv", index=False)

    m = score(daily_df["actual"].values, daily_df["predicted"].values)
    print(f"  val: MAPE={m['MAPE']:.2f}%  RMSE={m['RMSE']:.3f}  "
          f"(total {time.time()-t0:.0f}s)")
    return {
        "model": model, "algo": algo,
        "n_trials": len(trace),
        "cv_RMSE":  float(winner_row["cv_RMSE"]),
        "cv_MAPE":  float(winner_row["cv_MAPE"]),
        "cv_MAE":   float(winner_row["cv_MAE"]),
        "val_MAPE": float(m["MAPE"]),
        "val_MAE":  float(m["MAE"]),
        "val_RMSE": float(m["RMSE"]),
        "val_R2":   float(m["R2"]),
        "winner_params": json.dumps(winner),
    }


def plot_hpo_comparison(model: str, results_for_model: list[dict]):
    """3-bar chart of cv_RMSE for the 3 algorithms (when available)."""
    if len(results_for_model) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    algos = [r["algo"] for r in results_for_model]
    rmses = [r["cv_RMSE"] for r in results_for_model]
    mapes = [r["cv_MAPE"] for r in results_for_model]
    colours = {"grid": "#1f77b4", "random": "#2ca02c", "optuna": "#ff7f0e"}
    cols = [colours.get(a, "#888") for a in algos]
    bars = ax.bar(algos, rmses, color=cols, alpha=0.85)
    for bar, r in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{r:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("cv_RMSE (lower = better)")
    ax.set_title(f"{model} — HPO algorithm comparison (10 trials × 5 folds, RMSE)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / f"hpo_comparison_{model}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        help="comma-list, or 'standalone', 'hybrids', 'all'")
    parser.add_argument("--algos", default="auto",
                        help="comma-list (grid,random,optuna) or 'auto' "
                             "(3 for standalone, optuna only for hybrids)")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    classical  = ["arima", "sarimax"]
    standalone = ["xgboost", "ann", "lstm"]
    hybrids    = ["stl_xgb", "stl_ann", "stl_lstm",
                  "sarimax_xgb", "sarimax_lstm", "lstm_xgb"]
    if args.models == "classical":
        picks = classical
    elif args.models == "standalone":
        picks = standalone
    elif args.models == "hybrids":
        picks = hybrids
    elif args.models == "all":
        picks = classical + standalone + hybrids
    else:
        picks = [m.strip() for m in args.models.split(",")]

    splits, y, X, train_idx, val_idx, test_idx = load_data()
    print(f"Loaded data: y={len(y)}, X={X.shape}")
    print(f"Models: {picks}")

    summary = []
    by_model: dict[str, list] = {}
    for m in picks:
        if args.algos == "auto":
            if m in classical:
                algos = ["auto_arima"]
            elif m in standalone:
                algos = ["grid", "random", "optuna"]
            else:
                algos = ["optuna"]
        else:
            algos = [a.strip() for a in args.algos.split(",")]
        by_model[m] = []
        for a in algos:
            try:
                res = run_one(m, a, X, y, train_idx, val_idx,
                              n_trials=args.n_trials, n_folds=args.n_folds)
                summary.append(res); by_model[m].append(res)
                pd.DataFrame(summary).to_csv(OUT / "summary_phase2.csv",
                                              index=False)
            except Exception as exc:
                import traceback
                print(f"  FAILED {m} / {a}: {exc}")
                traceback.print_exc()
        plot_hpo_comparison(m, by_model[m])

    print("\n" + "="*70)
    print("PHASE 2 SUMMARY")
    print("="*70)
    if summary:
        df = pd.DataFrame(summary)
        print(df[["model","algo","cv_RMSE","cv_MAPE","val_MAPE","val_RMSE"]
                 ].to_string(index=False))


if __name__ == "__main__":
    main()
