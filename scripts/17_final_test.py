"""Plan §17 — single OOD test pass on the 396-day test block.

Per the methodology: test (2025-01-01 to 2026-01-31) is touched EXACTLY ONCE
after every val number is finalised. This script does that, then breaks the
results down by quarter, by month, and by horizon — exposing where the
+18.3 % drift documented in Ch5 §5.5.2 actually bites.

Saves:
  - artefacts/predictions/test/{model}.csv
  - artefacts/metrics/test_aggregate.csv     (one row per model)
  - artefacts/metrics/test_per_quarter.csv   (model x quarter)
  - artefacts/metrics/test_per_month.csv     (model x month)
  - artefacts/metrics/test_per_horizon.csv   (model x horizon)
  - artefacts/figures/fig_6_test_by_quarter.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")

OUT_PRED = ROOT / "artefacts" / "predictions" / "test"
OUT_METRICS = ROOT / "artefacts" / "metrics"
OUT_PRED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Test block + horizon helper
# ---------------------------------------------------------------------------

splits = Splits.from_config()
TEST_START = splits.test_start


def horizon_for_date(d: pd.Timestamp) -> int:
    """Day index within each weekly refit window (1..7)."""
    return ((d - TEST_START).days % 7) + 1


# ---------------------------------------------------------------------------
# Per-model rolling test forecasts
# ---------------------------------------------------------------------------

def naive_baselines(target: pd.Series, test_idx: pd.DatetimeIndex,
                     train_idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    out = {}
    out["naive_yest"] = target.shift(1).reindex(test_idx)
    out["naive_seasonal"] = target.shift(7).reindex(test_idx)
    weekday_means = target.loc[train_idx].groupby(
        target.loc[train_idx].index.dayofweek).mean()
    out["dow_mean"] = pd.Series(
        [weekday_means[d.dayofweek] for d in test_idx], index=test_idx)
    return out


def rolling_arima(target, test_idx, step=7):
    """Roll ARIMA(0,1,2) weekly across test."""
    from pmdarima import ARIMA as PmARIMA
    order = (0, 1, 2)
    full = target.copy()
    rows = []
    origin_pos = full.index.get_loc(test_idx[0]) - 1
    last_pos = full.index.get_loc(test_idx[-1])
    while origin_pos < last_pos:
        h = int(min(step, last_pos - origin_pos))
        m = PmARIMA(order=order, suppress_warnings=True)
        m.fit(full.iloc[: origin_pos + 1].values)
        yhat = m.predict(n_periods=h)
        for d, yp in zip(full.index[origin_pos + 1 : origin_pos + 1 + h], yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin_pos += step
    return pd.DataFrame(rows).set_index("date")["predicted"]


def rolling_sarimax(target, X_full, test_idx, step=7):
    from pmdarima import ARIMA as PmARIMA
    order, seasonal_order = (1, 1, 1), (0, 1, 1, 7)
    full = target.copy()
    rows = []
    origin_pos = full.index.get_loc(test_idx[0]) - 1
    last_pos = full.index.get_loc(test_idx[-1])
    while origin_pos < last_pos:
        h = int(min(step, last_pos - origin_pos))
        y_tr = full.iloc[: origin_pos + 1]
        X_tr = X_full.loc[y_tr.index]
        X_fu = X_full.iloc[origin_pos + 1 : origin_pos + 1 + h]
        m = PmARIMA(order=order, seasonal_order=seasonal_order,
                    suppress_warnings=True)
        m.fit(y_tr.values, X=X_tr.values)
        yhat = m.predict(n_periods=h, X=X_fu.values)
        for d, yp in zip(X_fu.index, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin_pos += step
    return pd.DataFrame(rows).set_index("date")["predicted"]


def rolling_nbglm(target, X_full, test_idx, step=7):
    """NB GLM with §5.2.5 exog + y_{t-7}; weekly refit."""
    import statsmodels.api as sm
    full_with_lag = pd.concat(
        [target.rename("y"), X_full, target.shift(7).rename("y_lag7")], axis=1)
    rows = []
    origin_pos = target.index.get_loc(test_idx[0]) - 1
    last_pos = target.index.get_loc(test_idx[-1])
    while origin_pos < last_pos:
        h = int(min(step, last_pos - origin_pos))
        tr = full_with_lag.iloc[: origin_pos + 1].dropna()
        y_tr = tr["y"].values
        X_tr = sm.add_constant(tr.drop(columns=["y"]).values)
        # Estimate alpha
        pois = sm.GLM(y_tr, X_tr, family=sm.families.Poisson()).fit(
            method="lbfgs", disp=0)
        mu = pois.predict()
        pearson = (y_tr - mu) / np.sqrt(np.maximum(mu, 1e-6))
        alpha = max(float(np.sum(pearson ** 2) / max(len(y_tr) - X_tr.shape[1], 1) - 1.0), 0.05)
        nb = sm.GLM(y_tr, X_tr, family=sm.families.NegativeBinomial(alpha=alpha)).fit(
            method="lbfgs", disp=0)
        future_idx = target.index[origin_pos + 1 : origin_pos + 1 + h]
        Xf = full_with_lag.loc[future_idx].drop(columns=["y"]).values
        Xfc = sm.add_constant(Xf, has_constant="add")
        if Xfc.shape[1] != X_tr.shape[1]:
            Xfc = np.column_stack([np.ones(len(Xf)), Xf])
        yhat = nb.predict(Xfc)
        for d, yp in zip(future_idx, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin_pos += step
    return pd.DataFrame(rows).set_index("date")["predicted"]


def rolling_xgboost(target, X_full, test_idx, step=7):
    """XGBoost with best params from CV; weekly refit."""
    from xgboost import XGBRegressor
    params = json.loads(
        (ROOT / "artefacts" / "models" / "xgboost_best_params.json").read_text())
    df = pd.concat([target.rename("y"), X_full], axis=1, join="inner").dropna()
    rows = []
    origin = df.index.get_loc(test_idx[0]) - 1
    last_pos = df.index.get_loc(test_idx[-1])
    while origin < last_pos:
        h = int(min(step, last_pos - origin))
        tr = df.iloc[: origin + 1]
        future_idx = df.index[origin + 1 : origin + 1 + h]
        m = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=42, verbosity=0, n_jobs=-1)
        m.fit(tr.drop(columns=["y"]).values, tr["y"].values)
        yhat = m.predict(df.loc[future_idx].drop(columns=["y"]).values)
        for d, yp in zip(future_idx, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin += step
    return pd.DataFrame(rows).set_index("date")["predicted"]


# ---------------------------------------------------------------------------
# Per-block metric breakdown
# ---------------------------------------------------------------------------

def breakdown_metrics(preds_df: pd.DataFrame, model_name: str) -> dict:
    """preds_df has columns: date, actual, predicted. Compute per-quarter,
    per-month, per-horizon metrics."""
    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df["quarter"] = preds_df["date"].dt.to_period("Q").astype(str)
    preds_df["month"] = preds_df["date"].dt.to_period("M").astype(str)
    preds_df["horizon"] = preds_df["date"].apply(horizon_for_date)

    aggregate = {"model": model_name, "n": len(preds_df),
                 **score(preds_df["actual"], preds_df["predicted"])}

    pq = []
    for q, sub in preds_df.groupby("quarter"):
        pq.append({"model": model_name, "quarter": q, "n": len(sub),
                    **score(sub["actual"], sub["predicted"])})

    pm = []
    for m, sub in preds_df.groupby("month"):
        pm.append({"model": model_name, "month": m, "n": len(sub),
                    **score(sub["actual"], sub["predicted"])})

    ph = []
    for h, sub in preds_df.groupby("horizon"):
        ph.append({"model": model_name, "horizon": int(h), "n": len(sub),
                    **score(sub["actual"], sub["predicted"])})

    return {"agg": aggregate, "quarter": pq, "month": pm, "horizon": ph}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(models_to_run: list[str]) -> None:
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    train_idx = splits.slice(g1, "train").index
    test_idx = splits.slice(g1, "test").index

    print(f"Test block: {test_idx[0].date()} -> {test_idx[-1].date()}  "
          f"({len(test_idx)} days)")
    print(f"Test mean: {target.loc[test_idx].mean():.2f}  "
          f"(train mean: {target.loc[train_idx].mean():.2f},  "
          f"+{(target.loc[test_idx].mean() / target.loc[train_idx].mean() - 1) * 100:.1f}% drift)\n")

    # Build exog matrices once
    X_train, scaler = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    X_full, _ = build_task1_exogenous(g1, scaler=scaler)
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    df_consensus = pd.concat([target.rename("y"), X_consensus],
                               axis=1, join="inner").dropna()

    all_agg, all_pq, all_pm, all_ph = [], [], [], []

    def run_and_record(name: str, predicted: pd.Series):
        predicted = predicted.reindex(test_idx).dropna()
        actual = target.loc[predicted.index]
        df_out = pd.DataFrame({"date": predicted.index,
                                "actual": actual.values,
                                "predicted": predicted.values})
        df_out.to_csv(OUT_PRED / f"{name}.csv", index=False)
        b = breakdown_metrics(df_out, name)
        all_agg.append(b["agg"])
        all_pq.extend(b["quarter"])
        all_pm.extend(b["month"])
        all_ph.extend(b["horizon"])
        print(f"  {name:>18s}: MAPE={b['agg']['MAPE']:6.2f}%  "
              f"MAE={b['agg']['MAE']:6.2f}  RMSE={b['agg']['RMSE']:6.2f}  "
              f"R2={b['agg']['R2']:+5.2f}")

    if "naive" in models_to_run or "all" in models_to_run:
        print("\n--- Naive baselines ---")
        baselines = naive_baselines(target, test_idx, train_idx)
        for nm, p in baselines.items():
            run_and_record(nm, p)

    if "arima" in models_to_run or "all" in models_to_run:
        print("\n--- ARIMA(0,1,2) rolling weekly refit ---")
        t0 = time.time()
        p = rolling_arima(target, test_idx)
        print(f"  ({time.time() - t0:.1f}s)")
        run_and_record("arima", p)

    if "nbglm" in models_to_run or "all" in models_to_run:
        print("\n--- NB GLM rolling weekly refit ---")
        t0 = time.time()
        p = rolling_nbglm(target, X_full, test_idx)
        print(f"  ({time.time() - t0:.1f}s)")
        run_and_record("nbglm", p)

    if "xgboost" in models_to_run or "all" in models_to_run:
        print("\n--- XGBoost rolling weekly refit ---")
        t0 = time.time()
        p = rolling_xgboost(target, X_consensus, test_idx)
        print(f"  ({time.time() - t0:.1f}s)")
        run_and_record("xgboost", p)

    if "sarimax" in models_to_run or "all" in models_to_run:
        print("\n--- SARIMAX(1,1,1)(0,1,1,7) rolling weekly refit (slow) ---")
        t0 = time.time()
        p = rolling_sarimax(target, X_full, test_idx)
        print(f"  ({time.time() - t0:.1f}s)")
        run_and_record("sarimax", p)

    # Save artefacts
    if all_agg:
        pd.DataFrame(all_agg).to_csv(OUT_METRICS / "test_aggregate.csv", index=False)
        pd.DataFrame(all_pq).to_csv(OUT_METRICS / "test_per_quarter.csv", index=False)
        pd.DataFrame(all_pm).to_csv(OUT_METRICS / "test_per_month.csv", index=False)
        pd.DataFrame(all_ph).to_csv(OUT_METRICS / "test_per_horizon.csv", index=False)
        print(f"\nWrote: artefacts/metrics/test_aggregate.csv ({len(all_agg)} models)")
        print(f"Wrote: artefacts/metrics/test_per_quarter.csv ({len(all_pq)} rows)")
        print(f"Wrote: artefacts/metrics/test_per_month.csv ({len(all_pm)} rows)")
        print(f"Wrote: artefacts/metrics/test_per_horizon.csv ({len(all_ph)} rows)")

        # Print key tables
        print("\n=== Test MAPE by quarter (drift exposure) ===")
        pq_df = pd.DataFrame(all_pq).pivot(index="model", columns="quarter",
                                             values="MAPE")
        pd.set_option("display.float_format", lambda v: f"{v:.2f}")
        print(pq_df.to_string())

        print("\n=== Test MAPE by horizon ===")
        ph_df = pd.DataFrame(all_ph).pivot(index="model", columns="horizon",
                                             values="MAPE")
        print(ph_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["all"],
                        help="Subset of: naive, arima, nbglm, xgboost, sarimax (default: all)")
    args = parser.parse_args()
    main(args.models)
