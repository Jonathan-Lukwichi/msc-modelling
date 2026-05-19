"""Plan §12 Step 7: residual hybrids (3) + STL hybrids (3).

  Residual (Zhang Alg 6): SARIMAX+XGB, SARIMAX+LSTM, LSTM+XGB
  STL (Alg 7):            STL+XGB, STL+ANN, STL+LSTM

Each hybrid is fitted ONCE on the training fold (no rolling refit of the hybrid
itself — the base SARIMAX uses its rolling val predictions from artefacts; the
refiner is trained on training-fold residuals only). This follows the plan §12.2
coding prompt and avoids re-running the 12-minute SARIMAX rolling refit per
hybrid.

Critical leakage guard: refiners see only training-fold residuals.
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
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score
from src.forecasting.hybrids import residual as R
from src.forecasting.hybrids import stl_hybrid as S


def align_y_to_X(y: pd.Series, X: pd.DataFrame):
    df = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    return df["y"], df.drop(columns=["y"])


def load_sarimax_order():
    p = ROOT / "artefacts" / "models" / "sarimax_order.txt"
    cfg = {}
    for line in p.read_text().strip().splitlines():
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    return eval(cfg["order"]), eval(cfg["seasonal_order"])


def load_json_params(name: str):
    return json.loads((ROOT / "artefacts" / "models" / f"{name}_best_params.json").read_text())


def write_hybrid_artefacts(name: str, val_pred: pd.DataFrame,
                            metrics: dict, train_residuals: pd.Series | None = None):
    out_pred = ROOT / "artefacts" / "predictions" / f"hybrid_{name}.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / f"hybrid_{name}_metrics.csv"
    for p in (out_pred, out_metrics):
        p.parent.mkdir(parents=True, exist_ok=True)
    val_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{"block": "val", **metrics}]).to_csv(out_metrics, index=False)
    if train_residuals is not None:
        train_residuals.to_csv(
            ROOT / "artefacts" / "metrics" / f"hybrid_{name}_train_residuals.csv",
            header=True,
        )
    print(f"  Wrote: {out_pred.relative_to(ROOT)}  /  {out_metrics.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Residual hybrids
# ---------------------------------------------------------------------------

def run_sarimax_xgb(target, X_train, X_aligned, train_idx, val_idx):
    print("\n=== Residual hybrid: SARIMAX + XGBoost ===")
    t0 = time.time()
    order, seasonal_order = load_sarimax_order()
    print(f"  Refitting SARIMAX on train ({order} x {seasonal_order})...")
    train_resid = R.sarimax_train_residuals(target, X_train, order, seasonal_order)
    print(f"  Train residuals: n={len(train_resid)}, "
          f"mean={train_resid.mean():.2f}, std={train_resid.std():.2f}")
    print("  Fitting XGBoost refiner on residuals...")
    refiner = R.fit_xgb_refiner(X_train, train_resid)
    sarimax_val = R._load_val_predictions("sarimax")
    refiner_val = R.xgb_refiner_predict(refiner, X_aligned.loc[val_idx])
    combined = sarimax_val.loc[val_idx] + refiner_val
    val_pred = pd.DataFrame({"predicted": combined,
                              "actual": target.loc[val_idx],
                              "block": "val"})
    metrics = score(val_pred["actual"], val_pred["predicted"])
    print(f"  Val metrics: MAPE={metrics['MAPE']:.3f}  MAE={metrics['MAE']:.3f}  "
          f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")
    print(f"  Took {time.time() - t0:.1f}s")
    write_hybrid_artefacts("sarimax_xgb", val_pred, metrics, train_resid)


def run_sarimax_lstm(target, X_train, X_aligned, train_idx, val_idx):
    print("\n=== Residual hybrid: SARIMAX + LSTM ===")
    t0 = time.time()
    order, seasonal_order = load_sarimax_order()
    print(f"  Refitting SARIMAX on train ({order} x {seasonal_order})...")
    train_resid = R.sarimax_train_residuals(target, X_train, order, seasonal_order)
    print("  Fitting LSTM refiner on residuals (light defaults)...")
    refiner = R.fit_lstm_refiner(X_train, train_resid)
    sarimax_val = R._load_val_predictions("sarimax")
    refiner_val = R.lstm_refiner_predict(refiner, X_aligned, val_idx)
    combined = sarimax_val.reindex(val_idx) + refiner_val.reindex(val_idx).fillna(0)
    val_pred = pd.DataFrame({"predicted": combined,
                              "actual": target.loc[val_idx],
                              "block": "val"})
    metrics = score(val_pred["actual"], val_pred["predicted"])
    print(f"  Val metrics: MAPE={metrics['MAPE']:.3f}  MAE={metrics['MAE']:.3f}  "
          f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")
    print(f"  Took {time.time() - t0:.1f}s")
    write_hybrid_artefacts("sarimax_lstm", val_pred, metrics, train_resid)


def run_lstm_xgb(target, X_train, X_aligned, train_idx, val_idx):
    print("\n=== Residual hybrid: LSTM + XGBoost ===")
    t0 = time.time()
    lstm_params = load_json_params("lstm")
    print(f"  Refitting LSTM on train with params {lstm_params}...")
    in_sample, _ = R.lstm_train_in_sample(target, X_train, lstm_params)
    train_resid = target.loc[in_sample.index] - in_sample
    train_resid.name = "residual"
    print(f"  Train residuals: n={len(train_resid)}, "
          f"mean={train_resid.mean():.2f}, std={train_resid.std():.2f}")
    print("  Fitting XGBoost refiner on residuals...")
    refiner = R.fit_xgb_refiner(X_train.loc[train_resid.index], train_resid)

    lstm_val = R._load_val_predictions("lstm")
    refiner_val = R.xgb_refiner_predict(refiner, X_aligned.loc[val_idx])
    combined = lstm_val.reindex(val_idx) + refiner_val
    val_pred = pd.DataFrame({"predicted": combined,
                              "actual": target.loc[val_idx],
                              "block": "val"})
    metrics = score(val_pred["actual"], val_pred["predicted"])
    print(f"  Val metrics: MAPE={metrics['MAPE']:.3f}  MAE={metrics['MAE']:.3f}  "
          f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")
    print(f"  Took {time.time() - t0:.1f}s")
    write_hybrid_artefacts("lstm_xgb", val_pred, metrics, train_resid)


# ---------------------------------------------------------------------------
# STL hybrids
# ---------------------------------------------------------------------------

def run_stl_hybrid(target, X_train, X_aligned, train_idx, val_idx,
                    refiner_kind: str):
    name = f"stl_{refiner_kind}"
    print(f"\n=== STL hybrid: STL + {refiner_kind.upper()} ===")
    t0 = time.time()
    y_train = target.loc[train_idx]
    decomp = S.decompose_train(y_train, period=7)
    print(f"  STL decomposition: trend SD={decomp.trend.std():.2f}, "
          f"seasonal SD={decomp.seasonal.std():.2f}, "
          f"residual SD={decomp.residual.std():.2f}")
    trend_fc = S.forecast_trend(decomp.trend, val_idx)
    seasonal_fc = S.forecast_seasonal(decomp.seasonal, val_idx)

    if refiner_kind == "xgb":
        refiner = S.fit_xgb_refiner_on_residual(X_train, decomp.residual)
    elif refiner_kind == "ann":
        refiner = S.fit_ann_refiner_on_residual(X_train, decomp.residual)
    elif refiner_kind == "lstm":
        refiner = S.fit_lstm_refiner_on_residual(X_train, decomp.residual)
    else:
        raise ValueError(refiner_kind)

    residual_fc = S.refiner_predict_val(refiner_kind, refiner, X_aligned, val_idx)
    combined = trend_fc + seasonal_fc + residual_fc.reindex(val_idx).fillna(0)
    val_pred = pd.DataFrame({"predicted": combined,
                              "actual": target.loc[val_idx],
                              "block": "val"})
    metrics = score(val_pred["actual"], val_pred["predicted"])
    print(f"  Val metrics: MAPE={metrics['MAPE']:.3f}  MAE={metrics['MAE']:.3f}  "
          f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")
    print(f"  Took {time.time() - t0:.1f}s")
    write_hybrid_artefacts(name, val_pred, metrics, decomp.residual)
    # Also save the STL components once
    if refiner_kind == "xgb":
        decomp_df = pd.DataFrame({
            "trend": decomp.trend,
            "seasonal": decomp.seasonal,
            "residual": decomp.residual,
        })
        decomp_df.to_csv(ROOT / "artefacts" / "metrics" / "stl_decomposition.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    eng = load_engineered()
    X_all = build_selected_X(eng)
    y_aligned, X_aligned = align_y_to_X(target, X_all)

    train_idx = splits.slice(g1, "train").index.intersection(y_aligned.index)
    val_idx = splits.slice(g1, "val").index.intersection(y_aligned.index)
    X_train = X_aligned.loc[train_idx]
    print(f"Train: {len(train_idx)} days  |  Val: {len(val_idx)} days")
    print(f"X_aligned columns: {len(X_aligned.columns)}  "
          f"(consensus-selected feature set)")

    # Residual hybrids
    run_sarimax_xgb(target, X_train, X_aligned, train_idx, val_idx)
    run_sarimax_lstm(target, X_train, X_aligned, train_idx, val_idx)

    lstm_path = ROOT / "artefacts" / "predictions" / "lstm.csv"
    if lstm_path.exists():
        run_lstm_xgb(target, X_train, X_aligned, train_idx, val_idx)
    else:
        print("\n  LSTM standalone results not found; skipping LSTM+XGB hybrid")
        print("  (Run scripts/08_lstm.py first.)")

    # STL hybrids
    run_stl_hybrid(target, X_train, X_aligned, train_idx, val_idx, "xgb")
    run_stl_hybrid(target, X_train, X_aligned, train_idx, val_idx, "ann")
    run_stl_hybrid(target, X_train, X_aligned, train_idx, val_idx, "lstm")

    print("\nAll requested hybrids done.")


if __name__ == "__main__":
    main()
