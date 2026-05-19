"""Plan §11.2 Step 6a: XGBoost standalone on the §3.4.3 consensus features.

Procedure:
  1. Load engineered matrix (upstream) -> select consensus features -> 23 cols.
  2. Build target y aligned to G1 with is_zero_day filtered.
  3. Grid HPO: fit on train, evaluate on val. Best by val MAPE.
  4. Refit on full train and run rolling-origin weekly refit on val.
  5. Persist predictions, metrics, best params, HPO trace, SHAP importance.
"""
from __future__ import annotations

from pathlib import Path
import json
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
from src.forecasting.models.xgboost_m import (
    grid_search_cv, rolling_forecast, shap_summary, GRID,
)


def align_y_to_X(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join on date index; X already has lags so dropna leaves the
    common usable window."""
    df = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    return df["y"], df.drop(columns=["y"])


def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    eng = load_engineered()
    X_all = build_selected_X(eng)  # 23 consensus features
    print(f"Engineered + consensus matrix: {X_all.shape}")

    # Align target to X (X has lags; drop the first 30 days of warmup)
    y_aligned, X_aligned = align_y_to_X(target, X_all)
    print(f"Aligned (target, X): {len(y_aligned)} rows")

    train_idx = splits.slice(g1, "train").index.intersection(y_aligned.index)
    val_idx = splits.slice(g1, "val").index.intersection(y_aligned.index)
    print(f"Train: {len(train_idx)} days  |  Val: {len(val_idx)} days")

    X_train, y_train = X_aligned.loc[train_idx], y_aligned.loc[train_idx]
    # val/test are NOT consumed during HPO; the val held-out check happens
    # after refit via the rolling_forecast call below.

    # HPO via inner rolling-origin CV inside train block (§3.6.1)
    n_combos = (len(GRID["n_estimators"]) * len(GRID["max_depth"])
                * len(GRID["learning_rate"]) * len(GRID["subsample"]))
    n_folds = 10
    print(f"\nGrid search via inner rolling-origin CV: "
          f"{n_combos} combos x {n_folds} folds = {n_combos * n_folds} fits...")
    t0 = time.time()
    best, trace = grid_search_cv(X_train, y_train, grid=GRID,
                                   n_folds=n_folds, seed=42, verbose=False)
    print(f"  HPO took {time.time() - t0:.1f}s")
    print(f"  Best params: {best.params}")
    print(f"  Best CV mean: MAPE={best.val_mape:.3f}  MAE={best.val_mae:.3f}  "
          f"RMSE={best.val_rmse:.3f}  R2={best.val_r2:.3f}")
    print(f"  (val block not consumed in HPO; reserved for held-out check below)")

    # Rolling-origin weekly refit with best params
    print("\nRunning rolling-origin XGBoost forecast on val (weekly refit)...")
    t0 = time.time()
    val_pred = rolling_forecast(X_aligned, y_aligned, val_idx,
                                 params=best.params, step_days=7, seed=42)
    print(f"  Val rolling forecast took {time.time() - t0:.1f}s "
          f"({len(val_pred)} rows)")

    val_pred = val_pred.set_index("date")
    val_pred["actual"] = y_aligned.loc[val_pred.index]
    val_pred["block"] = "val"
    val_metrics = score(val_pred["actual"], val_pred["predicted"])

    print("\nVal metrics (rolling refit):")
    for k, v in val_metrics.items():
        print(f"  {k:>4s}: {v:.4f}")

    # SHAP on the training fold for Figure 6.6
    print("\nComputing mean |SHAP| on train fold...")
    shap_df = shap_summary(X_train, y_train, best.params, seed=42)
    print("Top 10 by mean |SHAP|:")
    print(shap_df.head(10).to_string(index=False))

    out_pred = ROOT / "artefacts" / "predictions" / "xgboost.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "xgboost_metrics.csv"
    out_best = ROOT / "artefacts" / "models" / "xgboost_best_params.json"
    out_trace = ROOT / "artefacts" / "metrics" / "xgboost_hpo_trace.csv"
    out_shap = ROOT / "artefacts" / "metrics" / "xgboost_shap.csv"
    for p in (out_pred, out_metrics, out_best, out_trace, out_shap):
        p.parent.mkdir(parents=True, exist_ok=True)

    val_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{"block": "val", **val_metrics}]).to_csv(out_metrics,
                                                           index=False)
    out_best.write_text(json.dumps(best.params, indent=2))
    trace.to_csv(out_trace, index=False)
    shap_df.to_csv(out_shap, index=False)

    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")
    print(f"Wrote: {out_best.relative_to(ROOT)}")
    print(f"Wrote: {out_trace.relative_to(ROOT)}")
    print(f"Wrote: {out_shap.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
