"""Plan §11.4 Step 6c: LSTM standalone on §3.4.3 consensus features.

Optuna TPE 30 trials, 60-min time budget. Best by val MAPE; then rolling-origin
weekly refit on val.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score
from src.forecasting.models.lstm import tpe_search_cv, rolling_forecast


def align_y_to_X(y: pd.Series, X: pd.DataFrame):
    df = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    return df["y"], df.drop(columns=["y"])


def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    eng = load_engineered()
    X_all = build_selected_X(eng)
    y_aligned, X_aligned = align_y_to_X(target, X_all)

    train_idx = splits.slice(g1, "train").index.intersection(y_aligned.index)
    val_idx = splits.slice(g1, "val").index.intersection(y_aligned.index)
    print(f"Train: {len(train_idx)} days  |  Val: {len(val_idx)} days")

    X_train, y_train = X_aligned.loc[train_idx], y_aligned.loc[train_idx]
    # val/test are NOT consumed in HPO; the val held-out check happens after refit.

    print("\nLSTM TPE search via inner rolling-origin CV "
          "(15 trials x 10 folds, 180-minute budget)...")
    t0 = time.time()
    best, trace = tpe_search_cv(X_train, y_train,
                                  n_trials=15, n_folds=10,
                                  time_budget_minutes=180, seed=42)
    print(f"  HPO took {time.time() - t0:.1f}s, {len(trace)} trials")
    print(f"  Best params: {best.params}")
    print(f"  Best CV MAPE: {best.val_mape:.3f}")
    print(f"  (val block not consumed in HPO; reserved for held-out check below)")

    print("\nRunning rolling-origin LSTM forecast on val (weekly refit)...")
    t0 = time.time()
    val_pred = rolling_forecast(X_aligned, y_aligned, val_idx,
                                 params=best.params, step_days=7, seed=42)
    print(f"  Val rolling forecast took {time.time() - t0:.1f}s ({len(val_pred)} rows)")

    val_pred = val_pred.set_index("date")
    val_pred["actual"] = y_aligned.loc[val_pred.index]
    val_pred["block"] = "val"
    val_metrics = score(val_pred["actual"], val_pred["predicted"])

    print("\nVal metrics (rolling refit):")
    for k, v in val_metrics.items():
        print(f"  {k:>4s}: {v:.4f}")

    out_pred = ROOT / "artefacts" / "predictions" / "lstm.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "lstm_metrics.csv"
    out_best = ROOT / "artefacts" / "models" / "lstm_best_params.json"
    out_trace = ROOT / "artefacts" / "metrics" / "lstm_hpo_trace.csv"
    for p in (out_pred, out_metrics, out_best, out_trace):
        p.parent.mkdir(parents=True, exist_ok=True)

    val_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{"block": "val", **val_metrics}]).to_csv(out_metrics, index=False)
    out_best.write_text(json.dumps(best.params, indent=2))
    trace.to_csv(out_trace, index=False)
    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")
    print(f"Wrote: {out_best.relative_to(ROOT)}")
    print(f"Wrote: {out_trace.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
