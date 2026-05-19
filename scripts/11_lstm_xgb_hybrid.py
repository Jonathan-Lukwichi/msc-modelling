"""Plan §12 Step 7 (6th hybrid): LSTM + XGBoost residual hybrid.

Runs only LSTM+XGB so we don't waste time re-doing the 5 hybrids already cached.
Requires:
  - artefacts/predictions/lstm.csv (from scripts/08_lstm.py)
  - artefacts/models/lstm_best_params.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score
from src.forecasting.hybrids import residual as R


def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    eng = load_engineered()
    X_all = build_selected_X(eng)
    df = pd.concat([target.rename("y"), X_all], axis=1, join="inner").dropna()
    y_aligned = df["y"]
    X_aligned = df.drop(columns=["y"])

    train_idx = splits.slice(g1, "train").index.intersection(y_aligned.index)
    val_idx = splits.slice(g1, "val").index.intersection(y_aligned.index)
    X_train = X_aligned.loc[train_idx]
    print(f"Train: {len(train_idx)} days  |  Val: {len(val_idx)} days")

    lstm_params = json.loads(
        (ROOT / "artefacts" / "models" / "lstm_best_params.json").read_text())
    print(f"LSTM best params: {lstm_params}")

    print("\n=== Residual hybrid: LSTM + XGBoost ===")
    t0 = time.time()
    in_sample, _ = R.lstm_train_in_sample(target, X_train, lstm_params)
    train_resid = target.loc[in_sample.index] - in_sample
    train_resid.name = "residual"
    sigma = train_resid.std()
    train_resid = train_resid[train_resid.abs() <= 5 * sigma]
    print(f"  Train residuals: n={len(train_resid)}, "
          f"mean={train_resid.mean():.2f}, std={train_resid.std():.2f}")

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

    out_pred = ROOT / "artefacts" / "predictions" / "hybrid_lstm_xgb.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "hybrid_lstm_xgb_metrics.csv"
    for p in (out_pred, out_metrics):
        p.parent.mkdir(parents=True, exist_ok=True)
    val_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{"block": "val", **metrics}]).to_csv(out_metrics, index=False)
    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
