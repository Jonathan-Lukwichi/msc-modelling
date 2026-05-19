"""Plan §6 Step 1: reference floor baselines.

Runs naive_yest / naive_seasonal / dow_mean on val and test, writes metrics
and predictions to artefacts/.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Allow running as a script from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.metrics import score
from src.forecasting.models.naive import (
    predict_naive_yest,
    predict_naive_seasonal,
    predict_dow_mean,
)


def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    train = splits.slice(g1, "train")["total_daily_arrivals"]
    val = splits.slice(g1, "val")["total_daily_arrivals"]
    test = splits.slice(g1, "test")["total_daily_arrivals"]

    # Naive shifts operate on the full (filtered, contiguous) timeline so we
    # can recover values from prior blocks at block boundaries.
    yhat_yest = predict_naive_yest(target)
    yhat_seas = predict_naive_seasonal(target, lag=7)
    yhat_dow = predict_dow_mean(train, target.index)

    rows = []
    predictions = []
    for block_name, block_series in (("val", val), ("test", test)):
        idx = block_series.index
        for baseline_name, yhat_full in (
            ("naive_yest", yhat_yest),
            ("naive_seasonal", yhat_seas),
            ("dow_mean", yhat_dow),
        ):
            yhat = yhat_full.reindex(idx)
            metrics = score(block_series, yhat)
            row = {"block": block_name, "baseline": baseline_name, **metrics}
            rows.append(row)
            for date, actual, predicted in zip(idx, block_series.values, yhat.values):
                predictions.append({
                    "date": date,
                    "block": block_name,
                    "baseline": baseline_name,
                    "actual": actual,
                    "predicted": predicted,
                })

    metrics_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(predictions)

    out_metrics = ROOT / "artefacts" / "metrics" / "reference_floor.csv"
    out_pred = ROOT / "artefacts" / "predictions" / "reference_floor.csv"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_metrics, index=False)
    pred_df.to_csv(out_pred, index=False)

    print("Reference floor metrics:")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(metrics_df.to_string(index=False))
    print(f"\nWrote: {out_metrics.relative_to(ROOT)}")
    print(f"Wrote: {out_pred.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
