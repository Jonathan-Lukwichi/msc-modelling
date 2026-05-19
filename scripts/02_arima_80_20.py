"""Sensitivity check: ARIMA with a plain chronological 80/20 split.

Standalone — does NOT touch the §5.5.2 train/val/test split used elsewhere.
Re-uses src.forecasting.models.arima for order selection and rolling forecast
so the modelling logic is identical to scripts/02_arima.py; only the split
boundary changes.

Writes:
  artefacts/predictions/arima_80_20.csv
  artefacts/metrics/arima_80_20_metrics.csv
  artefacts/models/arima_80_20_order.txt
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1
from src.forecasting.metrics import score
from src.forecasting.models.arima import (
    pick_order, rolling_forecast, residual_diagnostics,
)


def main() -> None:
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    n = len(target)
    cut = int(round(n * 0.80))

    train = target.iloc[:cut]
    test_idx = target.iloc[cut:].index

    print("Chronological 80/20 split (full G1 zero-day-filtered):")
    print(f"  total days : {n}")
    print(f"  train days : {len(train):>4d}  "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"  test  days : {len(test_idx):>4d}  "
          f"({test_idx[0].date()} -> {test_idx[-1].date()})")

    print("\nPicking ARIMA order via stepwise AIC on the 80% train fold...")
    t0 = time.time()
    result = pick_order(train, max_p=3, max_q=3, d=1, seed=42)
    print(f"  Picked order: {result.order}, AIC = {result.aic:.2f}")
    print(f"  Order selection took {time.time() - t0:.1f}s")

    diag = residual_diagnostics(result.fitted_train.resid())
    print("\nResidual diagnostics (initial train fit):")
    for k, v in diag.items():
        print(f"  {k:>20s}: {v:.4f}")

    print("\nRunning rolling-origin forecast on the 20% test fold (weekly refit)...")
    t0 = time.time()
    test_pred = rolling_forecast(target, test_idx, result.order, step_days=7)
    print(f"  Test forecast took {time.time() - t0:.1f}s ({len(test_pred)} rows)")

    test_pred = test_pred.set_index("date")
    test_pred["actual"] = target.loc[test_pred.index]
    test_pred["block"] = "test_80_20"
    test_metrics = score(test_pred["actual"], test_pred["predicted"])

    print("\nTest metrics (80/20 chronological split):")
    for k, v in test_metrics.items():
        print(f"  {k:>4s}: {v:.4f}")

    out_pred = ROOT / "artefacts" / "predictions" / "arima_80_20.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "arima_80_20_metrics.csv"
    out_order = ROOT / "artefacts" / "models" / "arima_80_20_order.txt"
    for p in (out_pred, out_metrics, out_order):
        p.parent.mkdir(parents=True, exist_ok=True)

    test_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{"block": "test_80_20", **test_metrics}]).to_csv(out_metrics, index=False)
    out_order.write_text(
        f"order={result.order}\naic={result.aic:.4f}\n"
        f"train_days={len(train)}\ntest_days={len(test_idx)}\n"
        f"train_start={train.index[0].date()}\ntrain_end={train.index[-1].date()}\n"
        f"test_start={test_idx[0].date()}\ntest_end={test_idx[-1].date()}\n"
    )

    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")
    print(f"Wrote: {out_order.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
