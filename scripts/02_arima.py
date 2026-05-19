"""Plan §7 Step 2: ARIMA baseline.

Picks (p, 1, q) via stepwise AIC on the train fold, runs residual diagnostics,
then forecasts val (and test) with rolling-origin weekly refit.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.metrics import score
from src.forecasting.models.arima import (
    pick_order, rolling_forecast, residual_diagnostics,
)


def main(do_test: bool = False) -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    train = splits.slice(g1, "train")["total_daily_arrivals"]
    val_idx = splits.slice(g1, "val")["total_daily_arrivals"].index
    test_idx = splits.slice(g1, "test")["total_daily_arrivals"].index

    print("Picking ARIMA order via stepwise AIC...")
    t0 = time.time()
    result = pick_order(train, max_p=3, max_q=3, d=1, seed=42)
    print(f"  Picked order: {result.order}, AIC = {result.aic:.2f}")
    print(f"  Order selection took {time.time() - t0:.1f}s")

    diag = residual_diagnostics(result.fitted_train.resid())
    print("\nResidual diagnostics (initial train fit):")
    for k, v in diag.items():
        print(f"  {k:>20s}: {v:.4f}")

    print("\nRunning rolling-origin forecast on val (weekly refit)...")
    t0 = time.time()
    val_pred = rolling_forecast(target, val_idx, result.order, step_days=7)
    print(f"  Val forecast took {time.time() - t0:.1f}s ({len(val_pred)} rows)")

    val_pred = val_pred.set_index("date")
    val_pred["actual"] = target.loc[val_pred.index]
    val_pred["block"] = "val"
    val_metrics = score(val_pred["actual"], val_pred["predicted"])

    print("\nVal metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:>4s}: {v:.4f}")

    out_pred = ROOT / "artefacts" / "predictions" / "arima.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "arima_metrics.csv"
    out_diag = ROOT / "artefacts" / "metrics" / "arima_diagnostics.csv"
    out_order = ROOT / "artefacts" / "models" / "arima_order.txt"
    for p in (out_pred, out_metrics, out_diag, out_order):
        p.parent.mkdir(parents=True, exist_ok=True)

    test_metrics = None
    if do_test:
        print("\nRunning rolling-origin forecast on test (weekly refit)...")
        t0 = time.time()
        test_pred = rolling_forecast(target, test_idx, result.order, step_days=7)
        print(f"  Test forecast took {time.time() - t0:.1f}s ({len(test_pred)} rows)")
        test_pred = test_pred.set_index("date")
        test_pred["actual"] = target.loc[test_pred.index]
        test_pred["block"] = "test"
        test_metrics = score(test_pred["actual"], test_pred["predicted"])
        print("\nTest metrics:")
        for k, v in test_metrics.items():
            print(f"  {k:>4s}: {v:.4f}")
        all_pred = pd.concat([val_pred.reset_index(), test_pred.reset_index()])
    else:
        all_pred = val_pred.reset_index()

    metric_rows = [{"block": "val", **val_metrics}]
    if test_metrics is not None:
        metric_rows.append({"block": "test", **test_metrics})

    all_pred.to_csv(out_pred, index=False)
    pd.DataFrame(metric_rows).to_csv(out_metrics, index=False)
    pd.DataFrame([diag]).to_csv(out_diag, index=False)
    out_order.write_text(f"order={result.order}\naic={result.aic:.4f}\n")

    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")
    print(f"Wrote: {out_diag.relative_to(ROOT)}")
    print(f"Wrote: {out_order.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Also run rolling forecast on the test block (single OOD pass)")
    args = parser.parse_args()
    main(do_test=args.test)
