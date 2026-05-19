"""Split-sensitivity check for ARIMA.

Runs two additional chronological splits and compares against the existing
§5.5.2 val result + the 80/20-on-full-G1 result already saved:

  Variant A: 70/30 chronological on full G1 (zero-day-filtered)
  Variant B: 80/20 chronological with COVID period excluded
             (train starts 2022-03-01 — matches §5.5.2 exclusion philosophy)

Standalone — re-uses src.forecasting.models.arima only.
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
from src.forecasting.models.arima import pick_order, rolling_forecast


def run_variant(label: str, series: pd.Series, train_frac: float) -> dict:
    """Fit ARIMA on the first train_frac of `series`, rolling-forecast the rest."""
    n = len(series)
    cut = int(round(n * train_frac))
    train = series.iloc[:cut]
    test_idx = series.iloc[cut:].index

    print(f"\n{'='*70}")
    print(f"Variant: {label}")
    print(f"{'='*70}")
    print(f"  total days : {n}")
    print(f"  train days : {len(train):>4d}  "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"  test  days : {len(test_idx):>4d}  "
          f"({test_idx[0].date()} -> {test_idx[-1].date()})")

    print("\n  Picking ARIMA order via stepwise AIC...")
    t0 = time.time()
    result = pick_order(train, max_p=3, max_q=3, d=1, seed=42)
    print(f"    Picked order: {result.order}, AIC = {result.aic:.2f}  "
          f"({time.time() - t0:.1f}s)")

    print("  Running rolling-origin forecast (weekly refit)...")
    t0 = time.time()
    pred = rolling_forecast(series, test_idx, result.order, step_days=7)
    print(f"    Forecast took {time.time() - t0:.1f}s ({len(pred)} rows)")

    pred = pred.set_index("date")
    pred["actual"] = series.loc[pred.index]
    pred["block"] = label
    metrics = score(pred["actual"], pred["predicted"])

    print(f"\n  Metrics for {label}:")
    for k, v in metrics.items():
        print(f"    {k:>4s}: {v:.4f}")

    return {
        "label": label,
        "order": result.order,
        "aic": result.aic,
        "train_days": len(train),
        "test_days": len(test_idx),
        "train_start": train.index[0].date().isoformat(),
        "train_end": train.index[-1].date().isoformat(),
        "test_start": test_idx[0].date().isoformat(),
        "test_end": test_idx[-1].date().isoformat(),
        "MAPE": metrics["MAPE"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "_pred": pred,
    }


def main() -> None:
    g1 = load_g1()
    target_full = g1["total_daily_arrivals"]

    # Variant A: 70/30 on full G1
    res_a = run_variant("70_30_full", target_full, train_frac=0.70)

    # Variant B: 80/20 on post-COVID-only window
    post_covid_start = pd.Timestamp("2022-03-01")
    target_post = target_full.loc[target_full.index >= post_covid_start]
    res_b = run_variant("80_20_post_covid", target_post, train_frac=0.80)

    # Save predictions + summary metrics
    out_dir_pred = ROOT / "artefacts" / "predictions"
    out_dir_metr = ROOT / "artefacts" / "metrics"
    out_dir_pred.mkdir(parents=True, exist_ok=True)
    out_dir_metr.mkdir(parents=True, exist_ok=True)

    for res in (res_a, res_b):
        out_pred = out_dir_pred / f"arima_{res['label']}.csv"
        res["_pred"].reset_index().to_csv(out_pred, index=False)
        print(f"\n  Wrote: {out_pred.relative_to(ROOT)}")

    summary_rows = [
        {k: v for k, v in r.items() if k != "_pred"}
        for r in (res_a, res_b)
    ]
    summary_df = pd.DataFrame(summary_rows)
    out_summary = out_dir_metr / "arima_split_sensitivity.csv"
    summary_df.to_csv(out_summary, index=False)
    print(f"  Wrote: {out_summary.relative_to(ROOT)}")

    print(f"\n{'='*70}")
    print("Summary (all ARIMA split variants):")
    print(f"{'='*70}")
    print(summary_df[["label", "order", "train_days", "test_days",
                      "MAPE", "MAE", "RMSE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
