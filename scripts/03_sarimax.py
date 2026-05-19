"""Plan §8 Step 3a: SARIMAX baseline (Gaussian likelihood, §5.7 time-series pair).

§5.2.5 raw 10 exogenous block. Order picked by stepwise AIC ONCE on the train
fold, then re-estimated at each weekly origin on val (rolling refit).

Order is cached to artefacts/models/sarimax_order.txt so reruns skip the
~6-minute order search.
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
from src.forecasting.features import build_task1_exogenous
from src.forecasting.metrics import score
from src.forecasting.models.sarimax import (
    pick_order, fit_with_order, rolling_forecast, extract_coefficients,
)

ORDER_CACHE = ROOT / "artefacts" / "models" / "sarimax_order.txt"


def load_cached_order():
    if not ORDER_CACHE.exists():
        return None
    text = ORDER_CACHE.read_text()
    cfg = {}
    for line in text.strip().splitlines():
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    if "order" not in cfg or "seasonal_order" not in cfg:
        return None
    return eval(cfg["order"]), eval(cfg["seasonal_order"]), float(cfg.get("aic", "nan"))


def save_cached_order(order, seasonal_order, aic):
    ORDER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ORDER_CACHE.write_text(
        f"order={order}\nseasonal_order={seasonal_order}\naic={aic:.4f}\n"
    )


def main(force_repick: bool = False) -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    train_idx = splits.slice(g1, "train").index
    val_idx = splits.slice(g1, "val").index

    print(f"Train: {len(train_idx)} days  |  Val: {len(val_idx)} days")
    X_train, scaler = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    X_full, _ = build_task1_exogenous(g1, scaler=scaler)

    cached = load_cached_order() if not force_repick else None
    if cached is not None:
        order, seasonal_order, aic = cached
        print(f"Using cached order: {order} x {seasonal_order}, AIC = {aic:.2f}")
        fit = fit_with_order(target.loc[train_idx], X_train, order, seasonal_order)
    else:
        print("Picking SARIMAX order via stepwise AIC...")
        t0 = time.time()
        fit = pick_order(
            target.loc[train_idx], X_train,
            max_p=2, max_q=2, max_P=2, max_Q=2, m=7, seed=42,
        )
        print(f"  Picked order: {fit.order} x {fit.seasonal_order}, AIC = {fit.aic:.2f}")
        print(f"  Order selection took {time.time() - t0:.1f}s")
        save_cached_order(fit.order, fit.seasonal_order, fit.aic)

    print("\nRunning rolling-origin SARIMAX forecast on val...")
    t0 = time.time()
    val_pred = rolling_forecast(
        target, X_full, val_idx,
        order=fit.order, seasonal_order=fit.seasonal_order,
        step_days=7,
    )
    print(f"  Val forecast took {time.time() - t0:.1f}s ({len(val_pred)} rows)")

    val_pred = val_pred.set_index("date")
    val_pred["actual"] = target.loc[val_pred.index]
    val_pred["block"] = "val"
    val_metrics = score(val_pred["actual"], val_pred["predicted"])

    print("\nVal metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:>4s}: {v:.4f}")

    # Persist predictions and metrics FIRST so any downstream failure
    # (e.g. coefficient extraction) can't wipe out the 12-minute rolling refit.
    out_pred = ROOT / "artefacts" / "predictions" / "sarimax.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "sarimax_metrics.csv"
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    val_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{"block": "val", **val_metrics}]).to_csv(out_metrics, index=False)
    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")

    # Coefficient extraction (best-effort; failure does not lose predictions)
    try:
        coef_df = extract_coefficients(fit.fitted_train, list(X_train.columns))
        out_coef = ROOT / "artefacts" / "metrics" / "sarimax_coefficients.csv"
        coef_df.to_csv(out_coef, index=False)
        print(f"Wrote: {out_coef.relative_to(ROOT)}")
        print("\nCoefficients (sorted by |coef|):")
        coef_sorted = coef_df.copy()
        coef_sorted["abs"] = coef_sorted["coef"].abs()
        coef_sorted = coef_sorted.sort_values("abs", ascending=False).drop(columns="abs")
        print(coef_sorted.to_string(index=False))
    except Exception as exc:
        print(f"\nCoefficient extraction failed (non-fatal): {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-repick", action="store_true",
                        help="Re-run auto_arima order search even if cache exists")
    args = parser.parse_args()
    main(force_repick=args.force_repick)
