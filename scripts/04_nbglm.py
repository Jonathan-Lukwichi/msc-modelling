"""Plan §8 Step 3b: Negative Binomial GLM baseline (§5.7 headline parametric).

Same §5.2.5 raw 10 exogenous block + y_{t-7} autoregressive control. Paired
with SARIMAX as the two parallel parametric baselines per Ch5 §5.7.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.metrics import score
from src.forecasting.models.negbin import (
    fit_with_lag7, rolling_forecast, extract_coefficients,
)


def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    train_idx = splits.slice(g1, "train").index
    val_idx = splits.slice(g1, "val").index

    print(f"Train: {len(train_idx)} days  |  Val: {len(val_idx)} days")
    X_train, scaler = build_task1_exogenous(g1.loc[train_idx], fit_scaler=True)
    X_full, _ = build_task1_exogenous(g1, scaler=scaler)

    print("Fitting NB GLM on the train fold (with Normal sensitivity)...")
    t0 = time.time()
    fit = fit_with_lag7(target.loc[train_idx], X_train,
                        fit_normal_sensitivity=True)
    print(f"  Alpha (dispersion): {fit.alpha:.4f}")
    print(f"  AIC NB:     {fit.aic_nb:.2f}")
    print(f"  AIC Normal: {fit.aic_normal:.2f}")
    print(f"  AIC gap (Normal - NB) / NB = "
          f"{100 * (fit.aic_normal - fit.aic_nb) / fit.aic_nb:+.2f}%")
    print(f"  Fit took {time.time() - t0:.1f}s")

    feature_names = list(X_train.columns) + ["y_lag7"]
    coef_df = extract_coefficients(fit.fitted_train, feature_names)

    print("\nRunning rolling-origin NB GLM forecast on val...")
    t0 = time.time()
    val_pred = rolling_forecast(target, X_full, val_idx, step_days=7)
    print(f"  Val forecast took {time.time() - t0:.1f}s ({len(val_pred)} rows)")

    val_pred = val_pred.set_index("date")
    val_pred["actual"] = target.loc[val_pred.index]
    val_pred["block"] = "val"
    val_metrics = score(val_pred["actual"], val_pred["predicted"])

    print("\nVal metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:>4s}: {v:.4f}")

    # Persist
    out_pred = ROOT / "artefacts" / "predictions" / "nbglm.csv"
    out_metrics = ROOT / "artefacts" / "metrics" / "nbglm_metrics.csv"
    out_coef = ROOT / "artefacts" / "metrics" / "nbglm_coefficients.csv"
    out_disp = ROOT / "artefacts" / "metrics" / "nbglm_dispersion.csv"
    for p in (out_pred, out_metrics, out_coef, out_disp):
        p.parent.mkdir(parents=True, exist_ok=True)

    val_pred.reset_index().to_csv(out_pred, index=False)
    pd.DataFrame([{
        "block": "val",
        **val_metrics,
        "alpha": fit.alpha,
        "aic_nb": fit.aic_nb,
        "aic_normal": fit.aic_normal,
    }]).to_csv(out_metrics, index=False)
    coef_df.to_csv(out_coef, index=False)
    pd.DataFrame([{
        "alpha": fit.alpha,
        "aic_nb": fit.aic_nb,
        "aic_normal": fit.aic_normal,
    }]).to_csv(out_disp, index=False)

    print(f"\nWrote: {out_pred.relative_to(ROOT)}")
    print(f"Wrote: {out_metrics.relative_to(ROOT)}")
    print(f"Wrote: {out_coef.relative_to(ROOT)}")
    print(f"Wrote: {out_disp.relative_to(ROOT)}")

    print("\nTop 10 coefficients by |coef|:")
    coef_sorted = coef_df.copy()
    coef_sorted["abs"] = coef_sorted["coef"].abs()
    coef_sorted = coef_sorted.sort_values("abs", ascending=False).head(10).drop(columns="abs")
    print(coef_sorted.to_string(index=False))


if __name__ == "__main__":
    main()
