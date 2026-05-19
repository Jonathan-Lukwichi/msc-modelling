"""Diagnostics: (A) per-horizon MAPE decomposition + (D) noise-floor estimate.

A - For weekly-rolling-refit models, every 7-day block has horizons 1..7.
    Day-1 MAPE is usually much lower than day-7 MAPE. Reporting only the
    average MAPE hides this.

D - The irreducible error floor: even a "perfect conditional mean" model
    still leaves the within-group variance as residual error. We estimate
    that floor from train-fold (dow x is_public_holiday x is_weekend) group
    means and compare against the actual ARIMA / ensemble MAPE on val.

Standalone diagnostic — writes:
  artefacts/metrics/diag_per_horizon_mape.csv
  artefacts/metrics/diag_noise_floor.csv
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.metrics import score, mape


PRED_FILES = [
    ("arima",                 "arima.csv"),
    ("sarimax",               "sarimax.csv"),
    ("nbglm",                 "nbglm.csv"),
    ("ann",                   "ann.csv"),
    ("lstm",                  "lstm.csv"),
    ("hybrid_sarimax_xgb",    "hybrid_sarimax_xgb.csv"),
    ("hybrid_sarimax_lstm",   "hybrid_sarimax_lstm.csv"),
    ("ensemble_E1",           "ensemble_E1_simple_top3.csv"),
    ("ensemble_E2",           "ensemble_E2_inv_mape_weighted.csv"),
    ("ensemble_E3",           "ensemble_E3_optimal_convex.csv"),
]


def load_pred(path: Path, target: pd.Series) -> pd.DataFrame | None:
    """Load a prediction CSV. Ensure `actual` is attached from G1."""
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "actual" not in df.columns:
        df["actual"] = target.reindex(df.index)
    df = df.dropna(subset=["actual", "predicted"])
    return df


# ---------------------------------------------------------------------------
# Diagnostic A: per-horizon MAPE
# ---------------------------------------------------------------------------

def per_horizon_mape(pred_df: pd.DataFrame, block_start: pd.Timestamp,
                     step_days: int = 7) -> pd.DataFrame:
    """Split a rolling-forecast prediction frame into horizons 1..step_days."""
    df = pred_df.copy()
    df["days_from_start"] = (df.index - block_start).days
    df["horizon"] = (df["days_from_start"] % step_days) + 1
    rows = []
    for h in range(1, step_days + 1):
        sub = df[df["horizon"] == h]
        if len(sub) == 0:
            continue
        m = score(sub["actual"], sub["predicted"])
        rows.append({"horizon": h, "n_days": len(sub), **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic D: noise-floor estimate
# ---------------------------------------------------------------------------

def noise_floor(train_df: pd.DataFrame, val_df: pd.DataFrame,
                target_col: str = "total_daily_arrivals") -> dict:
    """Fit train-fold group means by (dow, is_public_holiday, is_weekend).

    Compute:
      - train residual SD after subtracting the group mean (== noise-floor SD)
      - implied floor MAPE = residual SD / overall mean
      - the actual MAPE of this naive group-mean predictor on val
    """
    train = train_df.copy()
    train["dow"] = train.index.dayofweek

    keys = ["dow", "is_public_holiday", "is_weekend"]
    group_means = train.groupby(keys)[target_col].mean()
    train["pred_group_mean"] = train.set_index(keys)[target_col].index.map(
        lambda k: group_means.loc[k]
    ).values
    # Defensive: fill any unseen group
    overall_mean = train[target_col].mean()

    residuals = train[target_col] - train["pred_group_mean"]
    residual_sd = float(residuals.std(ddof=1))
    residual_mae = float(residuals.abs().mean())
    floor_mape_implied = residual_sd / overall_mean * 100.0
    floor_mape_mae = residual_mae / overall_mean * 100.0

    # Apply group means to val
    val = val_df.copy()
    val["dow"] = val.index.dayofweek
    val_pred = []
    for _, row in val.iterrows():
        k = (int(row["dow"]), int(row["is_public_holiday"]), int(row["is_weekend"]))
        if k in group_means.index:
            val_pred.append(group_means.loc[k])
        else:
            val_pred.append(overall_mean)
    val["pred_group_mean"] = val_pred

    naive_metrics = score(val[target_col], val["pred_group_mean"])

    return {
        "train_overall_mean": float(overall_mean),
        "train_residual_sd_after_groups": residual_sd,
        "train_residual_mae_after_groups": residual_mae,
        "implied_floor_mape_from_sd": float(floor_mape_implied),
        "implied_floor_mape_from_mae": float(floor_mape_mae),
        "naive_group_mean_val_MAPE": naive_metrics["MAPE"],
        "naive_group_mean_val_MAE": naive_metrics["MAE"],
        "naive_group_mean_val_RMSE": naive_metrics["RMSE"],
        "naive_group_mean_val_R2": naive_metrics["R2"],
        "n_groups": int(group_means.shape[0]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    g1 = load_g1()
    splits = Splits.from_config()
    target = g1["total_daily_arrivals"]

    train_df = splits.slice(g1, "train")
    val_df = splits.slice(g1, "val")
    val_start = val_df.index[0]
    print(f"Val window: {val_df.index[0].date()} -> {val_df.index[-1].date()}  "
          f"({len(val_df)} days)")

    # ---------- A: per-horizon MAPE for every available model ----------
    print("\n" + "=" * 70)
    print("Diagnostic A: per-horizon MAPE (val, weekly rolling refit)")
    print("=" * 70)
    pred_dir = ROOT / "artefacts" / "predictions"

    all_horizon = []
    headline_summary = []
    for tag, fname in PRED_FILES:
        df = load_pred(pred_dir / fname, target)
        if df is None:
            print(f"  [skip] {fname} not found")
            continue
        # Restrict to val window (some files include test too)
        df = df.loc[(df.index >= val_df.index[0]) & (df.index <= val_df.index[-1])]
        if len(df) == 0:
            print(f"  [skip] {fname} has no val rows")
            continue

        full = score(df["actual"], df["predicted"])
        hz = per_horizon_mape(df, val_start, step_days=7)
        hz.insert(0, "model", tag)
        all_horizon.append(hz)

        h1 = float(hz.loc[hz["horizon"] == 1, "MAPE"].iloc[0])
        h7 = float(hz.loc[hz["horizon"] == 7, "MAPE"].iloc[0])
        headline_summary.append({
            "model": tag,
            "overall_MAPE": full["MAPE"],
            "h1_MAPE": h1,
            "h7_MAPE": h7,
            "h7_minus_h1": h7 - h1,
        })

    horizon_df = pd.concat(all_horizon, ignore_index=True)
    headline_df = pd.DataFrame(headline_summary).sort_values("overall_MAPE")

    out_hz = ROOT / "artefacts" / "metrics" / "diag_per_horizon_mape.csv"
    out_hz.parent.mkdir(parents=True, exist_ok=True)
    horizon_df.to_csv(out_hz, index=False)
    print(f"\nWrote: {out_hz.relative_to(ROOT)}")

    print("\nHeadline: overall MAPE vs day-1 vs day-7")
    print(headline_df.to_string(index=False,
                                formatters={"overall_MAPE": "{:.2f}".format,
                                            "h1_MAPE": "{:.2f}".format,
                                            "h7_MAPE": "{:.2f}".format,
                                            "h7_minus_h1": "{:+.2f}".format}))

    # ---------- D: noise floor ----------
    print("\n" + "=" * 70)
    print("Diagnostic D: noise floor")
    print("=" * 70)
    floor = noise_floor(train_df, val_df)
    floor_df = pd.DataFrame([floor])
    out_floor = ROOT / "artefacts" / "metrics" / "diag_noise_floor.csv"
    floor_df.to_csv(out_floor, index=False)
    print(f"\nWrote: {out_floor.relative_to(ROOT)}")

    print("\nGroup-based noise floor (train fold, dow x is_public_holiday x is_weekend):")
    print(f"  Number of (dow x holiday x weekend) groups : {floor['n_groups']}")
    print(f"  Train overall mean (count/day)             : {floor['train_overall_mean']:.2f}")
    print(f"  Train residual SD after group means        : {floor['train_residual_sd_after_groups']:.2f}")
    print(f"  Train residual MAE after group means       : {floor['train_residual_mae_after_groups']:.2f}")
    print(f"  Implied floor MAPE (from residual SD)      : {floor['implied_floor_mape_from_sd']:.2f}%")
    print(f"  Implied floor MAPE (from residual MAE)     : {floor['implied_floor_mape_from_mae']:.2f}%")
    print(f"\nNaive group-mean predictor applied to val:")
    print(f"  MAPE : {floor['naive_group_mean_val_MAPE']:.2f}%")
    print(f"  MAE  : {floor['naive_group_mean_val_MAE']:.2f}")
    print(f"  RMSE : {floor['naive_group_mean_val_RMSE']:.2f}")
    print(f"  R^2  : {floor['naive_group_mean_val_R2']:.4f}")

    # ---------- Combined interpretation ----------
    if len(headline_df) > 0:
        best = headline_df.iloc[0]
        floor_mae = floor["implied_floor_mape_from_mae"]
        print("\n" + "=" * 70)
        print("Interpretation")
        print("=" * 70)
        print(f"  Best model on val          : {best['model']}  ({best['overall_MAPE']:.2f}%)")
        print(f"  Day-1 MAPE of best model   : {best['h1_MAPE']:.2f}%")
        print(f"  Day-7 MAPE of best model   : {best['h7_MAPE']:.2f}%")
        print(f"  Naive group-mean baseline  : {floor['naive_group_mean_val_MAPE']:.2f}%")
        print(f"  Implied noise floor (MAE)  : {floor_mae:.2f}%")
        gap = best['overall_MAPE'] - floor_mae
        print(f"  Best - floor               : {gap:+.2f}pp")


if __name__ == "__main__":
    main()
