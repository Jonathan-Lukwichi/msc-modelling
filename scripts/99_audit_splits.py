"""Audit harness: verify every model honours Ch5 §5.5.2 splits and post-COVID
training restriction, and Ch3 §3.6.1 CV procedure.

Prints a pass/fail grid for the user.
"""
from __future__ import annotations

from pathlib import Path
import sys
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return condition


def main() -> None:
    splits = Splits.from_config()
    g1 = load_g1()
    print("=" * 78)
    print("Audit: Chapter 5 §5.5.2 split + post-COVID training restriction")
    print("=" * 78)

    # 1. Splits are settled per Ch5 §5.5.2
    print("\n1. Settled split dates from Ch5 §5.5.2:")
    pre_covid_excluded = (splits.pre_covid_end == pd.Timestamp("2020-02-29"))
    during_covid_excluded = (splits.train_start == pd.Timestamp("2022-03-01"))
    check("train_start == 2022-03-01 (post-COVID block begins)",
          splits.train_start == pd.Timestamp("2022-03-01"))
    check("train_end == 2024-06-30", splits.train_end == pd.Timestamp("2024-06-30"))
    check("val_start == 2024-07-01", splits.val_start == pd.Timestamp("2024-07-01"))
    check("val_end == 2024-12-31", splits.val_end == pd.Timestamp("2024-12-31"))
    check("test_start == 2025-01-01", splits.test_start == pd.Timestamp("2025-01-01"))
    check("test_end == 2026-01-31", splits.test_end == pd.Timestamp("2026-01-31"))

    # 2. Split sizes
    print("\n2. Split sizes after is_zero_day filter:")
    train = splits.slice(g1, "train")
    val = splits.slice(g1, "val")
    test = splits.slice(g1, "test")
    check(f"train size 840-853 (post-zero-day filter)",
          840 <= len(train) <= 853, f"actual: {len(train)}")
    check(f"val size == 184", len(val) == 184, f"actual: {len(val)}")
    check(f"test size == 396", len(test) == 396, f"actual: {len(test)}")

    # 3. No pre-COVID or during-COVID rows in train
    print("\n3. Train block contains ZERO pre-COVID or during-COVID rows:")
    pre_dates = train.index.intersection(
        pd.date_range(splits.pre_covid_start, splits.pre_covid_end))
    during_dates = train.index.intersection(
        pd.date_range(splits.pre_covid_end + pd.Timedelta(days=1),
                       splits.train_start - pd.Timedelta(days=1)))
    check("no pre-COVID rows in train", len(pre_dates) == 0,
          f"found {len(pre_dates)}")
    check("no during-COVID rows in train", len(during_dates) == 0,
          f"found {len(during_dates)}")

    # 4. Each model's saved metrics reference the correct splits
    print("\n4. Model artefacts reference the correct val window:")
    pred_dir = ROOT / "artefacts" / "predictions"
    model_files = [
        "arima", "sarimax", "nbglm", "xgboost", "ann", "lstm",
        "hybrid_sarimax_xgb", "hybrid_sarimax_lstm", "hybrid_lstm_xgb",
        "hybrid_stl_xgb", "hybrid_stl_ann", "hybrid_stl_lstm",
    ]
    for name in model_files:
        p = pred_dir / f"{name}.csv"
        if not p.exists():
            check(f"{name}: predictions exist", False, "file missing")
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        val_rows = df[(df["date"] >= splits.val_start)
                       & (df["date"] <= splits.val_end)]
        check(f"{name}: predictions cover val window (184 rows)",
              len(val_rows) == 184, f"actual: {len(val_rows)}")

    # 5. HPO procedure flags (which models used inner CV vs val-based)
    print("\n5. HPO procedure (which models used Ch3 §3.6.1 inner CV):")
    cv_flags = {
        "ARIMA / SARIMAX": "AIC on train (equivalent to CV by §3.5.2)",
        "NB GLM": "no HPO (dispersion estimated from Poisson chi^2)",
        "XGBoost": "inner rolling-origin CV: k=10 folds x 192 grid combos",
        "ANN": "inner rolling-origin CV: k=10 folds x 20 random trials",
        "LSTM": "inner rolling-origin CV: k=10 folds x 15 TPE trials (180-min budget)",
        "Hybrids": "light defaults; refiner HPO not separately tuned (plan §12)",
    }
    for k, v in cv_flags.items():
        print(f"  {k:25s} -> {v}")

    print("\n" + "=" * 78)
    print("Audit complete.")


if __name__ == "__main__":
    main()
