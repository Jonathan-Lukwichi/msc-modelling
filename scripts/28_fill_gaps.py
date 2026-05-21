"""Fill the cross-validation gaps surfaced by scripts/27.

Three deliverables:
  1. Compute MASE per (model, block) from saved predictions and the
     train target (seasonal-naive denominator with s=7), then update the
     canonical parquet leaderboard so the test_mase column is populated.
  2. Compute per-horizon test MAPE for XGBoost RMSE-best from the saved
     test predictions (which the upstream script did not emit).
  3. Compute SARIMAX test MAPE from the saved test predictions (the
     parquet currently has only the val row for SARIMAX).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRED = ROOT / "artefacts" / "predictions"
PRED_TEST = PRED / "test"


def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("actual") or cols.get("y_true") or cols.get("y")
    p = cols.get("predicted") or cols.get("yhat") or cols.get("y_pred")
    if a is None or p is None:
        raise ValueError(f"Cannot find actual/predicted columns in {path}")
    return df[[a, p]].rename(columns={a: "actual", p: "predicted"})


def _mase(y_true, y_pred, y_train, seasonality: int = 7) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    if len(y_tr) <= seasonality:
        return float("nan")
    denom = float(np.mean(np.abs(y_tr[seasonality:] - y_tr[:-seasonality])))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def _load_train_y():
    from src.forecasting.io import load_g1, Splits
    g1 = load_g1()
    splits = Splits.from_config()
    return g1.loc[splits.slice(g1, "train").index]["total_daily_arrivals"]


# =====================================================================
# 1. MASE for every model with saved val + test predictions
# =====================================================================
print("=" * 70)
print("1. Computing MASE for every model with saved predictions")
print("=" * 70)

y_train = _load_train_y()
print(f"Train target loaded: {len(y_train)} days, mean={y_train.mean():.2f}")

MODEL_FILES = {
    # canonical leaderboard key -> (val_csv name, test_csv name)
    "xgboost":             ("xgboost_rmse.csv",            "xgboost_rmse.csv"),
    "ann":                 ("ann_rmse.csv",                "ann_rmse.csv"),
    "lstm":                ("lstm_rmse.csv",               "lstm_rmse.csv"),
    "arima":               ("arima.csv",                   "arima.csv"),
    "sarimax":             ("sarimax.csv",                 "sarimax.csv"),
    "nbglm":               ("nbglm.csv",                   "nbglm.csv"),
    "hybrid_sarimax_lstm": ("hybrid_sarimax_lstm_rmse.csv", "hybrid_sarimax_lstm_rmse.csv"),
    "hybrid_sarimax_xgb":  ("hybrid_sarimax_xgb_rmse.csv",  "hybrid_sarimax_xgb_rmse.csv"),
    "hybrid_lstm_xgb":     ("hybrid_lstm_xgb_rmse.csv",    "hybrid_lstm_xgb_rmse.csv"),
}

mase_rows = []
for model, (val_name, test_name) in MODEL_FILES.items():
    for block, fname, src_dir in [("val", val_name, PRED), ("test", test_name, PRED_TEST)]:
        path = src_dir / fname
        if not path.exists():
            print(f"  [MISS] {model} {block} -> {path.name}")
            continue
        df = _read_preds(path)
        mase = _mase(df["actual"], df["predicted"], y_train)
        mae = float(np.mean(np.abs(df["actual"] - df["predicted"])))
        rmse = float(np.sqrt(np.mean((df["actual"] - df["predicted"]) ** 2)))
        mape = float(
            np.mean(np.abs((df["actual"] - df["predicted"]) / df["actual"].clip(lower=0.6)))
            * 100
        )
        mase_rows.append({
            "model": model, "block": block, "n": len(df),
            "MAPE": mape, "MAE": mae, "RMSE": rmse, "MASE": mase,
        })
        print(f"  {model:25s} {block:4s}  MAPE={mape:6.2f}  MAE={mae:5.2f}  "
              f"RMSE={rmse:5.2f}  MASE={mase:.3f}")

mase_df = pd.DataFrame(mase_rows)
mase_path = ROOT / "artefacts" / "metrics" / "mase_per_model.csv"
mase_df.to_csv(mase_path, index=False)
print(f"\nWrote: {mase_path.relative_to(ROOT)}")

# =====================================================================
# 2. Update parquet leaderboard with MASE columns
# =====================================================================
print("\n" + "=" * 70)
print("2. Updating canonical leaderboard parquet with MASE column")
print("=" * 70)

lb_path = ROOT / "artefacts" / "leaderboard_canonical.parquet"
lb = pq.read_table(lb_path).to_pandas()
print(f"Loaded {len(lb)} rows.")

# Build lookup: (model, block) -> MASE
mase_lookup = {(r["model"], r["block"]): r["MASE"] for _, r in mase_df.iterrows()}

# Update val_mase and test_mase columns for each leaderboard row
n_updated = 0
for i, row in lb.iterrows():
    m = row["model"]
    if pd.isna(row["val_mase"]):
        v = mase_lookup.get((m, "val"))
        if v is not None and not pd.isna(v):
            lb.at[i, "val_mase"] = v
            n_updated += 1
    if pd.isna(row["test_mase"]):
        v = mase_lookup.get((m, "test"))
        if v is not None and not pd.isna(v):
            lb.at[i, "test_mase"] = v
            n_updated += 1

print(f"Updated {n_updated} MASE cells.")

# Also fill missing test_mape / test_rmse for SARIMAX et al where val was the
# only row in the parquet. Read from mase_df rows for the corresponding block.
test_lookup = {
    (r["model"], r["block"]): (r["MAPE"], r["RMSE"])
    for _, r in mase_df.iterrows()
}
n_test_filled = 0
for i, row in lb.iterrows():
    m = row["model"]
    if pd.isna(row["test_mape"]):
        v = test_lookup.get((m, "test"))
        if v is not None:
            lb.at[i, "test_mape"] = v[0]
            lb.at[i, "test_rmse"] = v[1]
            n_test_filled += 1
print(f"Filled {n_test_filled} missing test_mape rows.")

from src.forecasting.leaderboard import CANONICAL_SCHEMA
# Coerce dtypes
for col, t in zip(CANONICAL_SCHEMA.names, CANONICAL_SCHEMA.types):
    if pa.types.is_floating(t):
        lb[col] = pd.to_numeric(lb[col], errors="coerce")
    elif pa.types.is_string(t):
        lb[col] = lb[col].astype("string").fillna("")
    elif pa.types.is_timestamp(t):
        lb[col] = pd.to_datetime(lb[col], errors="coerce")
    elif pa.types.is_integer(t):
        lb[col] = pd.to_numeric(lb[col], errors="coerce").fillna(0).astype("int64")

tbl = pa.Table.from_pandas(lb, schema=CANONICAL_SCHEMA, preserve_index=False)
pq.write_table(tbl, lb_path)
print(f"Wrote: {lb_path.relative_to(ROOT)}")
print("\nUpdated leaderboard:")
print(lb.sort_values("test_mape")[["model", "family", "criterion",
                                     "val_mape", "test_mape", "test_mase"]].head(10).to_string(index=False))

# =====================================================================
# 3. Per-horizon for XGBoost RMSE-tuned
# =====================================================================
print("\n" + "=" * 70)
print("3. Per-horizon test MAPE for XGBoost RMSE-best")
print("=" * 70)

xgb_test = _read_preds(PRED_TEST / "xgboost_rmse.csv").reset_index()
# Horizon = (date - test_start).days mod 7 + 1
test_start = xgb_test["date"].min()
xgb_test["horizon"] = ((xgb_test["date"] - test_start).dt.days % 7) + 1

per_horizon_rows = []
for h in range(1, 8):
    sub = xgb_test[xgb_test["horizon"] == h]
    if sub.empty:
        continue
    mape = float(
        np.mean(np.abs((sub["actual"] - sub["predicted"]) / sub["actual"].clip(lower=0.6)))
        * 100
    )
    mae = float(np.mean(np.abs(sub["actual"] - sub["predicted"])))
    rmse = float(np.sqrt(np.mean((sub["actual"] - sub["predicted"]) ** 2)))
    per_horizon_rows.append({
        "model": "xgboost", "horizon": h, "n": len(sub),
        "MAPE": mape, "MAE": mae, "RMSE": rmse,
    })
    print(f"  h={h}: n={len(sub):3d}  MAPE={mape:6.2f}  MAE={mae:5.2f}  RMSE={rmse:5.2f}")

# Append to existing per_horizon table if present
ph_path = ROOT / "artefacts" / "metrics" / "test_per_horizon.csv"
if ph_path.exists():
    existing = pd.read_csv(ph_path)
    # Drop any prior xgboost rows so we don't double up
    existing = existing[existing["model"] != "xgboost"]
    combined = pd.concat([existing, pd.DataFrame(per_horizon_rows)], ignore_index=True)
else:
    combined = pd.DataFrame(per_horizon_rows)
combined.to_csv(ph_path, index=False)
print(f"Wrote: {ph_path.relative_to(ROOT)}")

print("\n" + "=" * 70)
print("Done. Run scripts/27 again to verify all PASS.")
print("=" * 70)
