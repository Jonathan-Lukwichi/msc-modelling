"""Build the Task 1 portion of the handover_to_webapp/ package.

Copies the 6 deployable Task 1 pickles from artefacts/models/deploy/ with
alias filenames, generates per-model card.json files, and consolidates
the headline.json + per_horizon.json metrics that the web app dashboard
needs.

Run once; idempotent.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artefacts" / "models" / "deploy"
DST = ROOT / "artefacts" / "handover_to_webapp" / "task1_daily_arrivals"

ALIAS_MAP = [
    {"alias": "Stat 1",   "family": "Statistical", "src": "arima.pkl",
     "pkl": "stat1.pkl",   "sci": "ARIMA(p,1,q)",
     "internal_key": "arima",      "val_MAPE": 13.33, "val_RMSE": 10.20},
    {"alias": "Stat 2",   "family": "Statistical", "src": "sarimax.pkl",
     "pkl": "stat2.pkl",   "sci": "SARIMAX(p,1,q)(P,1,Q)_7",
     "internal_key": "sarimax",    "val_MAPE": 12.34, "val_RMSE": 8.98},
    {"alias": "ML 1",     "family": "ML",          "src": "xgboost.pkl",
     "pkl": "ml1.pkl",     "sci": "XGBoost",
     "internal_key": "xgboost",    "val_MAPE": 11.96, "val_RMSE": 9.39},
    {"alias": "ML 2",     "family": "ML",          "src": "ann.pkl",
     "pkl": "ml2.pkl",     "sci": "ANN (MLP)",
     "internal_key": "ann",        "val_MAPE": 12.32, "val_RMSE": 9.24},
    {"alias": "Hybrid 1", "family": "Hybrid",      "src": "hybrid_sarimax_xgb.pkl",
     "pkl": "hybrid1.pkl", "sci": "SARIMAX + XGBoost residual refiner",
     "internal_key": "sarimax_xgb","val_MAPE": 12.04, "val_RMSE": 8.88},
    {"alias": "Hybrid 2", "family": "Hybrid",      "src": "hybrid_sarimax_lstm.pkl",
     "pkl": "hybrid2.pkl", "sci": "SARIMAX + LSTM residual refiner",
     "internal_key": "sarimax_lstm","val_MAPE": 12.19, "val_RMSE": 9.05},
]


def badge(mape: float) -> tuple[str, str, str]:
    if mape is None:
        return ("unknown", "Unknown", "⚪")
    if mape < 15:
        return ("operational", "Operational", "🟢")
    if mape < 30:
        return ("planning", "Planning", "🟡")
    return ("research", "Research preview", "🔴")


def get_aggregates(internal_key: str, p1: pd.DataFrame) -> dict:
    row = p1[p1["model"] == internal_key]
    if row.empty:
        return {"weekly_avg_pct_error": None,
                "monthly_avg_pct_error": None,
                "yearly_avg_pct_error": None}
    r = row.iloc[0]
    return {
        "weekly_avg_pct_error":  float(r["weekly_avg_pct_error"]),
        "monthly_avg_pct_error": float(r["monthly_avg_pct_error"]),
        "yearly_avg_pct_error":  float(r["yearly_avg_pct_error"]),
    }


def main() -> None:
    (DST / "models").mkdir(parents=True, exist_ok=True)
    (DST / "metrics").mkdir(parents=True, exist_ok=True)
    (DST / "cards").mkdir(parents=True, exist_ok=True)

    p1 = pd.read_csv(ROOT / "artefacts" / "phase1_defaults" / "summary_phase1.csv")
    headline = []
    per_horizon = []

    for m in ALIAS_MAP:
        src_path = SRC / m["src"]
        dst_path = DST / "models" / m["pkl"]
        shutil.copy2(src_path, dst_path)
        size_mb = round(dst_path.stat().st_size / (1024 * 1024), 2)

        aggs = get_aggregates(m["internal_key"], p1)
        b_id, b_label, b_emoji = badge(m["val_MAPE"])

        card = {
            "alias": m["alias"],
            "family": m["family"],
            "task": "task1",
            "task_label": "Daily Total ED Arrivals",
            "resolution": "daily",
            "pickle_filename": m["pkl"],
            "pickle_size_mb": size_mb,
            "performance": {
                "val_MAPE": m["val_MAPE"],
                "val_RMSE": m["val_RMSE"],
                "weekly_avg_pct_error":  aggs["weekly_avg_pct_error"],
                "monthly_avg_pct_error": aggs["monthly_avg_pct_error"],
                "yearly_avg_pct_error":  aggs["yearly_avg_pct_error"],
            },
            "badge": b_id,
            "badge_label": b_label,
            "badge_emoji": b_emoji,
            "training_window": "train + val (2022-03-01 to 2024-12-31, 1037 days)",
            "last_trained_utc": datetime.now(timezone.utc).isoformat(),
            "description": (
                f"Alias {m['alias']} ({m['family']}). Steve Biko Hospital ED "
                f"daily total arrivals forecaster. "
                f"Operational badge tier: {b_label}."
            ),
            "internal_only": {
                "scientific_name": m["sci"],
                "source_pickle": str(src_path.relative_to(ROOT)),
            },
        }
        card_path = DST / "cards" / f"{m['pkl'].replace('.pkl','')}.json"
        card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False),
                              encoding="utf-8")

        headline.append({
            "alias": m["alias"], "task": "task1", "specialty": None,
            "resolution": "daily",
            "val_MAPE": m["val_MAPE"], "val_RMSE": m["val_RMSE"],
            "val_MAE": None, "val_R2": None,
            "weekly_avg_pct_error":  aggs["weekly_avg_pct_error"],
            "monthly_avg_pct_error": aggs["monthly_avg_pct_error"],
            "yearly_avg_pct_error":  aggs["yearly_avg_pct_error"],
            "badge": b_id, "badge_label": b_label, "badge_emoji": b_emoji,
        })

        per_horizon.append({
            "alias": m["alias"],
            "horizons": {
                "daily":   {"metric": "MAPE",
                            "value": m["val_MAPE"],
                            "badge": badge(m["val_MAPE"])[0]},
                "weekly":  {"metric": "pct_error_avg",
                            "value": aggs["weekly_avg_pct_error"],
                            "badge": badge(aggs["weekly_avg_pct_error"])[0]
                            if aggs["weekly_avg_pct_error"] is not None else None},
                "monthly": {"metric": "pct_error_avg",
                            "value": aggs["monthly_avg_pct_error"],
                            "badge": badge(aggs["monthly_avg_pct_error"])[0]
                            if aggs["monthly_avg_pct_error"] is not None else None},
                "yearly":  {"metric": "pct_error_avg",
                            "value": aggs["yearly_avg_pct_error"],
                            "badge": badge(aggs["yearly_avg_pct_error"])[0]
                            if aggs["yearly_avg_pct_error"] is not None else None},
            }
        })

        print(f"  [{m['alias']:<9s}] -> {m['pkl']:14s} {size_mb} MB   "
              f"MAPE {m['val_MAPE']:.2f}% {b_emoji}")

    (DST / "metrics" / "headline.json").write_text(
        json.dumps(headline, indent=2, ensure_ascii=False), encoding="utf-8")
    (DST / "metrics" / "per_horizon.json").write_text(
        json.dumps(per_horizon, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {DST/'cards'}/{6} cards")
    print(f"Wrote {DST/'metrics'/'headline.json'}")
    print(f"Wrote {DST/'metrics'/'per_horizon.json'}")


if __name__ == "__main__":
    main()
