"""Drift-aware refit comparison (Prompt 7 orchestrator).

For each base in {DoW mean, SARIMAX, XGBoost} runs three configurations:
  1. ``expanding``                         baseline (the current pipeline).
  2. ``sliding_450``                       sliding window of last 450 days.
  3. ``sliding_450 + RuLSIF``              sliding + Yamada (2013) relative
                                            density-ratio importance weights.

ANN and LSTM are omitted from this orchestrator because the full grid
(2 blocks x 3 configs x 2 deep models) would be 5+ hours on CPU; the
three lighter models are already enough to demonstrate the effect on
the +18.3 % drift documented in Ch5 §5.5.2.

Outputs:
  - artefacts/predictions/{val,test}/{model}_{config}.csv
  - artefacts/metrics/drift_aware_comparison.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.features import build_task1_exogenous
from src.forecasting.metrics import score
from src.forecasting.rolling import (
    RollingForecaster, FoldPrediction, make_sarimax_factory, make_xgboost_factory,
)
from src.forecasting.drift.sliding_cv import make_iw_sample_weight_fn


SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL = (0, 1, 1, 7)
XGB_BEST = {
    "n_estimators": 300, "max_depth": 5,
    "learning_rate": 0.01, "subsample": 1.0,
}
CONFIGS = ("expanding", "sliding_450", "sliding_450_rulsif")


def dow_mean_factory():
    """DoW-mean baseline; fits one mean per day-of-week from y_train."""
    def factory(X_train, y_train, sample_weight=None):
        df = pd.DataFrame({"y": y_train.values,
                              "dow": y_train.index.dayofweek})
        if sample_weight is not None:
            df["w"] = sample_weight
            means = df.groupby("dow").apply(
                lambda d: float(np.average(d["y"], weights=d["w"]))
            )
        else:
            means = df.groupby("dow")["y"].mean()

        class _Fitted:
            def predict(self, X_future, h):
                dates = X_future.index if X_future is not None else None
                if dates is None:
                    raise ValueError("dow_mean needs dates via X_future.index")
                preds = np.asarray(
                    [means.get(d.dayofweek, float(df["y"].mean())) for d in dates],
                    dtype=float,
                )
                return FoldPrediction(yhat=preds)

        return _Fitted()
    return factory


def run_config(
    model_name: str, base_factory, X_full, y_full,
    train_idx, val_idx, test_idx, config: str,
):
    """Run one (model, config) cell on val and test; return per-block rows."""
    window_days = None if config == "expanding" else 450
    swfn = (
        make_iw_sample_weight_fn(method="rulsif", recent_days=90)
        if config == "sliding_450_rulsif" else None
    )
    rf = RollingForecaster(
        model_factory=base_factory,
        step_days=7, horizon_days=7,
        min_train_days=365,
        window_days=window_days,
        sample_weight_fn=swfn,
    )
    rows = []
    for block_name, blk_idx in [("val", val_idx), ("test", test_idx)]:
        t0 = time.time()
        out = rf.fit_predict(X=X_full, y=y_full, eval_index=blk_idx)
        out["actual"] = y_full.loc[out.index].values
        s = score(out["actual"], out["yhat"])
        print(f"  [{model_name:11s} | {config:22s} | {block_name}] "
              f"({time.time() - t0:.0f}s)  MAPE={s['MAPE']:.3f}  "
              f"MAE={s['MAE']:.2f}  RMSE={s['RMSE']:.2f}")
        sub = ("test" if block_name == "test" else "")
        save_df = out.reset_index().rename(columns={"yhat": "predicted"})
        save_path = (ROOT / "artefacts" / "predictions" / sub /
                       f"{model_name}_{config}.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_df.to_csv(save_path, index=False)
        rows.append({
            "model": model_name, "config": config, "block": block_name,
            "MAPE": s["MAPE"], "MAE": s["MAE"],
            "RMSE": s["RMSE"], "R2": s["R2"], "n": len(out),
        })
    return rows


def main():
    print("=" * 70)
    print("PROMPT 7 — Drift-aware refit  (DoW / SARIMAX / XGBoost)")
    print("=" * 70 + "\n")

    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    # Exogenous block: §5.2.5 raw 10 for SARIMAX, consensus 23 for XGBoost
    X_train_full, scaler = build_task1_exogenous(
        splits.slice(g1, "train"), fit_scaler=True,
    )
    X_full_sarimax, _ = build_task1_exogenous(g1, scaler=scaler)
    eng = load_engineered()
    X_cons_full = build_selected_X(eng)

    train_idx = splits.slice(g1, "train").index
    val_idx = splits.slice(g1, "val").index
    test_idx = splits.slice(g1, "test").index
    print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}\n")

    all_rows = []
    for config in CONFIGS:
        print(f"\n=== Config: {config} ===")
        # DoW mean -- no exog needed; pass a placeholder X for date access.
        dow_X = pd.DataFrame(index=target.index)
        dow_X["const"] = 1.0
        all_rows += run_config(
            "dow_mean", dow_mean_factory(),
            dow_X, target, train_idx, val_idx, test_idx, config,
        )
        # SARIMAX with §5.2.5 raw 10 exog
        all_rows += run_config(
            "sarimax", make_sarimax_factory(SARIMAX_ORDER, SARIMAX_SEASONAL),
            X_full_sarimax, target, train_idx, val_idx, test_idx, config,
        )
        # XGBoost RMSE-best with consensus 23
        all_rows += run_config(
            "xgboost", make_xgboost_factory(XGB_BEST, seed=42),
            X_cons_full, target, train_idx, val_idx, test_idx, config,
        )

    df = pd.DataFrame(all_rows)
    out_path = ROOT / "artefacts" / "metrics" / "drift_aware_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path.relative_to(ROOT)}")

    print("\n--- Test MAPE by (model, config) ---")
    pivot = df[df["block"] == "test"].pivot_table(
        index="model", columns="config", values="MAPE",
    )
    pivot["Delta_vs_expanding"] = (
        pivot["sliding_450_rulsif"] - pivot["expanding"]
        if "sliding_450_rulsif" in pivot.columns else np.nan
    )
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
