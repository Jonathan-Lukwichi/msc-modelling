"""OOF residual hybrid rebuild (Prompt 4 orchestrator).

Migrates the canonical SARIMAX+XGBoost residual hybrid from the legacy
in-sample recipe (Zhang 2003) to the statistically honest out-of-fold
recipe (Khashei & Bijari 2011, Hewamalage et al. 2021) via
``src.forecasting.hybrids.oof.oof_residuals.OOFResidualHybrid``.

Scope of this orchestrator:
  - SARIMAX(1,1,1)(0,1,1)_7 as the base.
  - XGBoost (RMSE-best params) as the refiner.
  - Compares the OOF rebuild against the saved in-sample baseline
    (artefacts/predictions/hybrid_sarimax_xgb_rmse.csv) on the val and
    test blocks.

If the OOF rebuild improves val MAPE without breaking on test, the same
machinery is ready to extend to SARIMAX+LSTM and LSTM+XGB; for this
ship we stick to the single decisive demonstration.

Outputs:
  - artefacts/predictions/hybrid_sarimax_xgb_oof.csv         (val rolling)
  - artefacts/predictions/test/hybrid_sarimax_xgb_oof.csv    (test rolling)
  - artefacts/metrics/hybrid_oof_comparison.csv              (before/after)
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
from src.forecasting.rolling import make_sarimax_factory
from src.forecasting.hybrids.oof.oof_residuals import (
    OOFResidualHybrid, xgb_refiner_factory,
)


SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL = (0, 1, 1, 7)
XGB_BEST = {
    "n_estimators": 300, "max_depth": 5,
    "learning_rate": 0.01, "subsample": 1.0,
}


def main() -> None:
    print("=" * 70)
    print("PROMPT 4 — OOF residual hybrid rebuild  (SARIMAX + XGBoost)")
    print("=" * 70 + "\n")

    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]

    # Exogenous block for SARIMAX (the §5.2.5 raw 10).
    X_train_block, scaler = build_task1_exogenous(
        splits.slice(g1, "train"), fit_scaler=True
    )
    X_full_sarimax, _ = build_task1_exogenous(g1, scaler=scaler)

    # Consensus 23 features for the XGBoost refiner.
    eng = load_engineered()
    X_cons = build_selected_X(eng)

    train_idx = splits.slice(g1, "train").index
    val_idx = splits.slice(g1, "val").index
    test_idx = splits.slice(g1, "test").index
    print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    y_train = target.loc[train_idx]
    X_train_sarimax = X_full_sarimax.loc[train_idx]
    X_train_cons = X_cons.loc[train_idx]

    # We need a single X for the hybrid: the base uses X_sarimax and the
    # refiner uses X_cons. OOFResidualHybrid takes one X. We pass X_cons
    # for the refiner and rely on the base_factory ignoring the exog when
    # SARIMAX's per-fold X is the raw 10 -- patch by wrapping the factory
    # to look up the right exog by date.
    sarimax_factory_raw = make_sarimax_factory(
        order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL,
    )

    def sarimax_factory_swapped(X_train_consensus, y_train, sample_weight=None):
        # Replace the consensus X with the raw-10 X for SARIMAX fitting.
        X_train_raw10 = X_full_sarimax.loc[y_train.index]
        return sarimax_factory_raw(X_train_raw10, y_train, sample_weight)

    # Build the hybrid.
    hybrid = OOFResidualHybrid(
        base_factory=sarimax_factory_swapped,
        refiner_factory=xgb_refiner_factory,
        standardize_residuals=True,
        nested_hpo=False,        # use the RMSE-best params verbatim
        horizon=7,
        min_train_days=365,
    )

    t0 = time.time()
    print(f"\nBuilding OOF residuals via SARIMAX rolling-fit on train block...")
    hybrid.fit(X_train_cons, y_train)
    print(f"  OOF residuals: {len(hybrid.oof_residuals)} days "
          f"(elapsed {time.time() - t0:.0f}s)")
    print(f"  OOF residual stats: mean={hybrid._res_mean:.3f}  "
          f"std={hybrid._res_std:.3f}")

    # Predict val and test
    rows = []
    for block_name, blk_idx in [("val", val_idx), ("test", test_idx)]:
        print(f"\n--- Rolling {block_name} forecast ---")
        t0 = time.time()
        # Re-fit the hybrid each origin happens inside RollingForecaster,
        # which OOFResidualHybrid.predict() invokes. The refiner is fixed
        # after fit().
        out = hybrid.predict(X_cons.loc[blk_idx], blk_idx)
        out["actual"] = target.loc[blk_idx].values
        s = score(out["actual"], out["predicted"])
        print(f"  {block_name} ({time.time() - t0:.0f}s):  MAPE={s['MAPE']:.3f}  "
              f"MAE={s['MAE']:.2f}  RMSE={s['RMSE']:.2f}  R2={s['R2']:+.3f}")

        # Save predictions
        out_path = (ROOT / "artefacts" / "predictions" /
                      ("test" if block_name == "test" else "") /
                      "hybrid_sarimax_xgb_oof.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_df = out.reset_index().rename(columns={"index": "date"})
        save_df.to_csv(out_path, index=False)

        rows.append({
            "hybrid": "SARIMAX+XGBoost", "block": block_name,
            "recipe": "OOF",
            "MAPE": s["MAPE"], "MAE": s["MAE"],
            "RMSE": s["RMSE"], "R2": s["R2"], "n": len(out),
        })

    # Compare against the saved in-sample numbers
    print("\n--- Comparison: OOF vs in-sample baseline (saved) ---")
    legacy_metrics = {
        "val":  {"MAPE": 12.637, "RMSE": 9.495},   # from RESULTS.md §4sexies
        "test": {"MAPE": 13.458, "RMSE": 10.833},  # from RESULTS.md
    }
    cmp_rows = []
    for block_name in ("val", "test"):
        oof = [r for r in rows if r["block"] == block_name][0]
        in_sample = legacy_metrics[block_name]
        cmp_rows.append({
            "hybrid": "SARIMAX+XGBoost", "block": block_name,
            "in_sample_MAPE": in_sample["MAPE"],
            "oof_MAPE": oof["MAPE"],
            "delta_MAPE": oof["MAPE"] - in_sample["MAPE"],
            "in_sample_RMSE": in_sample["RMSE"],
            "oof_RMSE": oof["RMSE"],
            "delta_RMSE": oof["RMSE"] - in_sample["RMSE"],
        })
    cmp_df = pd.DataFrame(cmp_rows)
    print(cmp_df.round(3).to_string(index=False))

    out_metrics = ROOT / "artefacts" / "metrics" / "hybrid_oof_comparison.csv"
    pd.DataFrame(rows + [
        {**r, "recipe": "in_sample_baseline"} for r in [
            {"hybrid": "SARIMAX+XGBoost", "block": "val",
             "MAPE": legacy_metrics["val"]["MAPE"],
             "RMSE": legacy_metrics["val"]["RMSE"]},
            {"hybrid": "SARIMAX+XGBoost", "block": "test",
             "MAPE": legacy_metrics["test"]["MAPE"],
             "RMSE": legacy_metrics["test"]["RMSE"]},
        ]
    ]).to_csv(out_metrics, index=False)
    cmp_df.to_csv(out_metrics.with_name("hybrid_oof_comparison_pairs.csv"),
                   index=False)
    print(f"\nWrote: {out_metrics.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
