"""Random Forest baseline for the Chapter 6 leaderboard (Priority 1).

Random Forest is the most-cited model family in the 45-paper ED-forecasting
corpus (64% prevalence). Adding it to the leaderboard makes the comparison
set match the field's modal expectation.

Protocol: same as the other ML rows in scripts/19_rerun_rmse_best.py.
  - Consensus 23-feature set
  - Rolling weekly refit (RollingForecaster)
  - Val and test blocks held out
  - Quick HPO via small grid (no full audit; this is a baseline)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

from src.forecasting.io import load_g1, Splits
from src.forecasting.consensus import build_selected_X
from src.forecasting.engineering import load_engineered
from src.forecasting.metrics import score
from src.forecasting.rolling import RollingForecaster, FoldPrediction


def rf_factory(params: dict, seed: int = 42):
    """Return a fitted-model factory for RollingForecaster."""
    def factory(X_train, y_train, sample_weight=None):
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        # Wrap with the FoldPrediction-returning .predict
        class _Fitted:
            def predict(self, X_future, h):
                return FoldPrediction(yhat=model.predict(X_future))
        return _Fitted()
    return factory


def quick_hpo(X_train, y_train, splits, target):
    """Tiny 6-cell grid; pick min cv_RMSE across 5 rolling inner folds."""
    from src.forecasting.cv import subsampled_rolling_origin

    train_idx = splits.slice(load_g1(), "train").index
    folds = subsampled_rolling_origin(train_idx, n_folds=5, horizon_days=7)
    grid = [
        {"n_estimators": 200, "max_depth": 8,  "min_samples_leaf": 3},
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5},
        {"n_estimators": 500, "max_depth": 8,  "min_samples_leaf": 5},
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 5},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5},
    ]
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    eng = load_engineered()
    X_full = build_selected_X(eng).loc[train_idx]
    y_train = target.loc[train_idx]
    best = (None, np.inf)
    print("Quick HPO over 6 RF configs x 5 inner folds...")
    for i, p in enumerate(grid, 1):
        rmses = []
        for fold in folds:
            X_tr = X_full.iloc[fold.train_idx]
            y_tr = y_train.iloc[fold.train_idx]
            X_ev = X_full.iloc[fold.test_idx]
            y_ev = y_train.iloc[fold.test_idx]
            m = RandomForestRegressor(
                n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"], random_state=42,
                n_jobs=-1,
            )
            m.fit(X_tr, y_tr)
            yhat = m.predict(X_ev)
            rmses.append(float(np.sqrt(((y_ev.values - yhat) ** 2).mean())))
        cv_rmse = float(np.mean(rmses))
        print(f"  [{i}/6] {p} -> cv_RMSE={cv_rmse:.3f}")
        if cv_rmse < best[1]:
            best = (p, cv_rmse)
    print(f"Best params: {best[0]}  cv_RMSE={best[1]:.3f}\n")
    return best[0]


def main():
    print("=" * 70)
    print("RANDOM FOREST BASELINE (Priority 1)")
    print("=" * 70 + "\n")

    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    eng = load_engineered()
    X = build_selected_X(eng)

    train_idx = splits.slice(g1, "train").index
    val_idx = splits.slice(g1, "val").index
    test_idx = splits.slice(g1, "test").index
    print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}\n")

    best_params = quick_hpo(X, target, splits, target)

    rows = []
    for block_name, blk_idx in [("val", val_idx), ("test", test_idx)]:
        print(f"--- Rolling {block_name} forecast ---")
        t0 = time.time()
        rf = RollingForecaster(
            model_factory=rf_factory(best_params),
            step_days=7, horizon_days=7,
            min_train_days=365,
        )
        out = rf.fit_predict(X=X, y=target, eval_index=blk_idx)
        out["actual"] = target.loc[out.index].values
        out = out.rename(columns={"yhat": "predicted"})
        s = score(out["actual"], out["predicted"])
        print(f"  {block_name} ({time.time() - t0:.0f}s): MAPE={s['MAPE']:.3f}  "
              f"MAE={s['MAE']:.2f}  RMSE={s['RMSE']:.2f}  R2={s['R2']:+.3f}\n")
        # Save predictions
        out_path = (ROOT / "artefacts" / "predictions" /
                      ("test" if block_name == "test" else "") /
                      "random_forest.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save = out.reset_index().rename(columns={"index": "date"})
        save.to_csv(out_path, index=False)
        rows.append({"block": block_name,
                     "MAPE": s["MAPE"], "MAE": s["MAE"],
                     "RMSE": s["RMSE"], "R2": s["R2"], "n": len(out)})

    # Save metrics
    m_path = ROOT / "artefacts" / "metrics" / "random_forest_metrics.csv"
    pd.DataFrame(rows).to_csv(m_path, index=False)
    print(f"Wrote: {m_path}")

    # Compute MASE against seasonal-naive
    from src.forecasting.metrics import mase
    test_pred = pd.read_csv(
        ROOT / "artefacts" / "predictions" / "test" / "random_forest.csv",
        parse_dates=["date"],
    ).set_index("date")
    train_y = target.loc[train_idx]
    val_y = target.loc[val_idx]
    val_pred = pd.read_csv(
        ROOT / "artefacts" / "predictions" / "random_forest.csv",
        parse_dates=["date"],
    ).set_index("date")
    val_mase = mase(val_y.values, val_pred["predicted"].values, train_y.values, seasonality=7)
    test_mase = mase(target.loc[test_idx].values, test_pred["predicted"].values,
                       train_y.values, seasonality=7)
    print(f"  val_MASE  = {val_mase:.3f}")
    print(f"  test_MASE = {test_mase:.3f}")

    # Append to leaderboard parquet
    from src.forecasting.leaderboard import append_row
    val_row = next(r for r in rows if r["block"] == "val")
    test_row = next(r for r in rows if r["block"] == "test")
    append_row(
        parquet_path=ROOT / "artefacts" / "leaderboard_canonical.parquet",
        model="random_forest", family="ml", criterion="rmse", seed=42,
        val_metrics={"MAPE": val_row["MAPE"], "RMSE": val_row["RMSE"], "MASE": val_mase},
        test_metrics={"MAPE": test_row["MAPE"], "RMSE": test_row["RMSE"], "MASE": test_mase},
        source_csv=str(m_path.relative_to(ROOT)),
    )
    print(f"Leaderboard updated.")


if __name__ == "__main__":
    main()
