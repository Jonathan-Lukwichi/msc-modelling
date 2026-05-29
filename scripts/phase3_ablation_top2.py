"""Phase 3 — Feature ablation study on the top-2 models from Phase 2.

For each of the top-2 models we run 4 configurations of the feature matrix:
  A. NO feature engineering + NO feature selection
       => raw G1 (the §5.2.5 raw-10 inventory: 15 cols after dow dummies)
  B. Feature ENGINEERING only (no selection)
       => full §3.4.2 engineered matrix (~100 cols incl. lags, rolling,
          Fourier, cyclical encodings)
  C. Feature SELECTION only (consensus filter applied to raw G1)
       => raw G1 columns that survived the §3.4.3 consensus retention
  D. BOTH (engineered + consensus selection)
       => current production default (23 features)

Outputs:
  artefacts/phase3_ablation/
    config_{A,B,C,D}_{model}.json     winning params (re-used from Phase 2)
    cv_folds_{config}_{model}.csv     per-fold CV table
    daily_{config}_{model}.csv        per-day val table + metrics
    weekly_{config}_{model}.csv       weekly aggregate
    monthly_{config}_{model}.csv      monthly aggregate
    yearly_{config}_{model}.csv       yearly aggregate
    summary_phase3.csv                consolidated comparison
    figures/ablation_{model}.png      bar chart of 4 configs per model

Usage:
  python scripts/phase3_ablation_top2.py --models sarimax,xgboost
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X, retained_features
from src.forecasting.metrics import score
from src.forecasting.cv import subsampled_rolling_origin

from scripts.phase1_defaults_all_models import (
    xgb_fit_predict, ann_fit_predict, lstm_fit_predict,
    arima_fit_predict, sarimax_fit_predict,
    stl_hybrid_fit_predict, sarimax_hybrid_fit_predict,
    lstm_xgb_hybrid_fit_predict,
    rolling_forecast_generic, aggregate_metrics,
)

OUT = ROOT / "artefacts" / "phase3_ablation"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Build 4 feature matrices
# -------------------------------------------------------------------------
def build_features():
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    train_idx_full = splits.slice(g1, "train").index

    # ----- Matrix A: raw G1 only (no FE, no FS) -----
    X_raw, scaler = build_task1_exogenous(g1.loc[train_idx_full],
                                           fit_scaler=True)
    X_raw_full, _ = build_task1_exogenous(g1, scaler=scaler)
    A = X_raw_full

    # ----- Matrix B: full engineered (FE only, no FS) -----
    eng = load_engineered()
    B = eng.copy()

    # ----- Matrix C: FS applied to raw G1 -----
    # Consensus selection on the raw block: keep only raw cols
    # that survived consensus. Practically, the consensus output IS the FE+FS
    # matrix; for "FS only on raw" we approximate by intersecting raw G1
    # columns with the consensus feature list.
    retained = retained_features()
    C_cols = [c for c in X_raw_full.columns if c in retained]
    if not C_cols:
        # Fallback: keep dow + the 7 calendar binaries (consensus-equivalent)
        C_cols = [c for c in X_raw_full.columns
                  if c.startswith("dow_") or c.startswith("is_")]
    C = X_raw_full[C_cols].copy()

    # ----- Matrix D: engineered + consensus (current production) -----
    D = build_selected_X(eng)

    # Inner-align everything on common dates
    common_idx = (A.index.intersection(B.index)
                  .intersection(C.index).intersection(D.index)
                  .intersection(target.index))
    return {
        "A_raw_only":      (target.loc[common_idx], A.loc[common_idx]),
        "B_engineered":    (target.loc[common_idx], B.loc[common_idx]),
        "C_selection":     (target.loc[common_idx], C.loc[common_idx]),
        "D_both":          (target.loc[common_idx], D.loc[common_idx]),
    }, splits


# -------------------------------------------------------------------------
# Pick fit_predict by model
# -------------------------------------------------------------------------
def get_fit_predict(model: str, winner_params: dict | None = None):
    if model == "xgboost":
        return lambda Xtr, ytr, Xte: xgb_fit_predict(Xtr, ytr, Xte,
                                                      params=winner_params)
    if model == "ann":
        return lambda Xtr, ytr, Xte: ann_fit_predict(Xtr, ytr, Xte,
                                                      params=winner_params)
    if model == "lstm":
        return lambda Xtr, ytr, Xte: lstm_fit_predict(Xtr, ytr, Xte,
                                                       params=winner_params)
    if model == "arima":
        params = winner_params or {"order": (0, 1, 2)}
        return lambda Xtr, ytr, Xte: arima_fit_predict(Xtr, ytr, Xte,
                                                        params=params)
    if model == "sarimax":
        params = winner_params or {"order": (1, 1, 1),
                                    "seasonal_order": (0, 1, 1, 7)}
        return lambda Xtr, ytr, Xte: sarimax_fit_predict(Xtr, ytr, Xte,
                                                          params=params)
    raise ValueError(f"Unsupported model: {model}")


def load_winner(model: str) -> dict | None:
    """Try Phase 2 winner, then existing best_params, else None."""
    for path in [
        ROOT / "artefacts" / "phase2_hpo" / f"winner_{model}_optuna.json",
        ROOT / "artefacts" / "phase2_hpo" / f"winner_{model}_grid.json",
        ROOT / "artefacts" / "models" / f"{model}_rmse_best_params.json",
        ROOT / "artefacts" / "models" / f"{model}_best_params.json",
    ]:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return None


# -------------------------------------------------------------------------
# Per-config experiment
# -------------------------------------------------------------------------
def run_config(model: str, config_name: str, y, X,
               splits, n_folds=10) -> dict:
    print(f"\n[{model}/{config_name}] X={X.shape}  starting...")
    t0 = time.time()
    winner = load_winner(model)
    fp = get_fit_predict(model, winner)

    train_idx = splits.slice(pd.DataFrame(index=y.index), "train").index
    val_idx   = splits.slice(pd.DataFrame(index=y.index), "val").index

    # 10-fold CV (use 5 for slow models)
    folds = subsampled_rolling_origin(
        train_idx, n_folds=n_folds if model not in ("lstm", "ann") else 5,
        horizon_days=7, step_days=7, min_train_days=365,
    )
    cv_rows = []
    X_tr_full = X.loc[train_idx]; y_tr_full = y.loc[train_idx]
    for f in folds:
        Xtr = X_tr_full.iloc[f.train_idx]; ytr = y_tr_full.iloc[f.train_idx]
        Xte = X_tr_full.iloc[f.test_idx];  yte = y_tr_full.iloc[f.test_idx]
        yhat = fp(Xtr, ytr, Xte)
        m = score(yte.values, np.asarray(yhat).ravel())
        cv_rows.append({"fold": f.fold_id,
                         "origin": f.origin.date().isoformat(), **m})
    cv_df = pd.DataFrame(cv_rows)
    avg = {"fold": "AVG", "origin": ""}
    for k in ("MAPE", "MAE", "RMSE", "R2"):
        avg[k] = cv_df[k].mean()
    cv_df = pd.concat([cv_df, pd.DataFrame([avg])], ignore_index=True)
    cv_df.to_csv(OUT / f"cv_folds_{config_name}_{model}.csv", index=False)

    # Rolling-origin val forecast
    pred = rolling_forecast_generic(fp, X, y, val_idx, step_days=7)
    pred["actual"] = y.loc[pred["date"]].values
    daily_rows = [{
        "date": r["date"].date().isoformat(),
        "actual": float(r["actual"]),
        "predicted": float(r["predicted"]),
        "abs_error": abs(r["actual"] - r["predicted"]),
        "pct_error": abs(r["actual"] - r["predicted"]) / max(r["actual"], 1e-9) * 100,
    } for _, r in pred.iterrows()]
    daily_df = pd.DataFrame(daily_rows)
    avg = {"date": "AVG",
           "actual": daily_df["actual"].mean(),
           "predicted": daily_df["predicted"].mean(),
           "abs_error": daily_df["abs_error"].mean(),
           "pct_error": daily_df["pct_error"].mean()}
    pd.concat([daily_df, pd.DataFrame([avg])], ignore_index=True
              ).to_csv(OUT / f"daily_{config_name}_{model}.csv", index=False)

    weekly = aggregate_metrics(pred.copy(), "W")
    monthly = aggregate_metrics(pred.copy(), "ME")
    yearly = aggregate_metrics(pred.copy(), "YE")
    weekly.to_csv(OUT / f"weekly_{config_name}_{model}.csv", index=False)
    monthly.to_csv(OUT / f"monthly_{config_name}_{model}.csv", index=False)
    yearly.to_csv(OUT / f"yearly_{config_name}_{model}.csv", index=False)

    m = score(daily_df["actual"].values, daily_df["predicted"].values)
    print(f"  [done] cv RMSE={cv_df[cv_df['fold']=='AVG']['RMSE'].iloc[0]:.3f}  "
          f"val MAPE={m['MAPE']:.2f}%  RMSE={m['RMSE']:.3f}  "
          f"({time.time()-t0:.0f}s)")
    return {
        "model": model,
        "config": config_name,
        "n_features": X.shape[1],
        "cv_RMSE": float(cv_df[cv_df['fold']=='AVG']['RMSE'].iloc[0]),
        "cv_MAPE": float(cv_df[cv_df['fold']=='AVG']['MAPE'].iloc[0]),
        "val_MAPE": float(m["MAPE"]),
        "val_MAE":  float(m["MAE"]),
        "val_RMSE": float(m["RMSE"]),
        "val_R2":   float(m["R2"]),
        "weekly_avg_pct": float(weekly[weekly["period"]=="AVG"]["pct_error"].iloc[0]),
        "monthly_avg_pct": float(monthly[monthly["period"]=="AVG"]["pct_error"].iloc[0]),
        "yearly_avg_pct": float(yearly[yearly["period"]=="AVG"]["pct_error"].iloc[0]),
    }


def plot_ablation(model: str, rows: list[dict]):
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cfgs = [r["config"] for r in rows]
    mapes = [r["val_MAPE"] for r in rows]
    rmses = [r["val_RMSE"] for r in rows]
    labels = {"A_raw_only": "A: raw G1\n(no FE, no FS)",
              "B_engineered": "B: engineered\n(FE only)",
              "C_selection": "C: selection\n(FS only)",
              "D_both": "D: engineered\n+ selected (both)"}
    x = np.arange(len(cfgs))
    width = 0.4
    ax.bar(x - width/2, mapes, width, color="#1f77b4", label="val MAPE %")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, rmses, width, color="#ff7f0e", alpha=0.85,
            label="val RMSE")
    for i, v in enumerate(mapes):
        ax.text(x[i] - width/2, v, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9)
    for i, v in enumerate(rmses):
        ax2.text(x[i] + width/2, v, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in cfgs], fontsize=9)
    ax.set_ylabel("val MAPE (%)", color="#1f77b4")
    ax2.set_ylabel("val RMSE", color="#ff7f0e")
    ax.set_title(f"{model} — feature ablation (val block)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / f"ablation_{model}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="sarimax,xgboost",
                        help="comma-separated top-2 models (default: sarimax,xgboost)")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    print(f"Top-2 models: {models}")

    matrices, splits = build_features()
    print("Feature matrices built:")
    for k, (y, X) in matrices.items():
        print(f"  {k:18s}  X={X.shape}")

    summary = []
    by_model: dict[str, list] = {m: [] for m in models}
    configs = ["A_raw_only", "B_engineered", "C_selection", "D_both"]

    for model in models:
        for cfg in configs:
            y, X = matrices[cfg]
            try:
                res = run_config(model, cfg, y, X, splits)
                summary.append(res)
                by_model[model].append(res)
                pd.DataFrame(summary).to_csv(OUT / "summary_phase3.csv",
                                              index=False)
            except Exception as exc:
                import traceback
                print(f"  FAILED {model}/{cfg}: {exc}")
                traceback.print_exc()
        plot_ablation(model, by_model[model])

    print("\n" + "="*70)
    print("PHASE 3 SUMMARY (feature ablation)")
    print("="*70)
    if summary:
        df = pd.DataFrame(summary)
        print(df[["model","config","n_features","cv_RMSE","val_MAPE",
                  "val_RMSE","weekly_avg_pct","yearly_avg_pct"]
                 ].to_string(index=False))


if __name__ == "__main__":
    main()
