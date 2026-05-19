"""Uncertainty quantification: Quantile XGBoost + Split-Conformal Prediction.

Implements two complementary UQ methods on the RMSE-tuned XGBoost backbone:

  (A) Quantile XGBoost — three models with pinball loss at alpha = 0.025, 0.5,
      0.975. Gives non-parametric 95% prediction interval that captures the
      right-skew of count data (skew = +1.22 on train per Ch5 §5.2.1).

  (B) Split-Conformal Prediction — distribution-free PI with finite-sample
      coverage guarantee. Calibrated on val residuals, applied to test.

Plus the inherited intervals (already in artefacts):
  - SARIMAX Gaussian PI
  - NB GLM NB-pmf PI

Also evaluates point-prediction performance at three aggregation periods:
  - WEEKLY  (Mon-Sun aggregation, sum of daily counts)
  - MONTHLY (calendar month sum)
  - YEARLY  (calendar year sum)

For each (model, period), reports MAPE/MAE/RMSE.

Output:
  artefacts/predictions/test/xgboost_quantile.csv
  artefacts/metrics/uq_coverage.csv             -- coverage / width / CWS
  artefacts/metrics/aggregated_metrics_period.csv -- weekly / monthly / yearly
  artefacts/figures/fig_6_uq_intervals.png      -- PI comparison plot
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")

# RMSE-tuned XGBoost params (from §18 audit)
XGB_PARAMS = {"n_estimators": 500, "max_depth": 3,
                "learning_rate": 0.01, "subsample": 1.0}
ALPHA = 0.05   # 95% PI -> alpha=0.05 -> use 0.025 and 0.975 quantiles


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

splits = Splits.from_config()
g1 = load_g1()
target = g1["total_daily_arrivals"]
eng = load_engineered()
X_consensus = build_selected_X(eng)
df = pd.concat([target.rename("y"), X_consensus], axis=1, join="inner").dropna()
train_idx = splits.slice(g1, "train").index.intersection(df.index)
val_idx = splits.slice(g1, "val").index.intersection(df.index)
test_idx = splits.slice(g1, "test").index.intersection(df.index)
print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")


# ---------------------------------------------------------------------------
# (A) Quantile XGBoost - 3 quantiles, rolling weekly refit
# ---------------------------------------------------------------------------

def rolling_quantile_xgb(target_idx):
    """For each weekly origin, fit 3 quantile-XGB models (lower, median, upper).
    Return a DataFrame with columns: date, predicted_lower, predicted_median,
    predicted_upper.
    """
    from xgboost import XGBRegressor
    rows = []
    origin = df.index[df.index < target_idx[0]][-1]
    remaining = list(target_idx)
    while remaining:
        h = min(7, len(remaining))
        future = remaining[:h]
        tr_dates = df.index[df.index <= origin]
        X_tr = df.loc[tr_dates].drop(columns=["y"]).values
        y_tr = df.loc[tr_dates, "y"].values
        X_fu = df.loc[future].drop(columns=["y"]).values
        preds = {}
        for q_alpha, key in [(ALPHA / 2, "lower"),
                              (0.5, "median"),
                              (1 - ALPHA / 2, "upper")]:
            m = XGBRegressor(**XGB_PARAMS,
                              objective="reg:quantileerror",
                              quantile_alpha=q_alpha,
                              random_state=42, verbosity=0, n_jobs=-1)
            m.fit(X_tr, y_tr)
            preds[key] = m.predict(X_fu)
        for i, d in enumerate(future):
            rows.append({
                "date": d,
                "predicted_lower":  float(preds["lower"][i]),
                "predicted_median": float(preds["median"][i]),
                "predicted_upper":  float(preds["upper"][i]),
            })
        origin = future[-1]
        remaining = remaining[h:]
    out = pd.DataFrame(rows).set_index("date")
    return out


# ---------------------------------------------------------------------------
# (B) Split-Conformal — calibrated on val, applied to test
# ---------------------------------------------------------------------------

def conformal_width(val_preds: pd.Series, val_actuals: pd.Series, alpha: float = 0.05) -> float:
    """Split-conformal: empirical quantile of |residuals| on calibration set."""
    abs_resid = (val_actuals - val_preds).abs().dropna().values
    n = len(abs_resid)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q = min(q, 1.0)
    return float(np.quantile(abs_resid, q))


# ---------------------------------------------------------------------------
# Coverage / Width / CWS (Winkler score)
# ---------------------------------------------------------------------------

def evaluate_pi(actual, lower, upper, alpha=0.05):
    """Coverage = fraction inside PI. Width = mean width.
    Winkler score = width + (2/alpha) * shortfall (lower=better)."""
    actual = np.asarray(actual); lower = np.asarray(lower); upper = np.asarray(upper)
    inside = (actual >= lower) & (actual <= upper)
    coverage = float(np.mean(inside))
    width = float(np.mean(upper - lower))
    # Winkler (interval) score
    below = actual < lower
    above = actual > upper
    short_lower = (lower - actual) * below
    short_upper = (actual - upper) * above
    winkler = float(np.mean((upper - lower) + (2 / alpha) * (short_lower + short_upper)))
    return {"coverage": coverage, "width": width, "winkler_score": winkler,
            "target_coverage": 1 - alpha}


# ---------------------------------------------------------------------------
# Weekly / Monthly / Yearly evaluation
# ---------------------------------------------------------------------------

def evaluate_aggregated(actual_series: pd.Series, pred_series: pd.Series,
                        model_name: str, block: str) -> list[dict]:
    """Aggregate daily counts to weekly / monthly / yearly, then score."""
    rows = []
    common = actual_series.index.intersection(pred_series.index)
    a = actual_series.loc[common]; p = pred_series.loc[common]
    for freq, label in [("W-SUN", "weekly"), ("MS", "monthly"), ("YS", "yearly")]:
        ag_a = a.resample(freq).sum()
        ag_p = p.resample(freq).sum()
        # Drop partial periods at the edges
        if freq == "W-SUN":
            # Only keep full 7-day weeks
            counts = a.resample(freq).size()
            keep = counts[counts == 7].index
            ag_a = ag_a.loc[keep]; ag_p = ag_p.loc[keep]
        if freq == "MS":
            counts = a.resample(freq).size()
            keep = counts[counts >= 28].index
            ag_a = ag_a.loc[keep]; ag_p = ag_p.loc[keep]
        if len(ag_a) < 2:
            continue
        s = score(ag_a.values, ag_p.values)
        rows.append({"model": model_name, "block": block, "period": label,
                      "n_periods": len(ag_a), **s})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    coverage_rows = []
    aggregated_rows = []

    # ============ (A) QUANTILE XGBOOST ============
    print("\n" + "=" * 70)
    print("(A) Quantile XGBoost (alpha=0.025, 0.5, 0.975)")
    print("=" * 70)

    t0 = time.time()
    print("Running rolling Quantile XGBoost on val (3 quantiles × weekly refit)...")
    qxgb_val = rolling_quantile_xgb(val_idx)
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("Running rolling Quantile XGBoost on test...")
    qxgb_test = rolling_quantile_xgb(test_idx)
    print(f"  ({time.time() - t0:.0f}s)")

    # Attach actuals
    qxgb_val["actual"] = target.loc[qxgb_val.index]
    qxgb_test["actual"] = target.loc[qxgb_test.index]
    qxgb_val.reset_index().to_csv(
        ROOT / "artefacts" / "predictions" / "xgboost_quantile.csv", index=False)
    qxgb_test.reset_index().to_csv(
        ROOT / "artefacts" / "predictions" / "test" / "xgboost_quantile.csv", index=False)

    # PI evaluation
    for block, df_pi in [("val", qxgb_val), ("test", qxgb_test)]:
        pi = evaluate_pi(df_pi["actual"], df_pi["predicted_lower"],
                          df_pi["predicted_upper"])
        # Point performance of the median
        pt = score(df_pi["actual"].values, df_pi["predicted_median"].values)
        coverage_rows.append({"model": "Quantile_XGBoost", "block": block,
                                "method": "Quantile (a=0.025, 0.975)",
                                "alpha": ALPHA, **pi, **pt})
        print(f"  Quantile XGB ({block}): coverage={pi['coverage']:.3f} "
              f"(target {1-ALPHA:.2f}), width={pi['width']:.2f}, "
              f"median_MAPE={pt['MAPE']:.2f}%, Winkler={pi['winkler_score']:.2f}")

    # ============ (B) SPLIT-CONFORMAL on XGBoost RMSE-best ============
    print("\n" + "=" * 70)
    print("(B) Split-Conformal Prediction (calibration = val residuals)")
    print("=" * 70)

    # Load RMSE-best XGBoost rolling val + test predictions
    val_pt = pd.read_csv(ROOT / "artefacts" / "predictions" / "xgboost_rmse.csv",
                          parse_dates=["date"]).set_index("date")
    test_pt = pd.read_csv(ROOT / "artefacts" / "predictions" / "test" / "xgboost_rmse.csv",
                           parse_dates=["date"]).set_index("date")

    # First half of val for calibration, second half for honest coverage check
    half = len(val_pt) // 2
    calib = val_pt.iloc[:half]
    honest_val = val_pt.iloc[half:]
    width = conformal_width(calib["predicted"], calib["actual"], alpha=ALPHA)
    print(f"  Conformal width (95% PI) from {len(calib)} cal-fold residuals: "
          f"q = {width:.2f}")

    # Apply to (honest val) and test
    for name, df_pt in [("conformal_val_honest", honest_val),
                          ("conformal_test", test_pt)]:
        df_c = df_pt.copy()
        df_c["predicted_lower"] = df_c["predicted"] - width
        df_c["predicted_upper"] = df_c["predicted"] + width
        pi = evaluate_pi(df_c["actual"], df_c["predicted_lower"], df_c["predicted_upper"])
        pt = score(df_c["actual"].values, df_c["predicted"].values)
        block = "val_2nd_half" if "val" in name else "test"
        coverage_rows.append({"model": "Conformal_XGBoost", "block": block,
                                "method": "Split-Conformal (cal=val_1st_half)",
                                "alpha": ALPHA, **pi, **pt})
        print(f"  Conformal ({block}): coverage={pi['coverage']:.3f} "
              f"(target {1-ALPHA:.2f}), width={pi['width']:.2f}, "
              f"point_MAPE={pt['MAPE']:.2f}%, Winkler={pi['winkler_score']:.2f}")

    # ============ Inherit SARIMAX + NB GLM intervals ============
    print("\n" + "=" * 70)
    print("(C) Inherited parametric intervals (SARIMAX + NB GLM)")
    print("=" * 70)

    for tag, fn in [("SARIMAX_Gaussian", "sarimax.csv"),
                      ("NB_GLM_NBpmf",      "nbglm.csv")]:
        for block_name, sub in [
            ("val", ROOT / "artefacts" / "predictions" / fn),
            ("test", ROOT / "artefacts" / "predictions" / "test" / fn),
        ]:
            if not sub.exists():
                continue
            d = pd.read_csv(sub, parse_dates=["date"])
            if "lower_95" not in d.columns:
                continue
            pi = evaluate_pi(d["actual"], d["lower_95"], d["upper_95"])
            pt = score(d["actual"].values, d["predicted"].values)
            coverage_rows.append({"model": tag, "block": block_name,
                                    "method": "Parametric",
                                    "alpha": ALPHA, **pi, **pt})
            print(f"  {tag} ({block_name}): coverage={pi['coverage']:.3f} "
                  f"(target {1-ALPHA:.2f}), width={pi['width']:.2f}, "
                  f"point_MAPE={pt['MAPE']:.2f}%, Winkler={pi['winkler_score']:.2f}")

    cov_df = pd.DataFrame(coverage_rows)
    out_cov = ROOT / "artefacts" / "metrics" / "uq_coverage.csv"
    cov_df.to_csv(out_cov, index=False)
    print(f"\nWrote: {out_cov.relative_to(ROOT)}")

    # ============ Weekly / Monthly / Yearly evaluation ============
    print("\n" + "=" * 70)
    print("(D) Aggregated evaluation: Weekly / Monthly / Yearly")
    print("=" * 70)
    for model_tag, fname in [
        ("Naive_yest", None),
        ("ARIMA", "arima.csv"),
        ("SARIMAX", "sarimax.csv"),
        ("NB_GLM", "nbglm.csv"),
        ("XGBoost_RMSE", "xgboost_rmse.csv"),
        ("ANN_RMSE", "ann_rmse.csv"),
        ("LSTM_RMSE", "lstm_rmse.csv"),
        ("Quantile_XGB", "xgboost_quantile.csv"),
    ]:
        if model_tag == "Naive_yest":
            # Build from target
            val_pred = target.shift(1).reindex(val_idx)
            test_pred = target.shift(1).reindex(test_idx)
            actual_val = target.loc[val_idx]
            actual_test = target.loc[test_idx]
            aggregated_rows.extend(evaluate_aggregated(actual_val, val_pred,
                                                        model_tag, "val"))
            aggregated_rows.extend(evaluate_aggregated(actual_test, test_pred,
                                                        model_tag, "test"))
            continue
        for block, csv_dir in [("val", ROOT / "artefacts" / "predictions"),
                                ("test", ROOT / "artefacts" / "predictions" / "test")]:
            p = csv_dir / fname
            if not p.exists():
                continue
            d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            # Determine prediction column
            col = "predicted" if "predicted" in d.columns else "predicted_median"
            actual = target.loc[d.index]
            aggregated_rows.extend(evaluate_aggregated(actual, d[col],
                                                         model_tag, block))

    agg_df = pd.DataFrame(aggregated_rows)
    out_agg = ROOT / "artefacts" / "metrics" / "aggregated_metrics_period.csv"
    agg_df.to_csv(out_agg, index=False)
    print(f"Wrote: {out_agg.relative_to(ROOT)}")

    # Quick console preview
    if not agg_df.empty:
        print("\nAggregated MAPE by (model, period, block):")
        pivot = agg_df.pivot_table(index="model", columns=["period", "block"],
                                     values="MAPE", aggfunc="first")
        pd.set_option("display.float_format", lambda v: f"{v:.2f}")
        print(pivot.to_string())

    # ============ Figure: 4-band PI comparison on val ============
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patheffects import withStroke
        NAVY, TEAL, AMBER, GREEN, ROSE, NEUTRAL = (
            "#1e6091", "#0d9488", "#d97706", "#16a34a", "#dc2626", "#475569")
        plt.rcParams.update({
            "font.size": 11, "font.family": "sans-serif",
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
            "figure.dpi": 100, "savefig.dpi": 300,
            "savefig.bbox": "tight", "savefig.facecolor": "white",
        })

        # Use first 60 val days for legibility
        val_days = qxgb_val.index[:60]
        sar = pd.read_csv(ROOT / "artefacts" / "predictions" / "sarimax.csv",
                           parse_dates=["date"]).set_index("date").reindex(val_days)
        nb = pd.read_csv(ROOT / "artefacts" / "predictions" / "nbglm.csv",
                          parse_dates=["date"]).set_index("date").reindex(val_days)
        qxgb_sub = qxgb_val.loc[val_days]
        # Conformal on these 60 days (using same width)
        xgb_pt_60 = val_pt.reindex(val_days)
        confm_lower = xgb_pt_60["predicted"] - width
        confm_upper = xgb_pt_60["predicted"] + width

        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        actual_60 = target.reindex(val_days)
        for ax, (tag, lower, upper, point, color) in zip(axes, [
            ("SARIMAX (Gaussian PI)",      sar["lower_95"], sar["upper_95"],     sar["predicted"], NAVY),
            ("NB GLM (NB-pmf PI)",          nb["lower_95"], nb["upper_95"],       nb["predicted"], AMBER),
            ("Quantile XGBoost (a=0.025/0.975)", qxgb_sub["predicted_lower"],
                                                   qxgb_sub["predicted_upper"],
                                                   qxgb_sub["predicted_median"], TEAL),
            ("Split-Conformal XGBoost (width fr val cal)", confm_lower,
                                                            confm_upper,
                                                            xgb_pt_60["predicted"], ROSE),
        ]):
            ax.fill_between(val_days, lower, upper, color=color, alpha=0.2,
                             label="95% PI")
            ax.plot(val_days, point, color=color, linewidth=1.2, label="point")
            ax.plot(val_days, actual_60, color=NEUTRAL, linewidth=1.5,
                     linestyle="-", label="actual", zorder=5)
            ax.set_title(tag, loc="left", fontsize=11)
            ax.set_ylabel("arrivals")
            ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
        axes[-1].set_xlabel("Date (first 60 val days)")
        fig.suptitle("Figure 6.UQ — 95% prediction intervals: four UQ methods compared",
                      fontsize=13, y=1.005)
        out_fig = ROOT / "artefacts" / "figures" / "fig_6_uq_intervals.png"
        plt.tight_layout()
        plt.savefig(out_fig)
        plt.close()
        print(f"Wrote: {out_fig.relative_to(ROOT)}")
    except Exception as exc:
        print(f"  Figure failed: {exc}")


if __name__ == "__main__":
    main()
