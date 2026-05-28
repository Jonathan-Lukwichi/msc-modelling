"""Residual diagnostics for the headline XGBoost forecast (Priority 2).

Computes the standard residual checks expected by forecasting-specialist
reviewers (Box & Jenkins 1976; Hyndman & Athanasopoulos 2021 ch.3.3):
  - Mean and standard deviation of residuals
  - Ljung-Box Q-test on the residual series (lags=20)
  - Histogram + normality check (Shapiro-Wilk)
  - Autocorrelation function (ACF) up to lag 28
  - Q-Q plot vs Normal

The figure (4 panels) is saved to artefacts/figures/fig_6_residuals_xgboost.png
and the summary statistics to artefacts/metrics/residual_diagnostics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 100

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ljung_box(resid: np.ndarray, lags: int = 20) -> dict:
    """Return Ljung-Box Q statistic + p-value (chi-square with `lags` df)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    return {"Q": float(lb["lb_stat"].iloc[0]),
              "p_value": float(lb["lb_pvalue"].iloc[0])}


def diagnose_block(block_name: str, pred_csv: Path) -> dict:
    df = pd.read_csv(pred_csv, parse_dates=["date"]).set_index("date").sort_index()
    # Standardise column names
    cols = {c.lower(): c for c in df.columns}
    actual_col = cols.get("actual")
    pred_col = cols.get("predicted")
    if actual_col is None or pred_col is None:
        raise ValueError(f"Cannot find actual/predicted in {pred_csv}: {list(df.columns)}")
    resid = df[actual_col].values - df[pred_col].values

    summary = {
        "block": block_name,
        "n": int(len(resid)),
        "mean": float(np.mean(resid)),
        "std": float(np.std(resid, ddof=1)),
        "min": float(np.min(resid)),
        "max": float(np.max(resid)),
        "skew": float(stats.skew(resid)),
        "kurtosis_excess": float(stats.kurtosis(resid)),
    }
    summary.update(ljung_box(resid, lags=20))
    summary["LB_lags"] = 20
    # Shapiro-Wilk (sample size <= 5000)
    if len(resid) <= 5000:
        sw_stat, sw_p = stats.shapiro(resid)
        summary["shapiro_W"] = float(sw_stat)
        summary["shapiro_p"] = float(sw_p)
    return summary, resid, df


def plot_diagnostics(resid_val: np.ndarray, resid_test: np.ndarray,
                      dates_val: pd.DatetimeIndex, dates_test: pd.DatetimeIndex,
                      out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (1) Residual vs time (test only -- the operational block)
    ax = axes[0, 0]
    ax.plot(dates_test, resid_test, linewidth=0.7, color="#1e6091", label="test")
    ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)
    ax.set_title("(a) Residuals over time (test block, 396 days)",
                 loc="left", fontsize=10)
    ax.set_ylabel("Residual (patients)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)

    # (2) Histogram + Normal overlay
    ax = axes[0, 1]
    ax.hist(resid_test, bins=40, color="#0d9488", alpha=0.7, density=True,
             edgecolor="white")
    x = np.linspace(resid_test.min(), resid_test.max(), 200)
    sigma = np.std(resid_test, ddof=1)
    ax.plot(x, stats.norm.pdf(x, loc=np.mean(resid_test), scale=sigma),
             color="#dc2626", linewidth=1.5, label="Normal fit")
    ax.set_title(f"(b) Histogram of residuals  (test)  N(0, σ={sigma:.2f})",
                 loc="left", fontsize=10)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (3) ACF
    from statsmodels.graphics.tsaplots import plot_acf
    ax = axes[1, 0]
    plot_acf(resid_test, lags=28, ax=ax, zero=False)
    ax.set_title("(c) Residual autocorrelation (test, lags 1–28)",
                 loc="left", fontsize=10)
    ax.set_xlabel("Lag (days)")
    ax.grid(alpha=0.3)

    # (4) Q-Q plot
    ax = axes[1, 1]
    stats.probplot(resid_test, dist="norm", plot=ax)
    ax.set_title("(d) Q–Q plot vs Normal (test)", loc="left", fontsize=10)
    ax.get_lines()[0].set_color("#475569")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("#dc2626")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Figure 6.R — Residual diagnostics for the headline XGBoost forecast on the held-out test block",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Wrote: {out_path.relative_to(ROOT)}")


def main():
    print("=" * 70)
    print("RESIDUAL DIAGNOSTICS for the headline XGBoost forecast (Priority 2)")
    print("=" * 70 + "\n")

    val_csv = ROOT / "artefacts" / "predictions" / "xgboost_rmse.csv"
    test_csv = ROOT / "artefacts" / "predictions" / "test" / "xgboost_rmse.csv"

    val_sum, resid_val, val_df = diagnose_block("val", val_csv)
    test_sum, resid_test, test_df = diagnose_block("test", test_csv)

    for s in (val_sum, test_sum):
        print(f"--- {s['block']} block ({s['n']} days) ---")
        print(f"  mean = {s['mean']:+.3f}     std = {s['std']:.3f}")
        print(f"  min  = {s['min']:.2f}    max = {s['max']:.2f}")
        print(f"  skew = {s['skew']:+.3f}    kurtosis (excess) = {s['kurtosis_excess']:+.3f}")
        print(f"  Ljung-Box Q(20) = {s['Q']:.2f}    p-value = {s['p_value']:.4f}")
        if "shapiro_p" in s:
            print(f"  Shapiro-Wilk W = {s['shapiro_W']:.4f}    p = {s['shapiro_p']:.4f}")
        print()

    # Save summary
    out_csv = ROOT / "artefacts" / "metrics" / "residual_diagnostics.csv"
    pd.DataFrame([val_sum, test_sum]).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv.relative_to(ROOT)}")

    # Plot
    out_fig = ROOT / "artefacts" / "figures" / "fig_6_residuals_xgboost.png"
    plot_diagnostics(resid_val, resid_test, val_df.index, test_df.index, out_fig)

    # Interpretation hint for the chapter
    print("\n--- Interpretation hint for chap6 §6.4 ---")
    if test_sum["p_value"] < 0.05:
        print(f"  Ljung-Box p={test_sum['p_value']:.4f} < 0.05 => "
              f"residuals are NOT white noise -- some autocorrelation remains.")
    else:
        print(f"  Ljung-Box p={test_sum['p_value']:.4f} >= 0.05 => "
              f"cannot reject white-noise hypothesis (good).")
    if abs(test_sum["mean"]) < 0.5:
        print(f"  Mean residual {test_sum['mean']:+.2f} -- centred near zero (no bias).")
    else:
        print(f"  Mean residual {test_sum['mean']:+.2f} -- "
              f"some {'over' if test_sum['mean'] < 0 else 'under'}-prediction bias.")


if __name__ == "__main__":
    main()
