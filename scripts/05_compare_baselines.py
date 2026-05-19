"""Plan §15 Step 10 (interim, baselines only): consolidated leaderboard.

Reads every per-model metrics file written so far, assembles a long-form
leaderboard and a publication wide-form table, and plots Figure 6.4
(ARIMA vs SARIMAX vs NB GLM on val).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits

# Design language per CHAPTER_6_PLAN.md §5
NAVY, TEAL, GREEN, AMBER, ROSE = "#1e6091", "#0d9488", "#16a34a", "#d97706", "#dc2626"
NEUTRAL, LIGHT, DARK = "#475569", "#e5e7eb", "#0f172a"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def assemble_leaderboard() -> pd.DataFrame:
    """Stitch every artefacts/metrics/*_metrics.csv into one long table."""
    metrics_dir = ROOT / "artefacts" / "metrics"
    rows = []

    # Reference floor has multiple baselines per block
    ref = metrics_dir / "reference_floor.csv"
    if ref.exists():
        df = pd.read_csv(ref)
        for _, r in df.iterrows():
            rows.append({
                "model": r["baseline"], "family": "naive",
                "block": r["block"],
                "MAPE": r["MAPE"], "MAE": r["MAE"],
                "RMSE": r["RMSE"], "R2": r["R2"],
            })

    # Single-model metrics files
    for model_name, family in (("arima", "classical"),
                                ("sarimax", "classical"),
                                ("nbglm", "parametric_glm")):
        f = metrics_dir / f"{model_name}_metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append({
                "model": model_name, "family": family,
                "block": r["block"],
                "MAPE": r["MAPE"], "MAE": r["MAE"],
                "RMSE": r["RMSE"], "R2": r["R2"],
            })

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values(["block", "MAPE"]).reset_index(drop=True)
    return leaderboard


def plot_figure_6_4(out_path: Path) -> None:
    """ARIMA vs SARIMAX vs NB GLM on val: forecast plot + metric bars."""
    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    val = splits.slice(g1, "val")["total_daily_arrivals"]

    pred_dir = ROOT / "artefacts" / "predictions"
    arima = (pd.read_csv(pred_dir / "arima.csv", parse_dates=["date"])
             .set_index("date") if (pred_dir / "arima.csv").exists() else None)
    sarimax = (pd.read_csv(pred_dir / "sarimax.csv", parse_dates=["date"])
               .set_index("date") if (pred_dir / "sarimax.csv").exists() else None)
    nbglm = (pd.read_csv(pred_dir / "nbglm.csv", parse_dates=["date"])
             .set_index("date") if (pred_dir / "nbglm.csv").exists() else None)

    if not all([arima is not None, sarimax is not None, nbglm is not None]):
        missing = [n for n, x in [("arima", arima), ("sarimax", sarimax),
                                   ("nbglm", nbglm)] if x is None]
        print(f"  Skipping figure 6.4 — missing predictions for: {missing}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5),
                              gridspec_kw={"height_ratios": [3, 2]})

    # Row 1: forecast plot
    ax = axes[0]
    ax.plot(val.index, val.values, color=NEUTRAL, linewidth=1.5,
            label="Actual", zorder=4)
    ax.plot(arima.index, arima["predicted"], color=NAVY, linewidth=1.2,
            label="ARIMA", linestyle="--", zorder=2)
    ax.plot(sarimax.index, sarimax["predicted"], color=TEAL, linewidth=1.5,
            label="SARIMAX", zorder=3)
    ax.plot(nbglm.index, nbglm["predicted"], color=AMBER, linewidth=1.2,
            label="NB GLM", linestyle=":", zorder=2)
    # NB GLM 95% PI
    ax.fill_between(nbglm.index, nbglm["lower_95"], nbglm["upper_95"],
                     color=AMBER, alpha=0.10, label="NB GLM 95% PI", zorder=1)
    ax.set_ylabel("Daily ED arrivals")
    ax.set_title("Figure 6.4 — ARIMA vs SARIMAX vs NB GLM on validation block "
                 "(2024-07-01 to 2024-12-31)", loc="left", fontsize=12)
    ax.legend(loc="upper left", frameon=False, ncol=5)

    # Row 2: metric bars
    ax = axes[1]
    leaderboard = assemble_leaderboard()
    val_lb = leaderboard[leaderboard["block"] == "val"].copy()
    plot_models = ["arima", "sarimax", "nbglm"]
    val_lb = val_lb[val_lb["model"].isin(plot_models)].set_index("model").loc[plot_models]

    metrics_to_plot = ["MAPE", "MAE", "RMSE"]
    x = np.arange(len(plot_models))
    width = 0.27
    palette = [NAVY, TEAL, AMBER]
    for i, m in enumerate(metrics_to_plot):
        vals = val_lb[m].values
        bars = ax.bar(x + (i - 1) * width, vals, width, color=palette[i],
                      label=m, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["ARIMA", "SARIMAX", "NB GLM"])
    ax.set_ylabel("Metric value (MAPE %, MAE / RMSE patients)")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.set_title("Validation metrics", loc="left", fontsize=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Wrote: {out_path.relative_to(ROOT)}")


def main() -> None:
    leaderboard = assemble_leaderboard()
    print("Leaderboard (all baselines, val + test):")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(leaderboard.to_string(index=False))

    out_long = ROOT / "artefacts" / "tables" / "leaderboard_baselines.csv"
    out_long.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_long, index=False)
    print(f"\nWrote: {out_long.relative_to(ROOT)}")

    # Wide-form publication table
    val_lb = leaderboard[leaderboard["block"] == "val"].set_index("model")
    test_lb = leaderboard[leaderboard["block"] == "test"].set_index("model")
    wide_cols = []
    if not val_lb.empty:
        wide_cols.append(val_lb[["MAPE", "MAE"]].rename(
            columns={"MAPE": "val_MAPE", "MAE": "val_MAE"}))
    if not test_lb.empty:
        wide_cols.append(test_lb[["MAPE", "MAE"]].rename(
            columns={"MAPE": "test_MAPE", "MAE": "test_MAE"}))
    wide = pd.concat(wide_cols, axis=1)
    out_wide = ROOT / "artefacts" / "tables" / "table_6_2_baselines.csv"
    wide.to_csv(out_wide)
    print(f"Wrote: {out_wide.relative_to(ROOT)}")

    plot_figure_6_4(ROOT / "artefacts" / "figures" / "fig_6_4_baselines_val.png")


if __name__ == "__main__":
    main()
