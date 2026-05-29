"""Final consolidated report — combines Phase 1, 2, 3 results into one master.

Reads from:
  artefacts/phase1_defaults/summary_phase1.csv
  artefacts/phase2_hpo/summary_phase2.csv
  artefacts/phase3_ablation/summary_phase3.csv

Produces:
  artefacts/final_report/
    master_comparison.csv           defaults vs HPO winner for every model
    headline_table.md               thesis-ready markdown table
    figures/cross_model_bars.png    val MAPE & RMSE bar chart, all models
    figures/defaults_vs_hpo.png     side-by-side defaults vs HPO winner
    figures/horizons.png            daily / weekly / monthly / yearly errors
    figures/ablation_summary.png    Phase 3 feature-ablation summary
    figures/best_model_pred_actual.png   showcase plot for the winner
    final_report.md                 narrative + every table + figure refs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artefacts"
OUT = ART / "final_report"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Load all available summary CSVs
# -------------------------------------------------------------------------
def load_summaries():
    summaries = {}
    p1 = ART / "phase1_defaults" / "summary_phase1.csv"
    p2 = ART / "phase2_hpo"      / "summary_phase2.csv"
    p3 = ART / "phase3_ablation" / "summary_phase3.csv"
    summaries["phase1"] = pd.read_csv(p1) if p1.exists() else None
    summaries["phase2"] = pd.read_csv(p2) if p2.exists() else None
    summaries["phase3"] = pd.read_csv(p3) if p3.exists() else None
    return summaries


# -------------------------------------------------------------------------
# Build master comparison: defaults (phase 1) vs HPO winner (phase 2)
# -------------------------------------------------------------------------
def build_master(summaries: dict) -> pd.DataFrame:
    rows = []
    p1 = summaries["phase1"]
    p2 = summaries["phase2"]
    if p1 is None:
        return pd.DataFrame()
    for _, r in p1.iterrows():
        model = r["model"]
        base = {
            "model": model,
            "defaults_cv_RMSE":  r.get("cv_avg_RMSE", np.nan),
            "defaults_cv_MAPE":  r.get("cv_avg_MAPE", np.nan),
            "defaults_val_MAPE": r.get("val_MAPE", np.nan),
            "defaults_val_RMSE": r.get("val_RMSE", np.nan),
            "defaults_val_R2":   r.get("val_R2", np.nan),
            "defaults_yearly_pct": r.get("yearly_avg_pct_error", np.nan),
        }
        # Find the best HPO winner for this model (across algorithms)
        if p2 is not None:
            sub = p2[p2["model"] == model]
            if not sub.empty:
                winner = sub.loc[sub["val_RMSE"].idxmin()]
                base.update({
                    "hpo_algo":     winner["algo"],
                    "hpo_cv_RMSE":  winner["cv_RMSE"],
                    "hpo_cv_MAPE":  winner["cv_MAPE"],
                    "hpo_val_MAPE": winner["val_MAPE"],
                    "hpo_val_RMSE": winner["val_RMSE"],
                    "hpo_val_R2":   winner["val_R2"],
                    "delta_MAPE_pp": winner["val_MAPE"] - base["defaults_val_MAPE"],
                    "delta_RMSE":    winner["val_RMSE"] - base["defaults_val_RMSE"],
                })
        rows.append(base)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------
def plot_cross_model_bars(master: pd.DataFrame):
    df = master.sort_values("defaults_val_RMSE").copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    width = 0.35

    axes[0].bar(x - width/2, df["defaults_val_MAPE"], width,
                label="Defaults", color="#888")
    if "hpo_val_MAPE" in df.columns and df["hpo_val_MAPE"].notna().any():
        axes[0].bar(x + width/2, df["hpo_val_MAPE"], width,
                    label="HPO winner", color="#1f77b4")
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["model"],
                                                    rotation=45, ha="right")
    axes[0].set_ylabel("val MAPE (%)")
    axes[0].set_title("val MAPE — defaults vs HPO winner")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend()

    axes[1].bar(x - width/2, df["defaults_val_RMSE"], width,
                label="Defaults", color="#888")
    if "hpo_val_RMSE" in df.columns and df["hpo_val_RMSE"].notna().any():
        axes[1].bar(x + width/2, df["hpo_val_RMSE"], width,
                    label="HPO winner", color="#ff7f0e")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["model"],
                                                    rotation=45, ha="right")
    axes[1].set_ylabel("val RMSE")
    axes[1].set_title("val RMSE — defaults vs HPO winner")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "defaults_vs_hpo.png", dpi=120)
    plt.close(fig)


def plot_horizons(summaries: dict):
    p1 = summaries["phase1"]
    if p1 is None: return
    df = p1.sort_values("val_MAPE").copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    width = 0.2
    for i, (col, label, col_colour) in enumerate([
        ("val_MAPE", "Daily MAPE (%)", "#1f77b4"),
        ("weekly_avg_pct_error", "Weekly avg err (%)", "#2ca02c"),
        ("monthly_avg_pct_error", "Monthly avg err (%)", "#ff7f0e"),
        ("yearly_avg_pct_error", "Yearly avg err (%)", "#d62728"),
    ]):
        if col not in df.columns:
            continue
        ax.bar(x + (i - 1.5) * width, df[col], width, label=label,
               color=col_colour)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylabel("% error")
    ax.set_title("Forecast accuracy across horizons (daily → yearly)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "horizons.png", dpi=120)
    plt.close(fig)


def plot_best_pred_actual(summaries: dict):
    """Predicted-vs-actual line plot for the best-RMSE model."""
    candidate = None
    if summaries["phase2"] is not None and not summaries["phase2"].empty:
        df = summaries["phase2"]
        winner = df.loc[df["val_RMSE"].idxmin()]
        candidate = (winner["model"], winner["algo"],
                     ART / "phase2_hpo" /
                     f"val_preds_{winner['model']}_{winner['algo']}.csv")
    if candidate is None or not candidate[2].exists():
        # Fall back to Phase 1 daily
        if summaries["phase1"] is None or summaries["phase1"].empty:
            return
        df = summaries["phase1"]
        winner = df.loc[df["val_RMSE"].idxmin()]
        candidate = (winner["model"], "defaults",
                     ART / "phase1_defaults" / f"daily_{winner['model']}.csv")
    model, algo, path = candidate
    if not path.exists():
        return
    pred = pd.read_csv(path)
    pred = pred[pred.iloc[:, 0] != "AVG"].copy() if "date" in pred.columns else pred
    pred["date"] = pd.to_datetime(pred["date"])
    pred["actual"] = pred["actual"].astype(float)
    pred["predicted"] = pred["predicted"].astype(float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred["date"], pred["actual"], lw=1.4, color="#333",
            label="Actual arrivals", alpha=0.9)
    ax.plot(pred["date"], pred["predicted"], lw=1.4, color="#d62728",
            label=f"Predicted ({model} / {algo})", alpha=0.85)
    mape = (abs(pred["actual"] - pred["predicted"]) / pred["actual"]).mean() * 100
    rmse = ((pred["actual"] - pred["predicted"]) ** 2).mean() ** 0.5
    ax.set_title(f"Best model: {model} ({algo})  —  val MAPE {mape:.2f}%  RMSE {rmse:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily ED arrivals")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "best_model_pred_actual.png", dpi=120)
    plt.close(fig)


def plot_ablation_summary(summaries: dict):
    p3 = summaries["phase3"]
    if p3 is None or p3.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    cfgs = ["A_raw_only", "B_engineered", "C_selection", "D_both"]
    models = list(p3["model"].unique())
    x = np.arange(len(cfgs))
    width = 0.35
    for i, model in enumerate(models[:2]):
        sub = p3[p3["model"] == model].set_index("config").reindex(cfgs)
        ax.bar(x + (i - 0.5) * width, sub["val_MAPE"], width,
               label=f"{model} val MAPE")
        for j, v in enumerate(sub["val_MAPE"]):
            if pd.notna(v):
                ax.text(x[j] + (i - 0.5) * width, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["A: raw only", "B: FE only", "C: FS only", "D: both"])
    ax.set_ylabel("val MAPE (%)")
    ax.set_title("Phase 3 feature ablation — val MAPE per config (top-2 models)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ablation_summary.png", dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------------
# Markdown report
# -------------------------------------------------------------------------
def build_report(master: pd.DataFrame, summaries: dict):
    md = []
    md.append("# Master Model Comparison Report")
    md.append("")
    md.append("This report consolidates Phases 1, 2, and 3 of the Task-1 "
              "(daily ED arrivals) forecasting experiment.")
    md.append("")

    md.append("## Phase 1 — Chapter 5 defaults (no HPO)")
    md.append("")
    if summaries["phase1"] is not None:
        md.append(summaries["phase1"].to_markdown(index=False, floatfmt=".3f"))
    md.append("")

    md.append("## Phase 2 — HPO targeting minimum RMSE")
    md.append("")
    if summaries["phase2"] is not None:
        md.append(summaries["phase2"].to_markdown(index=False, floatfmt=".3f"))
    else:
        md.append("_Phase 2 in progress._")
    md.append("")

    md.append("## Master comparison — defaults vs HPO winner")
    md.append("")
    md.append(master.to_markdown(index=False, floatfmt=".3f"))
    md.append("")

    md.append("## Phase 3 — Feature ablation (top-2 models)")
    md.append("")
    if summaries["phase3"] is not None:
        md.append(summaries["phase3"].to_markdown(index=False, floatfmt=".3f"))
    else:
        md.append("_Phase 3 in progress._")
    md.append("")

    md.append("## Figures")
    md.append("")
    md.append("- ![Defaults vs HPO](figures/defaults_vs_hpo.png)")
    md.append("- ![Horizons](figures/horizons.png)")
    md.append("- ![Best model predicted vs actual](figures/best_model_pred_actual.png)")
    md.append("- ![Ablation summary](figures/ablation_summary.png)")
    md.append("")

    (OUT / "final_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {OUT/'final_report.md'}")


def main():
    summaries = load_summaries()
    master = build_master(summaries)
    master.to_csv(OUT / "master_comparison.csv", index=False)
    print(f"Wrote: {OUT/'master_comparison.csv'}")

    plot_cross_model_bars(master)
    plot_horizons(summaries)
    plot_best_pred_actual(summaries)
    plot_ablation_summary(summaries)
    build_report(master, summaries)

    print("\nDone. Open:")
    print(f"  {OUT/'final_report.md'}")
    print(f"  {FIG}")


if __name__ == "__main__":
    main()
