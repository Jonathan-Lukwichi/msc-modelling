"""Chapter 7 — app-aligned before/after visualisation.

Reads `artefacts/chapter7/app_aligned/app_aligned_summary.csv` and renders
five figures showing the larger cost-savings story under the app's
"naive baseline" framing.

  Figure 7.14  Annual total cost per method (stacked: inventory + scheduling)
  Figure 7.15  Saving vs the naive baseline (bar chart, R/year)
  Figure 7.16  Weekly cost (matches the app's KPI cards)
  Figure 7.17  Saving decomposition: supply vs staff
  Figure 7.18  Cost decomposition by component (6 components)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artefacts" / "chapter7" / "app_aligned"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SAVE = dict(dpi=200, bbox_inches="tight", pad_inches=0.08)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", **SAVE)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  -> {stem}")


METHOD_COL = {
    "Naive supply + Busy-day staffing":          "#7a0b22",
    "Naive supply + Average staffing":           "#c43d3d",
    "Textbook supply + Busy-day staffing":       "#d99054",
    "Textbook supply + Average staffing":        "#1f6dbf",
    "Forecast (s,S) + Average staffing":         "#2c8a3a",
    "Textbook supply + Forecast staffing":       "#8e44ad",
    "Forecast (s,S) + Forecast staffing (FULL APP)": "#0f8a85",
}


def fig_7_14_annual_stacked(df):
    print("Figure 7.14 — Annual total cost per method")
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = df["method"].tolist()
    inv = df["annual_inv"].values / 1e6
    sch = df["annual_sch"].values / 1e6
    err = df["annual_ci_half"].values / 1e6

    ax.bar(methods, inv, color="#1f6dbf", edgecolor="white",
            linewidth=1.2, label="Inventory cost")
    ax.bar(methods, sch, bottom=inv, color="#c43d3d", edgecolor="white",
            linewidth=1.2, label="Scheduling cost", alpha=0.92)
    ax.errorbar(methods, inv + sch, yerr=err, fmt="none",
                  ecolor="#333", lw=1.5, capsize=5)
    base = df.loc[df["method"] == "Naive supply + Busy-day staffing",
                    "annual_total"].iloc[0]
    for i, total in enumerate(inv + sch):
        pct = (base - df["annual_total"].iloc[i]) / base * 100
        lab = f"R{total:.1f}M"
        if df["method"].iloc[i] != "Naive supply + Busy-day staffing":
            lab += f"\n({pct:+.0f}%)"
        ax.text(i, total + err[i] * 1.08, lab, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222")
    ax.set_ylabel("Annual total cost (ZAR million)", fontsize=11)
    ax.set_ylim(0, (inv + sch).max() * 1.20)
    ax.tick_params(axis="x", labelsize=8, rotation=22)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    save(fig, "figure_7_14_annual_total_cost")


def fig_7_15_savings_vs_naive(df):
    print("Figure 7.15 — Saving vs naive baseline")
    df = df.sort_values("saving_vs_naive_zar", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    methods = df["method"].tolist()
    savings = df["saving_vs_naive_zar"].values / 1e6
    colours = [METHOD_COL.get(m, "#666") for m in methods]
    bars = ax.barh(methods, savings, color=colours, edgecolor="white",
                    linewidth=1.2)
    for bar, v, pct in zip(bars, savings,
                              df["saving_vs_naive_pct"].values):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                f"R{v:.2f}M ({pct:+.1f}%)",
                va="center", ha="left",
                fontsize=10, fontweight="bold", color="#222")
    ax.axvline(0, color="#555", lw=1.2)
    ax.set_xlabel("Annual saving vs naive baseline (ZAR million)",
                   fontsize=11)
    ax.set_xlim(min(savings) - 1, max(savings) * 1.30)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_15_savings_vs_naive")


def fig_7_16_weekly(df):
    print("Figure 7.16 — Weekly cost per method (matches app KPI cards)")
    fig, ax = plt.subplots(figsize=(12, 5))
    methods = df["method"].tolist()
    weekly = df["weekly_total"].values / 1000
    colours = [METHOD_COL.get(m, "#666") for m in methods]
    bars = ax.bar(methods, weekly, color=colours, edgecolor="white",
                   linewidth=1.2)
    for bar, v in zip(bars, weekly):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8,
                f"R{v:.0f}k", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
    ax.set_ylabel("Weekly total cost (ZAR thousand)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=22)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_16_weekly_cost")


def fig_7_17_supply_vs_staff_saving(df):
    print("Figure 7.17 — Supply vs staff saving decomposition")
    base_row = df[df["method"] == "Naive supply + Busy-day staffing"].iloc[0]
    base_inv = base_row["annual_inv"]
    base_sch = base_row["annual_sch"]
    methods = df["method"].tolist()
    inv_saved = (base_inv - df["annual_inv"].values) / 1e6
    sch_saved = (base_sch - df["annual_sch"].values) / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods))
    w = 0.38
    bars_i = ax.bar(x - w / 2, inv_saved, w, color="#1f6dbf",
                     edgecolor="white", linewidth=1.2,
                     label="Supply (inventory) saving")
    bars_s = ax.bar(x + w / 2, sch_saved, w, color="#c43d3d",
                     edgecolor="white", linewidth=1.2,
                     label="Staff (scheduling) saving")
    for bar, v in zip(bars_i, inv_saved):
        if abs(v) > 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                     v + (0.15 if v >= 0 else -0.4),
                     f"R{v:.1f}M", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color="#1d4d7a")
    for bar, v in zip(bars_s, sch_saved):
        if abs(v) > 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                     v + (0.15 if v >= 0 else -0.4),
                     f"R{v:.1f}M", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color="#7a0b22")
    ax.axhline(0, color="#555", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8, rotation=22)
    ax.set_ylabel("Annual saving vs naive baseline (ZAR million)",
                   fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    save(fig, "figure_7_17_supply_vs_staff_saving")


def fig_7_18_decomposition(df):
    print("Figure 7.18 — Cost decomposition by 6 components")
    methods = df["method"].tolist()
    n_weeks = df["n_weeks_mean"].values
    comps_keys = ["holding_mean", "ordering_mean", "stockout_mean",
                   "expiry_mean", "payroll_mean", "locum_mean"]
    labs = ["Holding", "Ordering", "Stockout",
             "Expiry", "Payroll", "Locum"]
    cols = ["#1f6dbf", "#2c8a3a", "#c43d3d",
             "#d99054", "#8e44ad", "#7a0b22"]
    annual = np.zeros((len(comps_keys), len(df)))
    for j, k in enumerate(comps_keys):
        annual[j] = df[k].values / n_weeks * 52 / 1e6
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(df))
    for j, (lab, col) in enumerate(zip(labs, cols)):
        ax.bar(methods, annual[j], bottom=bottom, color=col,
                edgecolor="white", linewidth=1.0, label=lab)
        bottom += annual[j]
    ax.set_ylabel("Annual cost (ZAR million)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=22)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_18_cost_decomposition")


def main():
    df = pd.read_csv(RES / "app_aligned_summary.csv")
    fig_7_14_annual_stacked(df)
    fig_7_15_savings_vs_naive(df)
    fig_7_16_weekly(df)
    fig_7_17_supply_vs_staff_saving(df)
    fig_7_18_decomposition(df)
    print()
    print(f"All figures written to: {OUT}")


if __name__ == "__main__":
    main()
