"""Chapter 7 INTEGRATED — before/after visualisations of the joint
forecast-to-decision pipeline (Eq 3.8 → 3.17 of Chapter 3).

Renders five figures from `artefacts/chapter7/integrated/integrated_summary.csv`:

  Figure 7.6  Total hospital cost per method (stacked: inventory + scheduling)
              with 95% CI bars
  Figure 7.7  Cost decomposition: holding / ordering / stockout / expiry
              / payroll / locum per method
  Figure 7.8  Coverage % per method
  Figure 7.9  Stockout incidence (% of days) per method
  Figure 7.10 Inventory cost vs scheduling cost (2D scatter, per method)

Output: artefacts/chapter7/integrated/figures/ (tight crop, NO embedded title).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artefacts" / "chapter7" / "integrated"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SAVE = dict(dpi=200, bbox_inches="tight", pad_inches=0.08)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", **SAVE)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  -> {stem}")


METHOD_COLOURS = {
    "Baseline":                                      "#7a0b22",
    "Forecast -> inventory only":                    "#1f6dbf",
    "Forecast -> scheduling only":                   "#d99054",
    "Forecast -> BOTH":                              "#8e44ad",
    "Alg 8 (MC grid S) + hist roster":               "#2c8a3a",
    "Alg 8 + Forecast roster (Eq 3.8-3.17 full chain)": "#0f8a85",
}


# ----------------------------------------------------------------------
# Figure 7.6 — total hospital cost per method (stacked: inv + sched)
# ----------------------------------------------------------------------
def fig_7_6(df: pd.DataFrame):
    print("Figure 7.6 — Total hospital cost (stacked: inventory + scheduling)")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    methods = df["method"].tolist()
    inv = df["inventory_cost_mean"].values / 1e6
    sch = df["scheduling_cost_mean"].values / 1e6
    err = df["total_ci_half"].values / 1e6

    ax.bar(methods, inv, color="#1f6dbf", edgecolor="white",
            linewidth=1.5, label="Inventory cost")
    ax.bar(methods, sch, bottom=inv, color="#c43d3d", edgecolor="white",
            linewidth=1.5, label="Scheduling cost", alpha=0.92)
    # Error bars on the total
    ax.errorbar(methods, inv + sch, yerr=err,
                  fmt="none", ecolor="#333", lw=1.5, capsize=5)

    base = df.loc[df["method"] == "Baseline", "total_cost_mean"].iloc[0]
    for i, (m, total) in enumerate(zip(methods, inv + sch)):
        pct = (base - df["total_cost_mean"].iloc[i]) / base * 100
        lab = f"R{total:.2f}M"
        if df["method"].iloc[i] != "Baseline":
            lab += f"\n({pct:+.1f}%)"
        ax.text(i, total + err[i] * 1.05, lab,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222")

    ax.set_ylabel("Total annual hospital cost (ZAR million)", fontsize=11)
    ax.set_ylim(0, (inv + sch).max() * 1.25)
    ax.tick_params(axis="x", labelsize=9, rotation=18)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    save(fig, "figure_7_6_total_hospital_cost")


# ----------------------------------------------------------------------
# Figure 7.7 — cost decomposition (6 components)
# ----------------------------------------------------------------------
def fig_7_7(df: pd.DataFrame):
    print("Figure 7.7 — Cost decomposition by component")
    comps = ["holding_mean", "ordering_mean", "stockout_mean",
              "expiry_mean", "payroll_mean", "locum_mean"]
    labs = ["Holding", "Ordering", "Stockout penalty",
             "Expiry", "Payroll", "Locum"]
    cols = ["#1f6dbf", "#2c8a3a", "#c43d3d",
             "#d99054", "#8e44ad", "#7a0b22"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    methods = df["method"].tolist()
    bottom = np.zeros(len(df))
    for comp, lab, col in zip(comps, labs, cols):
        vals = df[comp].values / 1e6
        ax.bar(methods, vals, bottom=bottom, color=col,
                edgecolor="white", linewidth=1.0, label=lab)
        bottom += vals
    ax.set_ylabel("Annual cost (ZAR million)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9, rotation=18)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_7_cost_decomposition")


# ----------------------------------------------------------------------
# Figure 7.8 — coverage % per method
# ----------------------------------------------------------------------
def fig_7_8(df: pd.DataFrame):
    print("Figure 7.8 — Lawful coverage % per method (headline)")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    methods = df["method"].tolist()
    vals = df["lawful_coverage_pct_mean"].values
    colours = [METHOD_COLOURS.get(m, "#666") for m in methods]
    bars = ax.bar(methods, vals, color=colours,
                   edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
    ax.axhline(96.9, color="#888", linestyle="--", lw=1.0,
                label="Actual (propped-up) coverage published by chapter7_simulation (96.9%)")
    ax.set_ylabel("Lawful coverage (% — 45h BCEA cap)", fontsize=11)
    ax.set_ylim(60, 105)
    ax.tick_params(axis="x", labelsize=9, rotation=18)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    save(fig, "figure_7_8_coverage")


# ----------------------------------------------------------------------
# Figure 7.9 — stockout incidence per method
# ----------------------------------------------------------------------
def fig_7_9(df: pd.DataFrame):
    print("Figure 7.9 — Stockout incidence")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    methods = df["method"].tolist()
    vals = df["stockout_incidence_pct_mean"].values
    colours = [METHOD_COLOURS.get(m, "#666") for m in methods]
    bars = ax.bar(methods, vals, color=colours,
                   edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
    ax.set_ylabel("Stockout incidence (% of days)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=9, rotation=18)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_9_stockout_incidence")


# ----------------------------------------------------------------------
# Figure 7.10 — inventory vs scheduling cost scatter
# ----------------------------------------------------------------------
def fig_7_10(df: pd.DataFrame):
    print("Figure 7.10 — Inventory vs scheduling cost (frontier)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in df.iterrows():
        col = METHOD_COLOURS.get(r["method"], "#666")
        ax.scatter(r["inventory_cost_mean"] / 1e6,
                    r["scheduling_cost_mean"] / 1e6,
                    s=180, color=col, edgecolor="white",
                    linewidth=1.5, zorder=3)
        ax.annotate(r["method"].replace(" -> ", "→"),
                     xy=(r["inventory_cost_mean"] / 1e6,
                          r["scheduling_cost_mean"] / 1e6),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=8, color="#222")
    ax.set_xlabel("Inventory cost (ZAR million)", fontsize=11)
    ax.set_ylabel("Scheduling cost (ZAR million)", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.spines["left"].set_color("#888")
    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_10_inv_vs_sched_frontier")


# ----------------------------------------------------------------------
# Figure 7.11 — LAWFUL vs ACTUAL coverage gap per method
# ----------------------------------------------------------------------
def fig_7_11_lawful_vs_actual(df: pd.DataFrame):
    print("Figure 7.11 — LAWFUL vs ACTUAL coverage gap")
    fig, ax = plt.subplots(figsize=(11, 5.0))
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.36
    lawful = df["lawful_coverage_pct_mean"].values
    actual = df["actual_coverage_pct_mean"].values
    bars_a = ax.bar(x - width / 2, actual, width,
                     color="#c43d3d", edgecolor="white", linewidth=1.5,
                     label="ACTUAL coverage (with overtime — propped-up)")
    bars_l = ax.bar(x + width / 2, lawful, width,
                     color="#2c8a3a", edgecolor="white", linewidth=1.5,
                     label="LAWFUL coverage (45h BCEA cap — headline)")
    for bar, v in zip(bars_a, actual):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=9, color="#7a0b22", fontweight="bold")
    for bar, v in zip(bars_l, lawful):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=9, color="#1d5d2a", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=18)
    ax.set_ylabel("Coverage (% of required nurse-hours)", fontsize=11)
    ax.set_ylim(50, 110)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    save(fig, "figure_7_11_lawful_vs_actual_coverage")


# ----------------------------------------------------------------------
# Figure 7.12 — Mean weekly hours per active nurse vs 45h BCEA cap
# ----------------------------------------------------------------------
def fig_7_12_overwork(df: pd.DataFrame):
    print("Figure 7.12 — Overwork (mean weekly hours vs 45h cap)")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    methods = df["method"].tolist()
    vals = df["mean_weekly_hours_mean"].values
    colours = [METHOD_COLOURS.get(m, "#666") for m in methods]
    bars = ax.bar(methods, vals, color=colours,
                   edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        pct = v / 45.0 * 100
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.1f}h\n({pct:.0f}% of legal)",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222")
    ax.axhline(45.0, color="#1d5d2a", linestyle="--", lw=1.5,
                label="BCEA s.9 lawful cap (45h)")
    ax.set_ylabel("Mean weekly hours per active nurse", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(axis="x", labelsize=9, rotation=18)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    save(fig, "figure_7_12_overwork_weekly_hours")


# ----------------------------------------------------------------------
# Figure 7.13 — Staffing shortfall + BCEA breaches per method
# ----------------------------------------------------------------------
def fig_7_13_shortfall(df: pd.DataFrame):
    print("Figure 7.13 — Staffing shortfall + BCEA breaches")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    methods = df["method"].tolist()
    colours = [METHOD_COLOURS.get(m, "#666") for m in methods]

    # Left — staffing shortfall in nurses
    shortfalls = df["staffing_shortfall_nurses_mean"].values
    bars = ax1.bar(methods, shortfalls, color=colours,
                    edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, shortfalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                  f"{v:.1f}", ha="center", va="bottom",
                  fontsize=10, fontweight="bold", color="#222")
    ax1.set_ylabel("Staffing shortfall (nurses)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=9, rotation=22)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.set_title("Staffing shortfall at lawful 45h cap\n"
                   "(needed - 23 active nurses)",
                   fontsize=10, color="#444", loc="left")
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color("#888")
    ax1.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax1.set_axisbelow(True)

    # Right — BCEA breaches per nurse per week
    breaches = df["bcea_breaches_per_nurse_wk_mean"].values
    bars = ax2.bar(methods, breaches, color=colours,
                    edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, breaches):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}", ha="center", va="bottom",
                  fontsize=10, fontweight="bold", color="#222")
    ax2.set_ylabel("BCEA breaches per active nurse per week", fontsize=11)
    ax2.tick_params(axis="x", labelsize=9, rotation=22)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.set_title("Weeks per nurse with weekly hours > 45 (BCEA breach)",
                   fontsize=10, color="#444", loc="left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color("#888")
    ax2.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax2.set_axisbelow(True)

    save(fig, "figure_7_13_shortfall_and_breaches")


def main():
    df = pd.read_csv(RES / "integrated_summary.csv")
    fig_7_6(df)
    fig_7_7(df)
    # NEW: lawful-vs-actual reframe takes the headline coverage slot
    fig_7_11_lawful_vs_actual(df)
    fig_7_12_overwork(df)
    fig_7_13_shortfall(df)
    # Keep prior figures
    fig_7_8(df)
    fig_7_9(df)
    fig_7_10(df)
    print()
    print(f"All figures written to: {OUT}")


if __name__ == "__main__":
    main()
