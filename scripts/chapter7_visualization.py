"""Chapter 7 — Before/after visualisation of the (s, S) optimisation
experiment.

Reads `artefacts/chapter7/results/*.csv` produced by
`chapter7_optimization_experiment.py` and renders five figures:

  Figure 7.1  Total annual cost per method, with 95% CI bars
  Figure 7.2  Cost decomposition (holding / ordering / stockout / expiry)
  Figure 7.3  Optimiser convergence curves (best-so-far vs trial)
  Figure 7.4  Grid heatmap of cost in (alpha, beta) slice at best gamma
  Figure 7.5  Stockout incidence per method

All figures are written tight-cropped, with NO embedded titles or
footers (titles go in LaTeX \\caption{...}), to
artefacts/chapter7/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artefacts" / "chapter7" / "results"
OUT = ROOT / "artefacts" / "chapter7" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SAVE = dict(dpi=200, bbox_inches="tight", pad_inches=0.08)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", **SAVE)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  -> {stem}")


# Colour palette consistent across all figures
METHOD_COLOURS = {
    "Baseline":         "#7a0b22",   # red — the floor
    "Forecast-driven":  "#1f6dbf",   # blue
    "Grid Search":      "#d99054",   # amber
    "Random Search":    "#8e44ad",   # purple
    "Bayesian (Optuna)":"#2c8a3a",   # green
    "Forecast + Optuna":"#0f8a85",   # teal — headline winner
}


# ----------------------------------------------------------------------
# Figure 7.1  Total cost per method with 95% CI bars
# ----------------------------------------------------------------------
def fig_7_1_total_cost(summary: pd.DataFrame):
    print("Figure 7.1 — Total annual cost per method")
    df = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = [METHOD_COLOURS.get(m, "#888") for m in df["method"]]
    bars = ax.bar(df["method"], df["total_cost_mean"] / 1e6,
                   yerr=df["cost_ci_half"] / 1e6,
                   color=colours, edgecolor="white", linewidth=1.5,
                   capsize=5, error_kw=dict(ecolor="#333", lw=1.5))
    base_cost = df.loc[df["method"] == "Baseline",
                        "total_cost_mean"].iloc[0]
    for bar, (_, r) in zip(bars, df.iterrows()):
        h = bar.get_height()
        pct = (base_cost - r["total_cost_mean"]) / base_cost * 100
        label = f"R{h:.2f}M"
        if r["method"] != "Baseline":
            label += f"\n({pct:+.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.04,
                label, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
    ax.set_ylabel("Total annual cost (ZAR million)", fontsize=11)
    ax.set_ylim(0, df["total_cost_mean"].max() / 1e6 * 1.25)
    ax.tick_params(axis="x", labelsize=10, rotation=15)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_1_total_cost")


# ----------------------------------------------------------------------
# Figure 7.2  Cost decomposition stacked bars
# ----------------------------------------------------------------------
def fig_7_2_cost_decomposition(decomp: pd.DataFrame):
    print("Figure 7.2 — Cost decomposition by component")
    df = decomp.copy()
    components = ["holding_mean", "ordering_mean",
                   "stockout_mean", "expiry_mean"]
    labels = ["Holding", "Ordering", "Stockout penalty", "Expiry"]
    colours = ["#1f6dbf", "#2c8a3a", "#c43d3d", "#d99054"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(df))
    for comp, lab, col in zip(components, labels, colours):
        vals = df[comp].values / 1e6
        ax.bar(df["method"], vals, bottom=bottom,
                color=col, label=lab, edgecolor="white", linewidth=1.0)
        bottom += vals
    ax.set_ylabel("Annual cost (ZAR million)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10, rotation=15)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_2_cost_decomposition")


# ----------------------------------------------------------------------
# Figure 7.3  Convergence curves (random + Optuna)
# ----------------------------------------------------------------------
def fig_7_3_convergence(random_trace: pd.DataFrame,
                          optuna_trace: pd.DataFrame,
                          baseline_cost: float):
    print("Figure 7.3 — Optimiser convergence")
    fig, ax = plt.subplots(figsize=(10, 5))

    def best_so_far(df):
        return df["total_cost_mean"].cummin().values / 1e6

    rt = random_trace.sort_values("trial").reset_index(drop=True)
    ot = optuna_trace.sort_values("trial").reset_index(drop=True)

    ax.plot(rt["trial"] + 1, best_so_far(rt), lw=2.0,
            color=METHOD_COLOURS["Random Search"],
            marker="o", markersize=5, label="Random Search")
    ax.plot(ot["trial"] + 1, best_so_far(ot), lw=2.0,
            color=METHOD_COLOURS["Bayesian (Optuna)"],
            marker="s", markersize=5, label="Bayesian (Optuna)")
    ax.axhline(baseline_cost / 1e6, color=METHOD_COLOURS["Baseline"],
                linestyle="--", lw=1.5, label="Baseline (no optimisation)")

    ax.set_xlabel("Trial number", fontsize=11)
    ax.set_ylabel("Best-so-far cost (ZAR million)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.tick_params(axis="both", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.spines["left"].set_color("#888")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    save(fig, "figure_7_3_convergence")


# ----------------------------------------------------------------------
# Figure 7.4  Grid heatmap (alpha × beta at best gamma)
# ----------------------------------------------------------------------
def fig_7_4_grid_heatmap(grid_trace: pd.DataFrame):
    print("Figure 7.4 — Grid-search heatmap")
    best_g = grid_trace.loc[grid_trace["total_cost_mean"].idxmin(),
                              "gamma"]
    sub = grid_trace[grid_trace["gamma"] == best_g].copy()
    sub = sub.sort_values(["alpha", "beta"])
    alphas = sorted(sub["alpha"].unique())
    betas = sorted(sub["beta"].unique())
    M = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            row = sub[(sub["alpha"] == a) & (sub["beta"] == b)]
            if not row.empty:
                M[i, j] = row["total_cost_mean"].iloc[0] / 1e6

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd",
                    origin="lower")
    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(j, i, f"{M[i, j]:.2f}M",
                    ha="center", va="center",
                    color="#222", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:.2f}" for b in betas], fontsize=10)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas], fontsize=10)
    ax.set_xlabel(r"$\beta$ (safety-stock multiplier)", fontsize=11)
    ax.set_ylabel(r"$\alpha$ (cycle-stock multiplier)", fontsize=11)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Total annual cost (ZAR M)", fontsize=10)
    # Outline winner
    min_idx = np.unravel_index(np.argmin(M), M.shape)
    rect = plt.Rectangle((min_idx[1] - 0.5, min_idx[0] - 0.5), 1, 1,
                          fill=False, edgecolor="#1d5d2a", linewidth=2.5)
    ax.add_patch(rect)
    save(fig, "figure_7_4_grid_heatmap")


# ----------------------------------------------------------------------
# Figure 7.5  Stockout incidence per method
# ----------------------------------------------------------------------
def fig_7_5_stockout(summary: pd.DataFrame):
    print("Figure 7.5 — Stockout incidence per method")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colours = [METHOD_COLOURS.get(m, "#888")
                for m in summary["method"]]
    bars = ax.bar(summary["method"],
                   summary["stockout_incidence_pct_mean"],
                   color=colours, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, summary["stockout_incidence_pct_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
    ax.set_ylabel("Stockout incidence (% of days)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=10, rotation=15)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "figure_7_5_stockout_incidence")


def main():
    summary = pd.read_csv(RES / "method_summary.csv")
    decomp  = pd.read_csv(RES / "cost_decomposition.csv")
    grid    = pd.read_csv(RES / "grid_trials.csv")
    rand    = pd.read_csv(RES / "random_trials.csv")
    opt     = pd.read_csv(RES / "optuna_trials.csv")
    base    = summary.loc[summary["method"] == "Baseline",
                            "total_cost_mean"].iloc[0]

    fig_7_1_total_cost(summary)
    fig_7_2_cost_decomposition(decomp)
    fig_7_3_convergence(rand, opt, base)
    fig_7_4_grid_heatmap(grid)
    fig_7_5_stockout(summary)
    print()
    print(f"All figures written to {OUT}")


if __name__ == "__main__":
    main()
