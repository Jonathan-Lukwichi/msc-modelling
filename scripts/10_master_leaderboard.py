"""Plan §15 Step 10 (full Task 1): consolidated leaderboard + master figures.

Reads every per-model metrics CSV in artefacts/metrics/ and produces:
  - artefacts/tables/leaderboard_task1.csv (long form)
  - artefacts/tables/table_6_5_task1_publication.csv (wide form, Susnjak-style)
  - artefacts/figures/fig_6_4_forecast_panel.png       (top-3 val forecasts)
  - artefacts/figures/fig_6_5_ranked_mape.png          (horizontal bar by MAPE)
  - artefacts/figures/fig_6_6_xgb_shap.png             (Susnjak Fig 9 layout)
  - artefacts/figures/fig_6_7_stl_decomposition.png    (Yu 2017 style)
  - artefacts/figures/fig_6_11_consolidated_panel.png  (multi-panel summary)
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


# Design language (plan §5)
NAVY, TEAL, GREEN, AMBER, ROSE = "#1e6091", "#0d9488", "#16a34a", "#d97706", "#dc2626"
NEUTRAL, LIGHT, DARK = "#475569", "#e5e7eb", "#0f172a"

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


FAMILY_COLOURS = {
    "naive": NEUTRAL,
    "classical": NAVY,
    "parametric_glm": AMBER,
    "ml": TEAL,
    "dl": "#6366f1",
    "hybrid_residual": ROSE,
    "hybrid_stl": "#a855f7",
}


MODEL_FAMILY = {
    "naive_yest": "naive", "naive_seasonal": "naive", "dow_mean": "naive",
    "arima": "classical", "sarimax": "classical",
    "nbglm": "parametric_glm",
    "xgboost": "ml",
    "ann": "dl", "lstm": "dl",
    "hybrid_sarimax_xgb": "hybrid_residual",
    "hybrid_sarimax_lstm": "hybrid_residual",
    "hybrid_lstm_xgb": "hybrid_residual",
    "hybrid_stl_xgb": "hybrid_stl",
    "hybrid_stl_ann": "hybrid_stl",
    "hybrid_stl_lstm": "hybrid_stl",
}


PRETTY_NAME = {
    "naive_yest": "Naïve (y_{t-1})",
    "naive_seasonal": "Naïve seasonal (y_{t-7})",
    "dow_mean": "DoW mean",
    "arima": "ARIMA",
    "sarimax": "SARIMAX",
    "nbglm": "NB GLM",
    "xgboost": "XGBoost",
    "ann": "ANN (MLP)",
    "lstm": "LSTM",
    "hybrid_sarimax_xgb": "SARIMAX + XGBoost",
    "hybrid_sarimax_lstm": "SARIMAX + LSTM",
    "hybrid_lstm_xgb": "LSTM + XGBoost",
    "hybrid_stl_xgb": "STL + XGBoost",
    "hybrid_stl_ann": "STL + ANN",
    "hybrid_stl_lstm": "STL + LSTM",
}


def _load_metrics(model_key: str) -> pd.DataFrame | None:
    # File name patterns: {key}_metrics.csv except hybrids use hybrid_{key}_metrics.csv
    if model_key.startswith("hybrid_"):
        p = ROOT / "artefacts" / "metrics" / f"{model_key}_metrics.csv"
    else:
        p = ROOT / "artefacts" / "metrics" / f"{model_key}_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def assemble_leaderboard() -> pd.DataFrame:
    rows = []
    # Reference floor (3 baselines in one csv)
    ref = ROOT / "artefacts" / "metrics" / "reference_floor.csv"
    if ref.exists():
        df = pd.read_csv(ref)
        for _, r in df.iterrows():
            rows.append({
                "model": r["baseline"], "family": "naive", "block": r["block"],
                "MAPE": r["MAPE"], "MAE": r["MAE"],
                "RMSE": r["RMSE"], "R2": r["R2"],
            })
    # Single-model metrics
    for key, family in MODEL_FAMILY.items():
        if key in ("naive_yest", "naive_seasonal", "dow_mean"):
            continue
        df = _load_metrics(key)
        if df is None:
            continue
        for _, r in df.iterrows():
            rows.append({
                "model": key, "family": family, "block": r["block"],
                "MAPE": r["MAPE"], "MAE": r["MAE"],
                "RMSE": r["RMSE"], "R2": r["R2"],
            })
    lb = pd.DataFrame(rows)
    lb["pretty"] = lb["model"].map(lambda m: PRETTY_NAME.get(m, m))
    lb = lb.sort_values(["block", "MAPE"]).reset_index(drop=True)
    return lb


def figure_ranked_mape(lb: pd.DataFrame, out: Path) -> None:
    val = lb[lb["block"] == "val"].copy()
    if val.empty:
        return
    val = val.sort_values("MAPE", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(val) + 1.5))
    colours = [FAMILY_COLOURS.get(f, NEUTRAL) for f in val["family"]]
    bars = ax.barh(val["pretty"], val["MAPE"], color=colours,
                    edgecolor="white", linewidth=0.5)
    for bar, v, f in zip(bars, val["MAPE"], val["family"]):
        ax.text(v * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=9,
                color=DARK)
    # Susnjak-style benchmark line: "MAPE < 10% is excellent" (Susnjak & Maddigan 2023)
    ax.axvline(10, color=GREEN, linestyle="--", alpha=0.6, linewidth=1)
    ax.text(10, len(val) - 0.5,
            " 10%: excellent threshold\n  (Susnjak & Maddigan 2023)",
            color=GREEN, fontsize=8, va="top")
    # Family legend
    families_in_plot = list(dict.fromkeys(val["family"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOURS[f])
               for f in families_in_plot]
    labels = [f.replace("_", " ") for f in families_in_plot]
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=9)
    ax.set_xlabel("Validation MAPE (%)")
    ax.set_title("Figure 6.5 — Task 1 model ranking by validation MAPE",
                  loc="left", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")


def figure_forecast_panel_top3(lb: pd.DataFrame, out: Path) -> None:
    val_lb = lb[lb["block"] == "val"].sort_values("MAPE").head(3)
    if len(val_lb) < 1:
        return
    splits = Splits.from_config()
    g1 = load_g1()
    val_actual = splits.slice(g1, "val")["total_daily_arrivals"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(val_actual.index, val_actual.values, color=NEUTRAL, linewidth=1.4,
            label="Actual", zorder=5)
    palette = [TEAL, NAVY, AMBER]
    for (_, row), c in zip(val_lb.iterrows(), palette):
        m = row["model"]
        if m in ("naive_yest", "naive_seasonal", "dow_mean"):
            # Reference-floor preds are in reference_floor.csv
            ref = pd.read_csv(ROOT / "artefacts" / "predictions" / "reference_floor.csv",
                               parse_dates=["date"])
            sel = ref[(ref["baseline"] == m) & (ref["block"] == "val")]
            x = sel["date"]; y = sel["predicted"]
        else:
            pref = "hybrid_" if m.startswith("hybrid_") and not m.startswith("hybrid_") else ""
            p = ROOT / "artefacts" / "predictions" / f"{m}.csv"
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            x = df.index; y = df["predicted"]
        ax.plot(x, y, color=c, linewidth=1.2, alpha=0.9,
                label=f"{row['pretty']} (val MAPE {row['MAPE']:.2f}%)",
                linestyle="--")
    ax.set_ylabel("Daily ED arrivals")
    ax.set_title("Figure 6.4 — Top-3 models on validation block "
                 "(actual = grey; forecasts = coloured)",
                 loc="left", fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")


def figure_xgb_shap(out: Path) -> None:
    p = ROOT / "artefacts" / "metrics" / "xgboost_shap.csv"
    if not p.exists():
        return
    shap_df = pd.read_csv(p).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(shap_df) + 1))
    ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color=TEAL,
            edgecolor="white", linewidth=0.5)
    for v, name in zip(shap_df["mean_abs_shap"], shap_df["feature"]):
        ax.text(v * 1.02, name, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (XGBoost on training fold)")
    ax.set_title("Figure 6.6 — XGBoost feature importance (mean |SHAP|)",
                  loc="left", fontsize=12)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")


def figure_stl_decomposition(out: Path) -> None:
    p = ROOT / "artefacts" / "metrics" / "stl_decomposition.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    splits = Splits.from_config()
    g1 = load_g1()
    train_actual = splits.slice(g1, "train")["total_daily_arrivals"]

    # Show the last 18 months of train for readability
    cutoff = train_actual.index.max() - pd.Timedelta(days=18 * 30)
    df_plot = df[df.index >= cutoff]
    train_plot = train_actual[train_actual.index >= cutoff]

    axes[0].plot(train_plot.index, train_plot.values, color=NEUTRAL, linewidth=0.9,
                  label="Observed")
    axes[0].plot(df_plot.index, df_plot["trend"], color=NAVY, linewidth=1.8,
                  label="Trend")
    axes[0].set_ylabel("Arrivals (count)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title("Figure 6.7 — STL decomposition of daily arrivals "
                       "(period = 7, robust); last 18 months of train",
                       loc="left", fontsize=12)
    axes[1].plot(df_plot.index, df_plot["seasonal"], color=TEAL, linewidth=0.9)
    axes[1].set_ylabel("Seasonal")
    axes[1].axhline(0, color=NEUTRAL, linewidth=0.6, alpha=0.5)
    axes[2].plot(df_plot.index, df_plot["residual"], color=AMBER, linewidth=0.6)
    axes[2].set_ylabel("Residual")
    axes[2].axhline(0, color=NEUTRAL, linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")


def figure_consolidated_panel(lb: pd.DataFrame, out: Path) -> None:
    """A four-quadrant panel: ranked MAPE, family aggregates, val-vs-MAE
    scatter, and a top-3 line plot. Inspired by Susnjak 2023 Fig 6."""
    val = lb[lb["block"] == "val"].copy()
    test = lb[lb["block"] == "test"].copy()
    if val.empty:
        return

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # 1. Ranked MAPE
    ax = fig.add_subplot(gs[0, 0])
    val_s = val.sort_values("MAPE")
    colours = [FAMILY_COLOURS.get(f, NEUTRAL) for f in val_s["family"]]
    ax.barh(val_s["pretty"], val_s["MAPE"], color=colours,
            edgecolor="white", linewidth=0.5)
    ax.axvline(10, color=GREEN, linestyle="--", alpha=0.6)
    ax.set_xlabel("Val MAPE (%)")
    ax.set_title("(a) Models ranked by val MAPE", loc="left", fontsize=11)
    ax.invert_yaxis()

    # 2. Family means
    ax = fig.add_subplot(gs[0, 1])
    fam_means = val.groupby("family")["MAPE"].mean().sort_values()
    fam_colours = [FAMILY_COLOURS[f] for f in fam_means.index]
    ax.barh(fam_means.index, fam_means.values, color=fam_colours,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean val MAPE per family (%)")
    ax.set_title("(b) Family-aggregate performance", loc="left", fontsize=11)
    ax.invert_yaxis()

    # 3. MAPE vs MAE scatter (val)
    ax = fig.add_subplot(gs[1, 0])
    for _, r in val.iterrows():
        ax.scatter(r["MAPE"], r["MAE"], s=100,
                    color=FAMILY_COLOURS.get(r["family"], NEUTRAL),
                    edgecolor=DARK, linewidth=0.5, zorder=3)
        ax.annotate(r["pretty"], (r["MAPE"], r["MAE"]),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=8, alpha=0.85)
    ax.set_xlabel("Val MAPE (%)")
    ax.set_ylabel("Val MAE (patients)")
    ax.set_title("(c) Joint accuracy view (MAPE vs MAE)", loc="left", fontsize=11)

    # 4. Top-3 forecast strip (last 60 val days)
    ax = fig.add_subplot(gs[1, 1])
    splits = Splits.from_config()
    g1 = load_g1()
    val_actual = splits.slice(g1, "val")["total_daily_arrivals"]
    tail = val_actual.iloc[-60:]
    ax.plot(tail.index, tail.values, color=NEUTRAL, linewidth=1.4,
             label="Actual", zorder=5)
    top3 = val.sort_values("MAPE").head(3)
    palette = [TEAL, NAVY, AMBER]
    for (_, r), c in zip(top3.iterrows(), palette):
        m = r["model"]
        if m in ("naive_yest", "naive_seasonal", "dow_mean"):
            ref = pd.read_csv(ROOT / "artefacts" / "predictions" / "reference_floor.csv",
                               parse_dates=["date"])
            sel = ref[(ref["baseline"] == m) & (ref["block"] == "val")]
            x = sel.set_index("date").loc[tail.index, "predicted"]
        else:
            df = pd.read_csv(ROOT / "artefacts" / "predictions" / f"{m}.csv",
                              parse_dates=["date"]).set_index("date")
            x = df.loc[tail.index, "predicted"]
        ax.plot(tail.index, x, linewidth=1.0, color=c, linestyle="--",
                 label=f"{r['pretty']} ({r['MAPE']:.2f}%)")
    ax.set_ylabel("Arrivals")
    ax.set_title("(d) Top-3 vs actual on val tail (60 d)",
                  loc="left", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("Figure 6.11 — Task 1 consolidated comparison panel",
                  fontsize=13, x=0.13, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")


def main() -> None:
    lb = assemble_leaderboard()
    print("Long-form leaderboard:")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(lb.to_string(index=False))

    out_long = ROOT / "artefacts" / "tables" / "leaderboard_task1.csv"
    out_long.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out_long, index=False)
    print(f"\nWrote: {out_long.relative_to(ROOT)}")

    # --- Prompt 2: also write the canonical parquet leaderboard ----------
    try:
        from src.forecasting.leaderboard import append_row, family_of

        parquet_path = ROOT / "artefacts" / "leaderboard_canonical.parquet"
        # Long form has columns: model, block, MAPE, MAE, RMSE, R2, ...
        # Pivot to (model -> val/test dicts) and append one row per model.
        for model in lb["model"].drop_duplicates():
            sub = lb[lb["model"] == model]
            val_row = sub[sub["block"] == "val"]
            test_row = sub[sub["block"] == "test"]
            vm = (
                {"MAPE": float(val_row["MAPE"].iloc[0]),
                 "RMSE": float(val_row["RMSE"].iloc[0])}
                if not val_row.empty else None
            )
            tm = (
                {"MAPE": float(test_row["MAPE"].iloc[0]),
                 "RMSE": float(test_row["RMSE"].iloc[0])}
                if not test_row.empty else None
            )
            append_row(
                parquet_path, model=str(model),
                family=family_of(str(model)),
                criterion="mape",
                val_metrics=vm, test_metrics=tm,
                source_csv="artefacts/tables/leaderboard_task1.csv",
            )
        print(f"Wrote: {parquet_path.relative_to(ROOT)}")
    except Exception as exc:
        print(f"  parquet leaderboard append skipped: {exc}")

    # Wide-form Susnjak-style table
    val = lb[lb["block"] == "val"].set_index("model")[["pretty", "family",
                                                         "MAPE", "MAE",
                                                         "RMSE", "R2"]]
    val = val.rename(columns={"MAPE": "val_MAPE", "MAE": "val_MAE",
                                "RMSE": "val_RMSE", "R2": "val_R2"})
    test = lb[lb["block"] == "test"].set_index("model")[["MAPE", "MAE"]]
    test = test.rename(columns={"MAPE": "test_MAPE", "MAE": "test_MAE"})
    wide = val.join(test, how="left").sort_values("val_MAPE")
    wide["rank_val_MAPE"] = wide["val_MAPE"].rank(method="dense").astype(int)
    out_wide = ROOT / "artefacts" / "tables" / "table_6_5_task1_publication.csv"
    wide.to_csv(out_wide)
    print(f"Wrote: {out_wide.relative_to(ROOT)}")

    figure_ranked_mape(lb, ROOT / "artefacts" / "figures" / "fig_6_5_ranked_mape.png")
    figure_forecast_panel_top3(lb, ROOT / "artefacts" / "figures" / "fig_6_4_forecast_panel.png")
    figure_xgb_shap(ROOT / "artefacts" / "figures" / "fig_6_6_xgb_shap.png")
    figure_stl_decomposition(ROOT / "artefacts" / "figures" / "fig_6_7_stl_decomposition.png")
    figure_consolidated_panel(lb, ROOT / "artefacts" / "figures" / "fig_6_11_consolidated_panel.png")


if __name__ == "__main__":
    main()
