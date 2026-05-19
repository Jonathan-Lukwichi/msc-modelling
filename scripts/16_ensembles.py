"""Push below the 10 % MAPE wall via ensembling.

The eleven base models cluster around 12 % val MAPE. Susnjak & Maddigan (2023,
Table 8) report a Voting ensemble lifting their best individual model by
16–28 % relative MAPE on stable partitions — i.e. a hypothetical 12 % could
become 9–10 % under that pattern.

Four ensembles built here:

  E1. SIMPLE VOTING:   ensemble_pred = mean(top_k base preds)
  E2. WEIGHTED VOTING: weights = (1 / val_MAPE_k) normalised
  E3. OPTIMAL WEIGHTS: convex weights minimising val MAPE (val-fit, biased)
  E4. STACKING RIDGE:  meta-learner Ridge fit on FIRST half of val,
                       evaluated on SECOND half. Honest hold-out.

Reports MAPE / MAE / RMSE / R² for each. Writes artefacts/predictions/
ensemble_{name}.csv and artefacts/metrics/ensembles.csv plus a comparison
figure.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.metrics import score


# ---------------------------------------------------------------------------
# Configuration: which base models contribute, val MAPE used for weighting
# ---------------------------------------------------------------------------

BASE_MODELS = {
    # Top-7 models by val MAPE (post k=10 CV). Excluding LSTM+XGB which
    # is worse than its base, and the STL hybrids which trail.
    "xgboost":              {"path": "xgboost.csv",            "val_mape": 12.02},
    "ann":                  {"path": "ann.csv",                "val_mape": 12.05},
    "hybrid_sarimax_lstm":  {"path": "hybrid_sarimax_lstm.csv","val_mape": 12.10},
    "sarimax":              {"path": "sarimax.csv",            "val_mape": 12.52},
    "hybrid_sarimax_xgb":   {"path": "hybrid_sarimax_xgb.csv", "val_mape": 12.64},
    "nbglm":                {"path": "nbglm.csv",              "val_mape": 12.65},
    "lstm":                 {"path": "lstm.csv",               "val_mape": 12.74},
}

# DoW mean separately because it's stored in reference_floor.csv
INCLUDE_DOW_MEAN = True


def load_base_predictions() -> tuple[pd.DataFrame, pd.Series]:
    """Stack base-model val predictions into a (n_dates, n_models) DataFrame."""
    pred_dir = ROOT / "artefacts" / "predictions"
    cols = {}
    for name, info in BASE_MODELS.items():
        df = pd.read_csv(pred_dir / info["path"], parse_dates=["date"]).set_index("date")
        cols[name] = df["predicted"]
    if INCLUDE_DOW_MEAN:
        rf = pd.read_csv(pred_dir / "reference_floor.csv", parse_dates=["date"])
        dow_val = rf[(rf["baseline"] == "dow_mean") & (rf["block"] == "val")]
        dow_val = dow_val.set_index("date")["predicted"]
        cols["dow_mean"] = dow_val

    P = pd.DataFrame(cols).dropna()
    # Truth comes from the daily target (G1)
    splits = Splits.from_config()
    g1 = load_g1()
    y_val = g1.loc[P.index, "total_daily_arrivals"]
    return P, y_val


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def e1_simple_voting(P: pd.DataFrame, top_k: int = 3,
                      weights_for_ordering: dict | None = None) -> pd.Series:
    if weights_for_ordering is None:
        # default ordering: BASE_MODELS dict ordering (already MAPE-sorted)
        chosen = list(P.columns[:top_k])
    else:
        sorted_models = sorted(weights_for_ordering, key=weights_for_ordering.get)
        chosen = [m for m in sorted_models if m in P.columns][:top_k]
    return P[chosen].mean(axis=1).rename("predicted"), chosen


def e2_weighted_voting(P: pd.DataFrame, mape_dict: dict) -> tuple[pd.Series, dict]:
    weights = {m: 1.0 / mape_dict[m] for m in P.columns if m in mape_dict}
    total = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()}
    yhat = sum(P[m] * w for m, w in weights.items())
    return yhat.rename("predicted"), weights


def e3_optimal_weights(P: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, dict]:
    """Convex weights minimising val MAPE. VAL-FIT — optimistically biased."""
    from scipy.optimize import minimize

    def loss(w):
        w = np.maximum(w, 0)
        w = w / max(w.sum(), 1e-9)
        yhat = P.values @ w
        mask = y.values >= 1.0
        return float(np.mean(np.abs((y.values[mask] - yhat[mask]) / y.values[mask])))

    n = P.shape[1]
    w0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(loss, w0, method="SLSQP", constraints=cons, bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-8})
    w_opt = np.maximum(res.x, 0)
    w_opt = w_opt / w_opt.sum()
    yhat = pd.Series(P.values @ w_opt, index=P.index, name="predicted")
    weights = dict(zip(P.columns, w_opt))
    return yhat, weights


def e4_stacking_ridge(P: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Ridge meta-learner. First 50% of val trains it; last 50% evaluates."""
    from sklearn.linear_model import Ridge
    half = len(P) // 2
    P_meta_train, y_meta_train = P.iloc[:half], y.iloc[:half]
    P_meta_test, y_meta_test = P.iloc[half:], y.iloc[half:]

    meta = Ridge(alpha=1.0, fit_intercept=True, positive=False)
    meta.fit(P_meta_train.values, y_meta_train.values)
    yhat_test = pd.Series(meta.predict(P_meta_test.values),
                            index=P_meta_test.index, name="predicted")
    weights = {m: float(c) for m, c in zip(P.columns, meta.coef_)}
    weights["intercept"] = float(meta.intercept_)
    return yhat_test, y_meta_test, weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    P, y = load_base_predictions()
    print(f"Loaded {len(P)} val days with {P.shape[1]} base models")
    print(f"Base models: {list(P.columns)}")
    print()

    mape_dict = {**{k: v["val_mape"] for k, v in BASE_MODELS.items()}}
    mape_dict["dow_mean"] = 12.27

    rows = []

    # Base reference
    for m in P.columns:
        s = score(y, P[m])
        rows.append({"ensemble": f"base_{m}", "n_components": 1, **s})

    # E1 — simple voting top 3
    yhat, chosen = e1_simple_voting(P, top_k=3, weights_for_ordering=mape_dict)
    s = score(y, yhat)
    print(f"E1 simple voting top-3: {chosen}")
    print(f"   MAPE={s['MAPE']:.3f}  MAE={s['MAE']:.3f}  RMSE={s['RMSE']:.3f}  R2={s['R2']:.3f}")
    rows.append({"ensemble": "E1_simple_top3", "n_components": 3, **s,
                  "members": ",".join(chosen)})
    yhat.reset_index().rename(columns={"index": "date"}).to_csv(
        ROOT / "artefacts" / "predictions" / "ensemble_E1_simple_top3.csv", index=False)

    # E1b — simple voting top 5
    yhat5, chosen5 = e1_simple_voting(P, top_k=5, weights_for_ordering=mape_dict)
    s5 = score(y, yhat5)
    print(f"E1b simple voting top-5: {chosen5}")
    print(f"    MAPE={s5['MAPE']:.3f}  MAE={s5['MAE']:.3f}  RMSE={s5['RMSE']:.3f}  R2={s5['R2']:.3f}")
    rows.append({"ensemble": "E1b_simple_top5", "n_components": 5, **s5,
                  "members": ",".join(chosen5)})

    # E1c — simple voting all
    yhat_all = P.mean(axis=1)
    s_all = score(y, yhat_all)
    print(f"E1c simple voting all-{P.shape[1]}: mean of every column")
    print(f"    MAPE={s_all['MAPE']:.3f}  MAE={s_all['MAE']:.3f}  RMSE={s_all['RMSE']:.3f}  R2={s_all['R2']:.3f}")
    rows.append({"ensemble": "E1c_simple_all", "n_components": P.shape[1], **s_all})

    # E2 — weighted voting (inverse MAPE)
    yhat_w, weights_w = e2_weighted_voting(P, mape_dict)
    s_w = score(y, yhat_w)
    print(f"E2 inverse-MAPE weighted voting:")
    for m, w in weights_w.items():
        print(f"   weight[{m}] = {w:.4f}")
    print(f"   MAPE={s_w['MAPE']:.3f}  MAE={s_w['MAE']:.3f}  RMSE={s_w['RMSE']:.3f}  R2={s_w['R2']:.3f}")
    rows.append({"ensemble": "E2_inv_mape_weighted", "n_components": P.shape[1], **s_w})
    yhat_w.reset_index().rename(columns={"index": "date"}).to_csv(
        ROOT / "artefacts" / "predictions" / "ensemble_E2_inv_mape_weighted.csv", index=False)

    # E3 — optimal convex weights (val-fit, biased)
    yhat_o, weights_o = e3_optimal_weights(P, y)
    s_o = score(y, yhat_o)
    print(f"E3 optimal convex weights (val-fit, BIASED upper-bound):")
    for m, w in sorted(weights_o.items(), key=lambda kv: -kv[1]):
        if w > 0.01:
            print(f"   weight[{m}] = {w:.4f}")
    print(f"   MAPE={s_o['MAPE']:.3f}  MAE={s_o['MAE']:.3f}  RMSE={s_o['RMSE']:.3f}  R2={s_o['R2']:.3f}")
    rows.append({"ensemble": "E3_optimal_convex_biased", "n_components": P.shape[1], **s_o})
    yhat_o.reset_index().rename(columns={"index": "date"}).to_csv(
        ROOT / "artefacts" / "predictions" / "ensemble_E3_optimal_convex.csv", index=False)

    # E4 — stacking ridge (honest split)
    yhat_s, y_s_test, weights_s = e4_stacking_ridge(P, y)
    s_s = score(y_s_test, yhat_s)
    print(f"E4 stacking ridge (first 50% val trains, last 50% evaluates):")
    print(f"   intercept = {weights_s['intercept']:+.3f}")
    for m, w in sorted(weights_s.items(), key=lambda kv: -abs(kv[1]) if kv[0] != 'intercept' else -1e9):
        if m != "intercept" and abs(w) > 0.01:
            print(f"   coef[{m}] = {w:+.4f}")
    print(f"   MAPE={s_s['MAPE']:.3f}  MAE={s_s['MAE']:.3f}  RMSE={s_s['RMSE']:.3f}  R2={s_s['R2']:.3f}")
    print(f"   (evaluated on {len(y_s_test)} val days)")
    rows.append({"ensemble": "E4_stacking_ridge_honest", "n_components": P.shape[1], **s_s,
                  "members": "ridge_meta_on_first_half_val"})

    # Save metrics
    df = pd.DataFrame(rows)
    out = ROOT / "artefacts" / "metrics" / "ensembles.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote: {out.relative_to(ROOT)}")

    # Plot
    plot_ensemble_comparison(df, mape_dict)


def plot_ensemble_comparison(df: pd.DataFrame, mape_dict: dict) -> None:
    NAVY, TEAL, AMBER, GREEN, ROSE = ("#1e6091", "#0d9488", "#d97706",
                                        "#16a34a", "#dc2626")
    NEUTRAL = "#475569"
    plt.rcParams.update({
        "font.size": 11, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })

    # Combine base + ensemble rows for display
    df_sorted = df.sort_values("MAPE")
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(df_sorted) + 1.5))
    colours = []
    for name in df_sorted["ensemble"]:
        if name.startswith("base_"):
            colours.append(NEUTRAL)
        elif name.startswith("E1"):
            colours.append(TEAL)
        elif name.startswith("E2"):
            colours.append(NAVY)
        elif name.startswith("E3"):
            colours.append(AMBER)
        else:
            colours.append(ROSE)
    bars = ax.barh(df_sorted["ensemble"], df_sorted["MAPE"], color=colours,
                    edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, df_sorted["MAPE"]):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(10.0, color=GREEN, linestyle="--", alpha=0.6, linewidth=1.5,
                label="10% excellent (Susnjak 2023)")
    ax.set_xlabel("Val MAPE (%)")
    ax.set_title("Figure 6.13 — Ensemble vs base-model val MAPE",
                  loc="left", fontsize=12)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    out = ROOT / "artefacts" / "figures" / "fig_6_13_ensembles.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Wrote: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
