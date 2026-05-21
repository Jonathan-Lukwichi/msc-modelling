"""Adaptive Conformal Inference UQ rerun (Prompt 8 orchestrator).

Replaces the static split-conformal numbers in scripts/21 with a
gamma-grid ACI sweep over the best base models. Bases used here:

  - XGBoost (RMSE-best params, predictions in artefacts/predictions/{val,test}/xgboost_rmse*.csv)
  - SARIMAX (predictions in artefacts/predictions/{val,test}/sarimax*.csv)
  - SARIMAX+LSTM hybrid (the only hybrid that beat its base on val)

For each base x method (split / aci) x gamma, we report coverage at
the 80% and 95% nominal levels and the Winkler score.

Output:
  - artefacts/metrics/uq_coverage_aci.csv
  - artefacts/figures/fig_6_uq_aci_coverage_trajectory.png

References:
  - Gibbs & Candès (2021) arXiv:2106.00170 — ACI.
  - Zaffran et al. (2022) ICML — ACI for time series.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.uq.aci import evaluate_aci_grid, aci_intervals


PRED = ROOT / "artefacts" / "predictions"
PRED_TEST = PRED / "test"

# Map a logical "base" name to (val_csv, test_csv).
BASES = {
    "XGBoost":      (PRED / "xgboost_rmse.csv",          PRED_TEST / "xgboost_rmse.csv"),
    "SARIMAX":      (PRED / "sarimax.csv",               PRED_TEST / "sarimax.csv"),
    "Hybrid_SARIMAX_LSTM": (
        PRED / "hybrid_sarimax_lstm_rmse.csv",
        PRED_TEST / "hybrid_sarimax_lstm_rmse.csv",
    ),
}

GAMMAS = (0.0, 0.001, 0.005, 0.01, 0.05)


def _read_preds(path: Path) -> pd.DataFrame:
    """Return DataFrame indexed by date with columns 'actual' and 'predicted'."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    # Normalise column names that vary across upstream scripts.
    cols = {c.lower(): c for c in df.columns}
    actual_col = cols.get("actual") or cols.get("y_true") or cols.get("y")
    pred_col = (cols.get("predicted") or cols.get("yhat")
                 or cols.get("y_pred") or cols.get("forecast"))
    if actual_col is None or pred_col is None:
        raise ValueError(
            f"Cannot identify actual/predicted columns in {path}; "
            f"got {list(df.columns)}"
        )
    return df[[actual_col, pred_col]].rename(
        columns={actual_col: "actual", pred_col: "predicted"}
    )


def run_one_base(name: str, val_csv: Path, test_csv: Path) -> list[dict]:
    val = _read_preds(val_csv)
    test = _read_preds(test_csv)

    # Calibration: |residuals| from val.
    cal_residuals = np.abs(val["actual"].values - val["predicted"].values)
    val_residuals_var = float(np.var(cal_residuals, ddof=0))

    rows = []
    for alpha_target, level in [(0.20, 80), (0.05, 95)]:
        grid = evaluate_aci_grid(
            y_eval=test["actual"].values,
            yhat_eval=test["predicted"].values,
            calib_residuals=cal_residuals,
            alpha_target=alpha_target,
            gammas=GAMMAS,
            dates=test.index,
        )
        for _, r in grid.iterrows():
            rows.append({
                "base": name,
                "level": level,
                "method": r["method"],
                "gamma": r["gamma"],
                "coverage": r["coverage"],
                "mean_width": r["mean_width"],
                "winkler": r["winkler"],
                "n_cal": len(cal_residuals),
                "n_test": len(test),
            })
        # Save full trajectory at the strongest gamma for the 95% level.
        best_gamma = float(grid.loc[
            (grid["gamma"] > 0) & (grid["coverage"] >= 0.93),
            "gamma"
        ].min() if any((grid["gamma"] > 0) & (grid["coverage"] >= 0.93))
        else GAMMAS[2])  # 0.005
        if alpha_target == 0.05:
            res = aci_intervals(
                y_eval=test["actual"].values,
                yhat_eval=test["predicted"].values,
                calib_residuals=cal_residuals,
                alpha_target=alpha_target,
                gamma=best_gamma,
                dates=test.index,
            )
            traj = pd.DataFrame({
                "date": test.index,
                "actual": test["actual"].values,
                "yhat": res.yhat,
                "lower": res.lower,
                "upper": res.upper,
                "alpha_t": res.alpha_trace,
                "covered": res.coverage_trace,
            })
            traj.to_csv(
                ROOT / "artefacts" / "predictions" /
                f"aci_trajectory_{name}.csv", index=False,
            )
    return rows


def figure_coverage_trajectory(df: pd.DataFrame) -> None:
    """Rolling coverage trajectory (95% target) for the three bases."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colours = {"XGBoost": "#0d9488", "SARIMAX": "#1e6091",
                "Hybrid_SARIMAX_LSTM": "#dc2626"}
    for base in BASES.keys():
        traj_path = ROOT / "artefacts" / "predictions" / f"aci_trajectory_{base}.csv"
        if not traj_path.exists():
            continue
        traj = pd.read_csv(traj_path, parse_dates=["date"])
        # Rolling 28-day coverage average
        traj["rolling_cov"] = traj["covered"].rolling(28, min_periods=14).mean()
        ax.plot(traj["date"], traj["rolling_cov"], label=base,
                  color=colours.get(base, "#475569"), linewidth=1.6)
    ax.axhline(0.95, color="#0f172a", linestyle="--", linewidth=1, alpha=0.6,
                label="Target 95%")
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("28-day rolling empirical coverage")
    ax.set_title(
        "Figure 6.UQ — ACI rolling coverage trajectory on the 396-day test block "
        "(Gibbs-Candès 2021)", loc="left", fontsize=11,
    )
    ax.legend(loc="lower right", frameon=False)
    out = ROOT / "artefacts" / "figures" / "fig_6_uq_aci_coverage_trajectory.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")


def main():
    print("=" * 70)
    print("PROMPT 8 — ACI gamma-grid over best bases (Gibbs-Candès 2021)")
    print("=" * 70 + "\n")
    rows: list[dict] = []
    for base, (val_csv, test_csv) in BASES.items():
        if not val_csv.exists() or not test_csv.exists():
            print(f"  Skip {base}: predictions not found "
                   f"({val_csv.name} / {test_csv.name})")
            continue
        print(f"\n--- {base} ---")
        base_rows = run_one_base(base, val_csv, test_csv)
        rows.extend(base_rows)
        # Print compact 95% summary
        sub = [r for r in base_rows if r["level"] == 95]
        for r in sub:
            tag = f"  gamma={r['gamma']:.3f}" if r["method"] == "aci" else "  split    "
            print(f"  {tag} cov={r['coverage']:.3f}  width={r['mean_width']:.2f}  "
                  f"Winkler={r['winkler']:.2f}")

    df = pd.DataFrame(rows)
    out = ROOT / "artefacts" / "metrics" / "uq_coverage_aci.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote: {out.relative_to(ROOT)}")
    print("\n--- Headline 95% table ---")
    print(
        df[df["level"] == 95].pivot_table(
            index=["base", "method", "gamma"],
            values=["coverage", "mean_width", "winkler"],
        ).round(3).to_string()
    )

    figure_coverage_trajectory(df)


if __name__ == "__main__":
    main()
