"""Ablation study: which design choices actually move the needle?

Three standalone models (XGBoost, ANN, LSTM) trained on a grid of six scenarios:

  1. BASELINE   — Ch5 §5.5.2 split (post-COVID), §3.4.3 consensus 23-feature set,
                  tuned hyperparameters. The reference everyone else is compared
                  against.
  2. IGNORE_COVID — same procedure but trained on the FULL 2019-05 -> 2024-06
                    window (pre-COVID + during-COVID + post-COVID combined).
                    Shows the cost of not respecting the §5.5.2 boundary.
  3. COVID_AWARE — full window + an `is_covid` indicator that flags the
                   2020-03 to 2022-03 disruption period. Shows whether the
                   model can recover the lost accuracy by being told about
                   COVID explicitly.
  4. NO_FE       — only the §5.2.5 raw 10 features (no lags, no rolling, no
                   Fourier). Shows the value of §3.4.2 engineering.
  5. NO_HPO      — vanilla hyperparameters (library defaults / naive choices).
                   Shows the value of §3.5.9 tuning.
  6. NO_FS       — all 100 engineered features (no §3.4.3 consensus filter).
                   Shows the value of the 4-method consensus selection.

Val MAPE / MAE / RMSE / R² are reported for every (model, scenario) cell.
A single ablation plot ranks the design choices by their impact.

This script uses a FIXED hyperparameter set per model so the only thing
varying between scenarios is the design choice being ablated. Where a
best_params.json from the CV runs is available, those params are loaded as
the "tuned" choice; otherwise sensible defaults are used.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------

def _full_g1_filtered() -> pd.DataFrame:
    """Load G1, filter zero-arrival days, return the FULL 2019-05 to 2026-01 frame."""
    return load_g1()


def _add_is_covid_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add an is_covid indicator that flags 2020-03-01 to 2022-02-28."""
    out = df.copy()
    out["is_covid_period"] = (
        (out.index >= "2020-03-01") & (out.index <= "2022-02-28")
    ).astype(int)
    return out


def _scenario_data(scenario: str, feature_kind: str = "consensus"):
    """Build (X, y) frames for the train window of a given scenario.

    feature_kind in {"consensus" (23), "raw10" (15 after DoW dummies), "all100"}.

    Returns:
        X_train, y_train  -- training-fold features and target for the scenario
        X_val, y_val      -- ALWAYS the Ch5 §5.5.2 val block (held-out check)
    """
    splits = Splits.from_config()
    g1 = _full_g1_filtered()
    target = g1["total_daily_arrivals"]

    if feature_kind == "consensus":
        eng = load_engineered()
        X_all = build_selected_X(eng)            # 23 cols
    elif feature_kind == "raw10":
        # Use §5.2.5 inventory directly (no engineered features).
        X_all, _ = build_task1_exogenous(g1, fit_scaler=True)  # 15 cols
    elif feature_kind == "all100":
        eng = load_engineered()
        # ALL these columns are direct constituents or near-duplicates of the
        # target total_daily_arrivals = arrival_normal_hours + arrival_after_hours.
        # Dropping them is essential to avoid target leakage in this scenario.
        drop = {
            # Target itself + raw counts that compose it
            "total_daily_arrivals", "patient_count", "patient_count_totals_raw",
            "arrival_subtotal", "arrival_normal_hours", "arrival_after_hours",
            "attendant_count",
            # Triage subtotals (sum to the target)
            "triage_nh_total", "triage_ah_total",
            # Per-priority counts within normal/after hours (sum to the target)
            "p1_normal_hours", "p2_normal_hours", "p3_normal_hours",
            "p1_after_hours", "p2_after_hours", "p3_after_hours",
            # Patient-flow counters tied to the daily total
            "internal_transfer_in", "external_transfer_in",
            "discharges_rht_abscond", "deaths_p4",
            "internal_transfer_out", "external_transfer_out",
            "carry_over_midnight", "transfer_in_subtotal", "separation_subtotal",
            # Meta calendar columns we don't want as features
            "year", "month", "day", "day_of_year", "is_zero_day",
        }
        X_all = eng.drop(columns=[c for c in drop if c in eng.columns],
                          errors="ignore")
        X_all = X_all.select_dtypes(include=[np.number])
    else:
        raise ValueError(feature_kind)

    # Inner-join target and features so X has the same dates as y
    df = pd.concat([target.rename("y"), X_all], axis=1, join="inner").dropna()

    val_idx = splits.slice(g1, "val").index.intersection(df.index)

    if scenario == "post_covid":
        # Ch5 §5.5.2 train block only
        train_idx = splits.slice(g1, "train").index.intersection(df.index)
    elif scenario == "full_window":
        # Everything before val (pre-COVID + during-COVID + post-COVID training)
        train_idx = df.index[df.index < val_idx[0]]
    elif scenario == "full_window_with_covid_flag":
        # Same as full_window but the is_covid flag is added below
        train_idx = df.index[df.index < val_idx[0]]
        df = _add_is_covid_flag(df)
    else:
        raise ValueError(scenario)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Hyperparameter sets
# ---------------------------------------------------------------------------

def _load_or_default(name: str, defaults: dict) -> dict:
    p = ROOT / "artefacts" / "models" / f"{name}_best_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return defaults
    return defaults


def tuned_params() -> dict:
    xgb_default = {"n_estimators": 200, "max_depth": 4,
                    "learning_rate": 0.05, "subsample": 0.85}
    ann_default = {"hidden_layers": 1, "units": 128, "dropout": 0.2,
                    "learning_rate": 0.001, "batch_size": 32, "seed": 42}
    lstm_default = {"lookback": 21, "units": 128, "dropout": 0.2,
                     "learning_rate": 0.001, "batch_size": 32, "seed": 42}
    return {
        "xgboost": _load_or_default("xgboost", xgb_default),
        "ann": _load_or_default("ann", ann_default),
        "lstm": _load_or_default("lstm", lstm_default),
    }


def naive_params() -> dict:
    """Vanilla/default hyperparameters used in the no-HPO scenario."""
    return {
        "xgboost": {"n_estimators": 100, "max_depth": 6,
                     "learning_rate": 0.3, "subsample": 1.0},
        "ann": {"hidden_layers": 1, "units": 64, "dropout": 0.0,
                 "learning_rate": 0.01, "batch_size": 32, "seed": 42},
        "lstm": {"lookback": 14, "units": 64, "dropout": 0.0,
                  "learning_rate": 0.01, "batch_size": 32, "seed": 42},
    }


# ---------------------------------------------------------------------------
# Model wrappers (single fit + val predict, no HPO inside)
# ---------------------------------------------------------------------------

def fit_predict_xgboost(X_train, y_train, X_val, params):
    from xgboost import XGBRegressor
    model = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train.values, y_train.values)
    return model.predict(X_val.values)


def fit_predict_ann(X_train, y_train, X_val, params):
    import torch
    from src.forecasting.models.ann import _MLP, _seed_everything
    _seed_everything(params.get("seed", 42))
    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32)
    Xva = ((X_val - mean) / std).astype(np.float32)
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32)
    Xtr_t = torch.from_numpy(Xtr.values)
    ytr_t = torch.from_numpy(ytr.values)
    Xva_t = torch.from_numpy(Xva.values)
    n_hidden = [params["units"]] * params["hidden_layers"]
    model = _MLP(n_in=Xtr.shape[1], n_hidden=n_hidden, dropout=params["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    bs = params["batch_size"]
    n = len(Xtr_t)
    for epoch in range(60):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        yhat_norm = model(Xva_t).numpy()
    return yhat_norm * y_std + y_mean


def fit_predict_lstm(X_train, y_train, X_val, params):
    import torch
    from src.forecasting.models.lstm import (
        _LSTMNet, _build_sequences, _seed_everything,
    )
    _seed_everything(params.get("seed", 42))
    mean = X_train.mean()
    std = X_train.std(ddof=0).replace(0, 1.0)
    Xtr = ((X_train - mean) / std).astype(np.float32).values
    Xva = ((X_val - mean) / std).astype(np.float32).values
    y_mean, y_std = float(y_train.mean()), float(y_train.std(ddof=0))
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).values

    lookback = params["lookback"]
    Xtr_seq, ytr_seq = _build_sequences(Xtr, ytr, lookback)
    model = _LSTMNet(n_features=Xtr.shape[1], units=params["units"],
                      dropout=params["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    bs = params["batch_size"]
    n = Xtr_seq.shape[0]
    for epoch in range(30):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr_seq[idx])
            yb = torch.from_numpy(ytr_seq[idx])
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    # Predict val using sliding window
    full = np.vstack([Xtr, Xva])
    preds = []
    with torch.no_grad():
        for i in range(len(Xva)):
            pos = len(Xtr) + i
            if pos < lookback:
                continue
            window = full[pos - lookback : pos]
            preds.append(float(model(torch.from_numpy(window[None, :, :])).item()))
    return np.array(preds) * y_std + y_mean


MODELS = {
    "xgboost": fit_predict_xgboost,
    "ann": fit_predict_ann,
    "lstm": fit_predict_lstm,
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    description: str
    data_scenario: str            # post_covid / full_window / full_window_with_covid_flag
    feature_kind: str             # consensus / raw10 / all100
    use_tuned_params: bool        # True = tuned (loaded or sensible), False = naive defaults


SCENARIOS = [
    Scenario("baseline",
              "§5.5.2 split (post-COVID) + §3.4.3 consensus 23-feature set + tuned HPO",
              "post_covid", "consensus", True),
    Scenario("ignore_covid",
              "Full 2019-05 -> 2024-06 train window (pre+during+post-COVID combined)",
              "full_window", "consensus", True),
    Scenario("covid_aware",
              "Full window + is_covid_period flag indicating 2020-03 -> 2022-02",
              "full_window_with_covid_flag", "consensus", True),
    Scenario("no_feature_engineering",
              "§5.2.5 raw 10 features only (no lags, no rolling, no Fourier)",
              "post_covid", "raw10", True),
    Scenario("no_hpo",
              "Vanilla/default hyperparameters (no HPO search)",
              "post_covid", "consensus", False),
    Scenario("no_feature_selection",
              "All 100 engineered features (no §3.4.3 consensus filter)",
              "post_covid", "all100", True),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tuned = tuned_params()
    naive = naive_params()
    rows = []
    print(f"Ablation: {len(SCENARIOS)} scenarios x 3 models = {len(SCENARIOS)*3} runs\n")
    for scen in SCENARIOS:
        print(f"=== Scenario: {scen.name} ===")
        print(f"    {scen.description}")
        X_train, y_train, X_val, y_val = _scenario_data(
            scen.data_scenario, scen.feature_kind,
        )
        print(f"    Train rows: {len(X_train)}  |  Val rows: {len(X_val)}  "
              f"|  Features: {X_train.shape[1]}")
        params_set = tuned if scen.use_tuned_params else naive
        for model_name, fit_predict in MODELS.items():
            t0 = time.time()
            params = params_set[model_name]
            try:
                yhat = fit_predict(X_train, y_train, X_val, params)
                # Align in case of lookback truncation (LSTM)
                actual = y_val.iloc[len(y_val) - len(yhat):]
                m = score(actual.values, yhat)
                status = "OK"
            except Exception as exc:
                m = {"MAPE": float("nan"), "MAE": float("nan"),
                      "RMSE": float("nan"), "R2": float("nan")}
                status = f"FAIL: {exc}"
            took = time.time() - t0
            rows.append({
                "scenario": scen.name,
                "description": scen.description,
                "model": model_name,
                "MAPE": m["MAPE"], "MAE": m["MAE"],
                "RMSE": m["RMSE"], "R2": m["R2"],
                "took_s": round(took, 1),
                "status": status,
                "params": json.dumps(params),
                "n_train": len(X_train), "n_val": len(X_val),
                "n_features": X_train.shape[1],
            })
            print(f"    {model_name:<8s} -> MAPE={m['MAPE']:6.3f}  "
                  f"MAE={m['MAE']:5.2f}  R2={m['R2']:+5.2f}  "
                  f"({status}, {took:.1f}s)")
        print()

    df = pd.DataFrame(rows)
    out_csv = ROOT / "artefacts" / "tables" / "ablation_study.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv.relative_to(ROOT)}")

    # Pivot for plotting
    pivot_mape = df.pivot(index="scenario", columns="model", values="MAPE")
    print("\nVal MAPE matrix (scenarios x models):")
    print(pivot_mape.to_string(float_format=lambda v: f"{v:.2f}"))

    # Compute deltas vs baseline
    if "baseline" in pivot_mape.index:
        baseline = pivot_mape.loc["baseline"]
        deltas = pivot_mape.subtract(baseline, axis=1)
        print("\nDelta MAPE vs baseline (positive = worse than baseline):")
        print(deltas.to_string(float_format=lambda v: f"{v:+.2f}"))

        # Plot
        try:
            import matplotlib.pyplot as plt
            from scripts._plot_helpers import NAVY, TEAL, AMBER, GREEN, ROSE, NEUTRAL
        except Exception:
            import matplotlib.pyplot as plt
            NAVY, TEAL, AMBER, GREEN, ROSE, NEUTRAL = (
                "#1e6091", "#0d9488", "#d97706", "#16a34a", "#dc2626", "#475569")
        scenario_order = [s.name for s in SCENARIOS]
        scenario_order = [s for s in scenario_order if s in deltas.index]
        deltas_ordered = deltas.loc[scenario_order]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [3, 4]})

        # (a) Absolute MAPE per scenario per model
        ax = axes[0]
        x = np.arange(len(scenario_order))
        width = 0.25
        for i, model_name in enumerate(["xgboost", "ann", "lstm"]):
            if model_name not in pivot_mape.columns:
                continue
            ax.bar(x + (i - 1) * width, pivot_mape.loc[scenario_order, model_name],
                    width, label=model_name.upper(),
                    color=[TEAL, NAVY, AMBER][i],
                    edgecolor="white", linewidth=0.5)
        ax.axhline(10, color=GREEN, linestyle="--", alpha=0.5,
                    linewidth=1, label="10% excellent (Susnjak 2023)")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order, rotation=20, ha="right")
        ax.set_ylabel("Val MAPE (%)")
        ax.set_title("(a) Absolute val MAPE per scenario", loc="left", fontsize=11)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

        # (b) Delta vs baseline as heatmap
        ax = axes[1]
        nonbase = [s for s in scenario_order if s != "baseline"]
        delta_plot = deltas_ordered.loc[nonbase]
        im = ax.imshow(delta_plot.values, cmap="RdYlGn_r", aspect="auto",
                        vmin=-3, vmax=8)
        ax.set_xticks(np.arange(len(delta_plot.columns)))
        ax.set_xticklabels([c.upper() for c in delta_plot.columns])
        ax.set_yticks(np.arange(len(nonbase)))
        ax.set_yticklabels(nonbase)
        for i in range(delta_plot.shape[0]):
            for j in range(delta_plot.shape[1]):
                v = delta_plot.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                             color="white" if abs(v) > 3 else "black",
                             fontsize=10)
        ax.set_title("(b) Δ val MAPE vs baseline (red=worse, green=better)",
                      loc="left", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Δ MAPE (pp)")
        fig.suptitle("Figure 6.12 — Ablation study: design-choice impact on val MAPE",
                      fontsize=13, x=0.05, ha="left")
        plt.tight_layout()
        out_png = ROOT / "artefacts" / "figures" / "fig_6_12_ablation.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Wrote: {out_png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
