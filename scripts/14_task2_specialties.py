"""Plan §13 Step 8: Task 2 — per-specialty daily forecast.

Five specialties at daily resolution (Medicine, Orthopaedics, Surgery,
Paediatrics, Gynaecology) per Ch5 §5.3.1. Each one gets its own ARIMA,
SARIMAX, and XGBoost fit on the share-of-header target:

    share_t = spec_count_t / total_daily_arrivals_t

with per-specialty exogenous block per Ch5 §5.3.3:
  - Medicine:    temp + wind
  - Orthopaedics: temp only
  - Surgery:     no weather; explicit is_weekend / is_long_weekend /
                 is_public_holiday interaction columns to capture the
                 +41% / +32% / +26% sign-reversal (Ch5 §5.3.3)
  - Paediatrics: wind only
  - Gynaecology: neither (weather-flat)

Two weekly specialties (Maternity, Psychiatry) get a SARIMA(p,1,q)(P,1,Q)_52
on weekly sum. ~90% zero days motivates the weekly aggregation.

Outputs:
  - artefacts/predictions/task2_{specialty}_{model}.csv
  - artefacts/metrics/task2_per_specialty_metrics.csv
  - artefacts/metrics/task2_sum_consistency.csv
  - artefacts/figures/fig_6_8_per_specialty_heatmap.png
  - artefacts/figures/fig_6_9_surgery_sign_reversal.png
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g3, Splits, load_g1
from src.forecasting.features import build_task1_exogenous, StandardScaler
from src.forecasting.metrics import score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Spec mapping (config → G3 column)
# ---------------------------------------------------------------------------

DAILY_SPECIALTIES = {
    "Medicine":     "spec_medicine",
    "Orthopaedics": "spec_orthopaedics",
    "Surgery":      "spec_surgery",
    "Paediatrics":  "spec_paediatrics",
    "Gynaecology":  "spec_gynae",
}
WEEKLY_SPECIALTIES = {
    "Maternity":  "spec_maternity",
    "Psychiatry": "spec_psychiatry",
}


def _load_task2_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "features_task2.yaml").read_text()
    )


# ---------------------------------------------------------------------------
# Feature builders per specialty
# ---------------------------------------------------------------------------

def build_specialty_exogenous(
    df: pd.DataFrame, specialty: str,
    scaler: StandardScaler | None = None, fit_scaler: bool = False,
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """Per-specialty X block per Ch5 §5.3.3."""
    cfg = _load_task2_config()
    binaries = cfg["shared_calendar"]
    if specialty in cfg["daily_specialties"]:
        spec_cfg = cfg["daily_specialties"][specialty]
    elif specialty in cfg["weekly_specialties"]:
        spec_cfg = cfg["weekly_specialties"][specialty]
    else:
        raise KeyError(specialty)

    weather_cols = spec_cfg.get("weather", []) or []
    interaction_cols = spec_cfg.get("interactions", []) or []

    # DoW dummies (6, Monday as reference)
    dow = df["day_of_week"].astype(int)
    out = pd.DataFrame(index=df.index)
    for d in range(1, 7):
        out[f"dow_{d}"] = (dow == d).astype(int)

    # Calendar binaries
    for b in binaries:
        if b in df.columns:
            out[b] = df[b].astype(int)

    # Weather (scaled on train)
    if weather_cols:
        weather_block = df[weather_cols].copy()
        if fit_scaler:
            scaler = StandardScaler.fit(df, weather_cols)
        if scaler is not None:
            weather_block = scaler.transform(weather_block)
        for c in weather_cols:
            out[c] = weather_block[c]

    # Per-specialty interaction columns (Surgery sign-reversal)
    for c in interaction_cols:
        out[f"{specialty.lower()}_{c}"] = df[c].astype(int)

    return out, scaler


# ---------------------------------------------------------------------------
# Per-specialty model fits
# ---------------------------------------------------------------------------

def fit_arima_share(y_train_share, val_idx, full_share):
    """ARIMA(p, 1, q) on the share series; rolling weekly refit on val."""
    from pmdarima import auto_arima, ARIMA as PmARIMA
    np.random.seed(42)
    base = auto_arima(
        y_train_share.values,
        start_p=0, start_q=0, max_p=3, max_q=3,
        d=1, seasonal=False,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic",
        random_state=42,
    )
    order = base.order
    rows = []
    full = pd.concat([y_train_share, full_share.loc[val_idx]]).sort_index()
    origin_pos = full.index.get_loc(val_idx[0]) - 1
    step = 7
    while origin_pos < full.index.get_loc(val_idx[-1]):
        h = int(min(step, full.index.get_loc(val_idx[-1]) - origin_pos))
        m = PmARIMA(order=order, suppress_warnings=True)
        m.fit(full.iloc[: origin_pos + 1].values)
        yhat = m.predict(n_periods=h)
        dates = full.index[origin_pos + 1 : origin_pos + 1 + h]
        for d, yp in zip(dates, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin_pos += step
    return pd.DataFrame(rows).set_index("date"), order


def fit_sarimax_share(y_train_share, X_train, X_full, val_idx, full_share):
    """SARIMAX(p, 1, q)(P, 1, Q)_7 on share with per-specialty exog."""
    from pmdarima import auto_arima, ARIMA as PmARIMA
    np.random.seed(42)
    base = auto_arima(
        y_train_share.values, X=X_train.values,
        start_p=0, start_q=0, max_p=2, max_q=2,
        max_P=2, max_Q=2, d=1, D=1,
        seasonal=True, m=7,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic",
        random_state=42,
    )
    order = base.order
    seasonal_order = base.seasonal_order

    rows = []
    full = pd.concat([y_train_share, full_share.loc[val_idx]]).sort_index()
    origin_pos = full.index.get_loc(val_idx[0]) - 1
    step = 7
    while origin_pos < full.index.get_loc(val_idx[-1]):
        h = int(min(step, full.index.get_loc(val_idx[-1]) - origin_pos))
        m = PmARIMA(order=order, seasonal_order=seasonal_order,
                    suppress_warnings=True)
        y_tr = full.iloc[: origin_pos + 1]
        X_tr = X_full.loc[y_tr.index]
        X_fu = X_full.iloc[X_full.index.get_loc(full.index[origin_pos + 1]):
                            X_full.index.get_loc(full.index[origin_pos + h])]
        # Adjust X_fu to exact h rows
        X_fu = X_full.iloc[origin_pos + 1 : origin_pos + 1 + h]
        m.fit(y_tr.values, X=X_tr.values)
        yhat = m.predict(n_periods=h, X=X_fu.values)
        dates = full.index[origin_pos + 1 : origin_pos + 1 + h]
        for d, yp in zip(dates, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin_pos += step

    return pd.DataFrame(rows).set_index("date"), order, seasonal_order, base


def fit_xgb_share(y_train_share, X_train, X_full, val_idx):
    """XGBoost with light defaults (no per-specialty HPO; ~30s)."""
    from xgboost import XGBRegressor
    params = {"n_estimators": 200, "max_depth": 4,
               "learning_rate": 0.05, "subsample": 0.85}
    rows = []
    full_y = pd.concat([y_train_share]).sort_index()
    train_end = full_y.index[-1]
    step = 7
    # rolling weekly refit on val
    origin = train_end
    val_remaining = list(val_idx)
    while val_remaining:
        h = min(step, len(val_remaining))
        future_dates = val_remaining[:h]
        train_y = y_train_share[y_train_share.index <= origin]
        X_tr = X_full.loc[train_y.index]
        X_fu = X_full.loc[future_dates]
        m = XGBRegressor(**params, objective="reg:squarederror",
                          random_state=42, verbosity=0, n_jobs=-1)
        m.fit(X_tr.values, train_y.values)
        yhat = m.predict(X_fu.values)
        for d, yp in zip(future_dates, yhat):
            rows.append({"date": d, "predicted": float(yp)})
        origin = future_dates[-1]
        # Extend training data with observed val so far (rolling expanding)
        # but here we expand from full y_train_share once we know more val truth
        # Note: y_train_share itself stays static; we use it as the train set
        # and just slide the origin forward to refit weekly.
        val_remaining = val_remaining[h:]
    return pd.DataFrame(rows).set_index("date"), params


# ---------------------------------------------------------------------------
# Weekly specialties
# ---------------------------------------------------------------------------

def run_weekly_specialty(g3: pd.DataFrame, splits: Splits,
                          name: str, col: str) -> dict:
    """SARIMA(p,1,q)(P,1,Q)_52 on weekly-sum target."""
    from pmdarima import auto_arima
    print(f"\n=== Weekly specialty: {name} ===")
    target = g3[col]
    train = target.loc[splits.slice(g3, "train").index]
    val = target.loc[splits.slice(g3, "val").index]

    # Resample to weekly Mon-Sun sum
    train_w = train.resample("W-SUN").sum()
    val_w = val.resample("W-SUN").sum()
    print(f"  Train: {len(train_w)} weeks  |  Val: {len(val_w)} weeks  "
          f"(mean/wk: train={train_w.mean():.2f}, val={val_w.mean():.2f})")

    np.random.seed(42)
    model = auto_arima(
        train_w.values, start_p=0, start_q=0,
        max_p=2, max_q=2, max_P=1, max_Q=1,
        d=1, D=1, seasonal=True, m=52,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic",
        random_state=42,
    )
    yhat = model.predict(n_periods=len(val_w))
    metrics = score(val_w.values, np.asarray(yhat))
    print(f"  Order: {model.order} x {model.seasonal_order}  "
          f"AIC={model.aic():.2f}")
    print(f"  Val MAPE={metrics['MAPE']:.3f}  MAE={metrics['MAE']:.3f}  "
          f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")
    return {
        "specialty": name, "resolution": "weekly", "model": "sarima_w",
        "block": "val", **metrics,
        "n_train": len(train_w), "n_val": len(val_w),
        "order": str(model.order),
        "seasonal_order": str(model.seasonal_order),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    splits = Splits.from_config()
    g3 = load_g3()  # filtered is_zero_day == 0
    train_idx_full = splits.slice(g3, "train").index
    val_idx_full = splits.slice(g3, "val").index
    print(f"G3: {g3.shape}  |  Train: {len(train_idx_full)}  Val: {len(val_idx_full)}")

    out_pred = ROOT / "artefacts" / "predictions"
    out_metrics = ROOT / "artefacts" / "metrics"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    rows = []
    abs_pred_by_specialty = {}

    # ------------ Daily specialties ------------
    for name, col in DAILY_SPECIALTIES.items():
        print(f"\n=== Daily specialty: {name}  (G3 col: {col}) ===")
        t0 = time.time()
        # Target: share-of-header (drop NaN AND infinity from division by zero)
        share = (g3[col] / g3["total_daily_arrivals"]).rename("share")
        share = share.replace([np.inf, -np.inf], np.nan)
        share_train = share.loc[train_idx_full].dropna()
        share_val = share.loc[val_idx_full].dropna()

        # Restrict indices to where share is defined
        train_idx = share_train.index
        val_idx = share_val.index

        # Per-specialty exogenous
        X_train, scaler = build_specialty_exogenous(g3.loc[train_idx],
                                                     name, fit_scaler=True)
        X_full, _ = build_specialty_exogenous(g3, name, scaler=scaler)

        print(f"  Train: {len(share_train)}  Val: {len(share_val)}  "
              f"X cols: {X_train.shape[1]}")

        # ARIMA on share
        try:
            arima_pred, order_a = fit_arima_share(share_train, val_idx, share)
            arima_pred["actual"] = share.loc[arima_pred.index]
            arima_metrics = score(arima_pred["actual"], arima_pred["predicted"])
            arima_pred.to_csv(out_pred / f"task2_{name}_arima.csv")
            rows.append({"specialty": name, "resolution": "daily",
                          "model": "arima", "block": "val", **arima_metrics,
                          "n_train": len(share_train), "n_val": len(share_val),
                          "order": str(order_a)})
            print(f"  ARIMA{order_a}: MAPE={arima_metrics['MAPE']:.3f} "
                  f"MAE={arima_metrics['MAE']:.4f} R2={arima_metrics['R2']:+.3f}")
        except Exception as exc:
            print(f"  ARIMA FAILED: {exc}")

        # SARIMAX on share with exog
        try:
            sarimax_pred, order_s, seasonal_s, base = fit_sarimax_share(
                share_train, X_train, X_full, val_idx, share)
            sarimax_pred["actual"] = share.loc[sarimax_pred.index]
            sarimax_metrics = score(sarimax_pred["actual"],
                                      sarimax_pred["predicted"])
            sarimax_pred.to_csv(out_pred / f"task2_{name}_sarimax.csv")
            rows.append({"specialty": name, "resolution": "daily",
                          "model": "sarimax", "block": "val",
                          **sarimax_metrics,
                          "n_train": len(share_train), "n_val": len(share_val),
                          "order": str(order_s),
                          "seasonal_order": str(seasonal_s)})
            print(f"  SARIMAX{order_s}x{seasonal_s}: "
                  f"MAPE={sarimax_metrics['MAPE']:.3f} "
                  f"MAE={sarimax_metrics['MAE']:.4f} "
                  f"R2={sarimax_metrics['R2']:+.3f}")

            # Save Surgery SARIMAX coefficients for Figure 6.9
            if name == "Surgery":
                from src.forecasting.models.sarimax import extract_coefficients
                coef_df = extract_coefficients(base, list(X_train.columns))
                coef_df.to_csv(out_metrics / "task2_surgery_sarimax_coefficients.csv",
                                index=False)
        except Exception as exc:
            print(f"  SARIMAX FAILED: {exc}")

        # XGBoost on share
        try:
            xgb_pred, xgb_params = fit_xgb_share(share_train, X_train,
                                                  X_full, val_idx)
            xgb_pred["actual"] = share.loc[xgb_pred.index]
            xgb_metrics = score(xgb_pred["actual"], xgb_pred["predicted"])
            xgb_pred.to_csv(out_pred / f"task2_{name}_xgboost.csv")
            rows.append({"specialty": name, "resolution": "daily",
                          "model": "xgboost", "block": "val",
                          **xgb_metrics,
                          "n_train": len(share_train), "n_val": len(share_val)})
            print(f"  XGBoost: MAPE={xgb_metrics['MAPE']:.3f} "
                  f"MAE={xgb_metrics['MAE']:.4f} R2={xgb_metrics['R2']:+.3f}")
            abs_pred_by_specialty[name] = xgb_pred["predicted"].copy()
        except Exception as exc:
            print(f"  XGBoost FAILED: {exc}")

        print(f"  Took {time.time() - t0:.1f}s")

    # ------------ Weekly specialties ------------
    for name, col in WEEKLY_SPECIALTIES.items():
        try:
            row = run_weekly_specialty(g3, splits, name, col)
            rows.append(row)
        except Exception as exc:
            print(f"  WEEKLY {name} FAILED: {exc}")

    # ------------ Aggregate metrics ------------
    metrics_df = pd.DataFrame(rows)
    metrics_path = out_metrics / "task2_per_specialty_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path.relative_to(ROOT)}")

    # ------------ Sum-consistency check (post hoc per §3.5.10) ------------
    if abs_pred_by_specialty:
        print("\n=== Sum-consistency (post hoc per §3.5.10) ===")
        # Sum predicted shares across the 5 daily specialties; check vs 1.0
        shares_df = pd.DataFrame(abs_pred_by_specialty)
        shares_df["sum"] = shares_df.sum(axis=1)
        deviation = (shares_df["sum"] - 1.0).abs().mean()
        print(f"  Mean absolute deviation of summed shares from 1.0: "
              f"{deviation:.4f}")
        sum_path = out_metrics / "task2_sum_consistency.csv"
        shares_df.to_csv(sum_path)
        print(f"  Wrote: {sum_path.relative_to(ROOT)}")

    # ------------ Figures ------------
    try:
        build_figures(metrics_df, out_pred)
    except Exception as exc:
        print(f"\nFigure generation failed: {exc}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_figures(metrics_df: pd.DataFrame, pred_dir: Path) -> None:
    import matplotlib.pyplot as plt

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

    # Figure 6.8 — per-specialty MAPE heatmap
    pivot = metrics_df[metrics_df["block"] == "val"].pivot_table(
        index="specialty", columns="model", values="MAPE", aggfunc="first",
    )
    spec_order = list(DAILY_SPECIALTIES.keys()) + list(WEEKLY_SPECIALTIES.keys())
    pivot = pivot.reindex(spec_order)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    capped = np.minimum(pivot.values, 50)
    im = ax.imshow(capped, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=50)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            cell_label = f"{v:.1f}" + ("†" if v > 50 else "")
            ax.text(j, i, cell_label, ha="center", va="center",
                     color="white" if v > 30 else "black", fontsize=9)
    ax.set_title("Figure 6.8 — Per-specialty val MAPE (%) by model",
                  loc="left", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Val MAPE (%, capped at 50)")
    out = ROOT / "artefacts" / "figures" / "fig_6_8_per_specialty_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  Wrote: {out.relative_to(ROOT)}")

    # Figure 6.9 — Surgery sign-reversal validation
    coef_path = ROOT / "artefacts" / "metrics" / "task2_surgery_sarimax_coefficients.csv"
    if coef_path.exists():
        coefs = pd.read_csv(coef_path)
        # Pick rows for Surgery sign-reversal columns
        target_cols = ["surgery_is_weekend", "surgery_is_long_weekend",
                        "surgery_is_public_holiday"]
        sub = coefs[coefs["feature"].isin(target_cols)].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            colours = [ROSE if c > 0 else NAVY for c in sub["coef"]]
            ax.barh(sub["feature"], sub["coef"], color=colours,
                    xerr=sub["std_err"].fillna(0), edgecolor="white",
                    error_kw={"ecolor": NEUTRAL, "capsize": 4})
            ax.axvline(0, color=NEUTRAL, linewidth=1, alpha=0.6)
            for i, (c, p, name) in enumerate(zip(sub["coef"], sub["p_value"],
                                                  sub["feature"])):
                mark = "*" if p < 0.05 else ""
                ax.text(c, i, f" {c:+.3f}{mark}", va="center",
                         fontsize=10, color=NEUTRAL)
            ax.set_title("Figure 6.9 — Surgery sign-reversal: SARIMAX exogenous "
                          "coefficients on weekend / long weekend / public holiday "
                          "(positive = arrivals UP on those days, validating §5.3.3)",
                          loc="left", fontsize=11)
            ax.set_xlabel("Coefficient on share-of-header target (positive bars in ROSE)")
            out_sr = ROOT / "artefacts" / "figures" / "fig_6_9_surgery_sign_reversal.png"
            plt.tight_layout()
            plt.savefig(out_sr)
            plt.close()
            print(f"  Wrote: {out_sr.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
