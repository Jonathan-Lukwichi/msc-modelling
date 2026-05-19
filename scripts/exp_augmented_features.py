"""Augmented-features experiment: do Tier-A+B external signals help?

Compares baseline models (existing features) vs. the same models trained
with the new external signals from artefacts/external_signals/g1_enriched.csv
appended:

  Parametric track (SARIMAX, NB-GLM): exog_raw10 + NEW_EXOG
  ML track (XGBoost):                 consensus_23 + NEW_EXOG

NEW_EXOG = 24 signals = Open-Meteo weather extras + SASSA + school terms +
           flu seasonal + COVID waves + RTMC road risk.

All models share the §5.5.2 train/val split + rolling-origin weekly refit
on val. SARIMAX uses the cached order; XGBoost uses cached best params.

Output: artefacts/metrics/exp_augmented_features.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.features import build_task1_exogenous
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.metrics import score
from src.forecasting.models.sarimax import (
    fit_with_order, rolling_forecast as sarimax_rolling,
)
from src.forecasting.models.negbin import rolling_forecast as nb_rolling
from src.forecasting.models.xgboost_m import rolling_forecast as xgb_rolling
try:
    from src.forecasting.models.ann import rolling_forecast as ann_rolling
    HAS_ANN = True
except ImportError as _exc:
    print(f"[note] ANN unavailable in this interpreter: {_exc}")
    ann_rolling = None
    HAS_ANN = False
try:
    from src.forecasting.models.lstm import rolling_forecast as lstm_rolling
    HAS_LSTM = True
except ImportError as _exc:
    print(f"[note] LSTM unavailable in this interpreter: {_exc}")
    lstm_rolling = None
    HAS_LSTM = False


ENRICHED = ROOT / "artefacts" / "external_signals" / "g1_enriched.csv"
SARIMAX_ORDER_CACHE = ROOT / "artefacts" / "models" / "sarimax_order.txt"
XGB_BEST_PARAMS = ROOT / "artefacts" / "models" / "xgboost_best_params.json"
ANN_BEST_PARAMS = ROOT / "artefacts" / "models" / "ann_best_params.json"
LSTM_BEST_PARAMS = ROOT / "artefacts" / "models" / "lstm_best_params.json"
OUT_METRICS = ROOT / "artefacts" / "metrics" / "exp_augmented_features.csv"


# Tier A + B external signals confirmed REAL in g1_enriched.csv.
# Excluded: Eskom (only 3 LS days), Google Trends (pytrends fail).
NEW_EXOG: list[str] = [
    # Weather extras (Open-Meteo) - G1 only had temp_mean_C + wind_max_kmh
    "temp_max_C", "temp_min_C",
    "humidity_mean_pct", "pressure_mean_hPa",
    "precip_sum_mm", "wind_gusts_kmh",
    "heat_wave_flag", "storm_flag", "heavy_rain_flag", "cold_snap_flag",
    "solar_MJ_per_m2",
    # SASSA payment-day effects
    "is_sassa_pay_day", "days_since_sassa_pay",
    # School calendar (term-level granularity beyond is_school_holiday)
    "is_exam_period", "term_number", "days_until_term_end",
    # Flu / respiratory load (NICD proxy + WHO + COVID waves)
    "flu_seasonal_index", "covid_wave_active", "respiratory_load_index",
    # Road safety / RTMC high-risk periods
    "rtmc_high_risk_flag", "rtmc_risk_intensity",
    "days_to_easter", "days_from_easter", "days_to_xmas",
]


def load_enriched_signals() -> pd.DataFrame:
    """Return just the new external signal columns, indexed by date."""
    df = pd.read_csv(ENRICHED, parse_dates=["date"]).set_index("date")
    missing = [c for c in NEW_EXOG if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing in g1_enriched.csv: {missing}")
    sig = df[NEW_EXOG].copy()
    # NaN policy: forward-fill weekly seasonal layers, zero-fill flags
    fill_zero = {"covid_wave_active", "heat_wave_flag", "storm_flag",
                 "heavy_rain_flag", "cold_snap_flag", "is_sassa_pay_day",
                 "is_exam_period", "rtmc_high_risk_flag"}
    for c in sig.columns:
        if c in fill_zero:
            sig[c] = sig[c].fillna(0).astype(int)
        else:
            sig[c] = sig[c].ffill().bfill()
    # Median-fill anything still NaN (days_to_easter etc. are NaN outside 30d window)
    sig = sig.fillna(sig.median(numeric_only=True))
    return sig


def standardize_for_glm(sig: pd.DataFrame, train_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Z-score the continuous augmented features on the train fold (binaries left alone)."""
    binary_cols = {"covid_wave_active", "heat_wave_flag", "storm_flag",
                   "heavy_rain_flag", "cold_snap_flag", "is_sassa_pay_day",
                   "is_exam_period", "rtmc_high_risk_flag"}
    out = sig.copy()
    for col in sig.columns:
        if col in binary_cols:
            continue
        m = sig.loc[train_idx, col].mean()
        s = sig.loc[train_idx, col].std(ddof=0) or 1.0
        out[col] = (sig[col] - m) / s
    return out


def load_sarimax_order() -> tuple[tuple, tuple]:
    cfg: dict[str, str] = {}
    for line in SARIMAX_ORDER_CACHE.read_text().strip().splitlines():
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    return eval(cfg["order"]), eval(cfg["seasonal_order"])


def read_baseline_mape(csv_name: str) -> float | None:
    """Read val MAPE from an existing metrics CSV."""
    p = ROOT / "artefacts" / "metrics" / csv_name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    val_row = df[df["block"] == "val"] if "block" in df.columns else df
    if val_row.empty or "MAPE" not in val_row.columns:
        return None
    return float(val_row["MAPE"].iloc[0])


# -------------------------------------------------------------------------
# Per-model experiment functions
# -------------------------------------------------------------------------

def run_sarimax_augmented(target: pd.Series, splits, sig: pd.DataFrame) -> dict:
    print("\n[SARIMAX augmented] building exog + fitting...")
    train_idx = splits.slice(pd.DataFrame(index=target.index), "train").index
    val_idx = splits.slice(pd.DataFrame(index=target.index), "val").index

    X_train_raw, scaler = build_task1_exogenous(
        load_g1().loc[train_idx], fit_scaler=True
    )
    X_full_raw, _ = build_task1_exogenous(load_g1(), scaler=scaler)

    # Augment with new signals
    sig_z = standardize_for_glm(sig, train_idx)
    X_train_aug = pd.concat([X_train_raw, sig_z.loc[train_idx]], axis=1)
    X_full_aug = pd.concat([X_full_raw, sig_z.loc[X_full_raw.index]], axis=1)
    print(f"  exog matrix: {X_train_aug.shape[1]} cols "
          f"(raw 10 = {X_train_raw.shape[1]} + new = {sig_z.shape[1]})")

    order, seasonal_order = load_sarimax_order()
    print(f"  using cached order {order} x {seasonal_order}")

    t0 = time.time()
    pred = sarimax_rolling(
        target, X_full_aug, val_idx,
        order=order, seasonal_order=seasonal_order, step_days=7,
    )
    print(f"  rolling forecast took {time.time() - t0:.1f}s ({len(pred)} rows)")

    pred = pred.set_index("date")
    pred["actual"] = target.loc[pred.index]
    metrics = score(pred["actual"], pred["predicted"])
    out_path = ROOT / "artefacts" / "predictions" / "sarimax_augmented.csv"
    pred.reset_index().to_csv(out_path, index=False)
    return {"model": "SARIMAX", "augmented": True, **metrics}


def run_nbglm_augmented(target: pd.Series, splits, sig: pd.DataFrame) -> dict:
    print("\n[NB-GLM augmented] building exog + fitting...")
    train_idx = splits.slice(pd.DataFrame(index=target.index), "train").index
    val_idx = splits.slice(pd.DataFrame(index=target.index), "val").index

    X_train_raw, scaler = build_task1_exogenous(
        load_g1().loc[train_idx], fit_scaler=True
    )
    X_full_raw, _ = build_task1_exogenous(load_g1(), scaler=scaler)

    sig_z = standardize_for_glm(sig, train_idx)
    X_full_aug = pd.concat([X_full_raw, sig_z.loc[X_full_raw.index]], axis=1)
    print(f"  exog matrix: {X_full_aug.shape[1]} cols")

    t0 = time.time()
    pred = nb_rolling(target, X_full_aug, val_idx, step_days=7)
    print(f"  rolling forecast took {time.time() - t0:.1f}s ({len(pred)} rows)")

    pred = pred.set_index("date")
    pred["actual"] = target.loc[pred.index]
    metrics = score(pred["actual"], pred["predicted"])
    out_path = ROOT / "artefacts" / "predictions" / "nbglm_augmented.csv"
    pred.reset_index().to_csv(out_path, index=False)
    return {"model": "NB-GLM", "augmented": True, **metrics}


def run_xgb_augmented(target: pd.Series, splits, sig: pd.DataFrame) -> dict:
    print("\n[XGBoost augmented] building features + fitting...")
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    print(f"  consensus base: {X_consensus.shape[1]} cols")

    # Inner-join target (zero-day-filtered), consensus features (not filtered),
    # and new signals (zero-day-filtered) so all three share the same date set.
    df_joined = pd.concat(
        [target.rename("y"), X_consensus, sig],
        axis=1, join="inner",
    ).dropna()
    y = df_joined["y"]
    X_full_aug = df_joined.drop(columns=["y"])
    print(f"  augmented ML matrix: {X_full_aug.shape}")

    val_idx = splits.slice(pd.DataFrame(index=y.index), "val").index

    best_params = json.loads(XGB_BEST_PARAMS.read_text())
    print(f"  using cached best params: {best_params}")

    t0 = time.time()
    pred = xgb_rolling(X_full_aug, y, val_idx, params=best_params,
                       step_days=7, seed=42)
    print(f"  rolling forecast took {time.time() - t0:.1f}s ({len(pred)} rows)")

    pred = pred.set_index("date")
    pred["actual"] = y.loc[pred.index]
    metrics = score(pred["actual"], pred["predicted"])
    out_path = ROOT / "artefacts" / "predictions" / "xgboost_augmented.csv"
    pred.reset_index().to_csv(out_path, index=False)
    return {"model": "XGBoost", "augmented": True, **metrics}


def _ml_inner_join(target, X_consensus, sig):
    """Inner-join target / consensus / new signals on shared dates + dropna."""
    df_joined = pd.concat(
        [target.rename("y"), X_consensus, sig], axis=1, join="inner"
    ).dropna()
    return df_joined["y"], df_joined.drop(columns=["y"])


def run_ann_augmented(target, splits, sig) -> dict:
    print("\n[ANN augmented] building features + fitting...")
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    y, X_full_aug = _ml_inner_join(target, X_consensus, sig)
    print(f"  augmented ML matrix: {X_full_aug.shape}")

    val_idx = splits.slice(pd.DataFrame(index=y.index), "val").index
    params = json.loads(ANN_BEST_PARAMS.read_text())
    print(f"  using cached best params: {params}")
    t0 = time.time()
    pred = ann_rolling(X_full_aug, y, val_idx, params=params, step_days=7, seed=42)
    print(f"  rolling forecast took {time.time() - t0:.1f}s ({len(pred)} rows)")
    pred = pred.set_index("date")
    pred["actual"] = y.loc[pred.index]
    metrics = score(pred["actual"], pred["predicted"])
    out_path = ROOT / "artefacts" / "predictions" / "ann_augmented.csv"
    pred.reset_index().to_csv(out_path, index=False)
    return {"model": "ANN", "augmented": True, **metrics}


def run_lstm_augmented(target, splits, sig) -> dict:
    print("\n[LSTM augmented] building features + fitting...")
    eng = load_engineered()
    X_consensus = build_selected_X(eng)
    y, X_full_aug = _ml_inner_join(target, X_consensus, sig)
    print(f"  augmented ML matrix: {X_full_aug.shape}")

    val_idx = splits.slice(pd.DataFrame(index=y.index), "val").index
    params = json.loads(LSTM_BEST_PARAMS.read_text())
    print(f"  using cached best params: {params}")
    t0 = time.time()
    pred = lstm_rolling(X_full_aug, y, val_idx, params=params, step_days=7, seed=42)
    print(f"  rolling forecast took {time.time() - t0:.1f}s ({len(pred)} rows)")
    pred = pred.set_index("date")
    pred["actual"] = y.loc[pred.index]
    metrics = score(pred["actual"], pred["predicted"])
    out_path = ROOT / "artefacts" / "predictions" / "lstm_augmented.csv"
    pred.reset_index().to_csv(out_path, index=False)
    return {"model": "LSTM", "augmented": True, **metrics}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def reuse_prediction(name: str, target: pd.Series) -> dict | None:
    """Re-score an already-saved augmented prediction CSV (skips re-fit)."""
    p = ROOT / "artefacts" / "predictions" / f"{name}_augmented.csv"
    if not p.exists():
        return None
    pred = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    if "actual" not in pred.columns:
        pred["actual"] = target.loc[pred.index]
    m = score(pred["actual"], pred["predicted"])
    return {"model": name.upper() if name != "nbglm" else "NB-GLM", **m}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        help="comma-list: sarimax, nbglm, xgboost  (or 'all')")
    parser.add_argument("--reuse-cached", action="store_true",
                        help="If an augmented predictions CSV already exists, "
                             "re-score it instead of re-fitting.")
    args = parser.parse_args()
    pick = {m.strip().lower() for m in args.models.split(",")} \
           if args.models != "all" else {"sarimax", "nbglm", "xgboost", "ann", "lstm"}

    print("=" * 70)
    print("Augmented-features experiment (Tier A + B real signals)")
    print("=" * 70)
    print(f"New features added: {len(NEW_EXOG)}")
    print(f"Models to run     : {sorted(pick)}")

    splits = Splits.from_config()
    g1 = load_g1()
    target = g1["total_daily_arrivals"]
    sig = load_enriched_signals()
    print(f"Loaded enriched signals: {sig.shape}")

    results: list[dict] = []

    runners = [
        ("sarimax", "sarimax", run_sarimax_augmented, "SARIMAX"),
        ("nbglm",   "nbglm",   run_nbglm_augmented,   "NB-GLM"),
        ("xgboost", "xgboost", run_xgb_augmented,     "XGBoost"),
        ("ann",     "ann",     run_ann_augmented,     "ANN"),
        ("lstm",    "lstm",    run_lstm_augmented,    "LSTM"),
    ]
    for key, fname, runner, label in runners:
        if key not in pick:
            continue
        if key == "ann" and not HAS_ANN:
            print(f"[{label}] skipped — torch not available in this interpreter")
            continue
        if key == "lstm" and not HAS_LSTM:
            print(f"[{label}] skipped — torch not available in this interpreter")
            continue
        if args.reuse_cached:
            cached = reuse_prediction(fname, target)
            if cached is not None:
                cached["model"] = label
                print(f"[{label}] reusing cached predictions -> "
                      f"MAPE={cached['MAPE']:.2f}%")
                results.append({**cached, "augmented": True})
                continue
        try:
            results.append(runner(target, splits, sig))
        except Exception as exc:
            import traceback
            print(f"[{label}] FAILED: {exc}")
            traceback.print_exc()

    # ---- Comparison vs baselines ----
    baseline_map = {
        "SARIMAX": read_baseline_mape("sarimax_metrics.csv"),
        "NB-GLM":  read_baseline_mape("nbglm_metrics.csv"),
        "XGBoost": read_baseline_mape("xgboost_metrics.csv"),
        "ANN":     read_baseline_mape("ann_metrics.csv"),
        "LSTM":    read_baseline_mape("lstm_metrics.csv"),
        "ARIMA":   read_baseline_mape("arima_metrics.csv"),
    }

    rows = []
    for r in results:
        bl = baseline_map.get(r["model"])
        rows.append({
            "model": r["model"],
            "baseline_MAPE": bl,
            "augmented_MAPE": r["MAPE"],
            "delta_pp": (r["MAPE"] - bl) if bl is not None else None,
            "augmented_MAE": r["MAE"],
            "augmented_RMSE": r["RMSE"],
            "augmented_R2": r["R2"],
        })
    # Add baselines for models we did not re-run (ANN, LSTM, ARIMA)
    for m, bl in baseline_map.items():
        if m in {r["model"] for r in results} or bl is None:
            continue
        rows.append({
            "model": m, "baseline_MAPE": bl, "augmented_MAPE": None,
            "delta_pp": None, "augmented_MAE": None,
            "augmented_RMSE": None, "augmented_R2": None,
        })
    table = pd.DataFrame(rows)

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_METRICS, index=False)
    print(f"\nWrote: {OUT_METRICS.relative_to(ROOT)}")

    print("\n" + "=" * 70)
    print("Comparison: baseline vs augmented MAPE")
    print("=" * 70)
    fmt = {
        "baseline_MAPE":  lambda v: f"{v:6.2f}" if pd.notna(v) else "  n/a ",
        "augmented_MAPE": lambda v: f"{v:6.2f}" if pd.notna(v) else "  n/a ",
        "delta_pp":       lambda v: f"{v:+5.2f}" if pd.notna(v) else "  -- ",
    }
    print(table.to_string(index=False, formatters=fmt,
                           columns=["model", "baseline_MAPE",
                                    "augmented_MAPE", "delta_pp"]))


if __name__ == "__main__":
    main()
