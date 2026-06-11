"""Build the Task 2 portion of the handover_to_webapp/ package.

For each (specialty, alias) pair where the alias produces a useful forecast,
refit the model on train+val and write a portable pickle + card.json. Also
generate the per-specialty headline.json and the catalogue.json that the
web app uses to filter the model dropdown per specialty.

Run once. Idempotent.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g3, Splits
from src.forecasting.features import build_task2_exogenous

DST = ROOT / "artefacts" / "handover_to_webapp" / "task2_specialties"

# Per-specialty deployable models from Phase 1c+ Task 2 metrics
# (only the ones below the badge threshold; NB-GLM excluded per thesis scope)
ALIAS_TO_SCI = {
    "Stat 1": "ARIMA",        "Stat 2": "SARIMAX",
    "ML 1":   "XGBoost",      "ML 2":   "ANN",
}

DAILY_PLAN = {
    "Medicine":    [("Stat 1", "ARIMA", 21.37), ("ML 1", "XGBoost", 21.61),
                     ("ML 2", "ANN", 22.49)],
    "Orthopaedics":[("Stat 1", "ARIMA", 84.55)],
    "Surgery":     [("ML 1", "XGBoost", 54.24), ("ML 2", "ANN", 53.57)],
    "Gynaecology": [("Stat 1", "ARIMA", 47.33), ("ML 1", "XGBoost", 46.95),
                     ("ML 2", "ANN", 46.33)],
    "Paediatrics": [("Stat 1", "ARIMA", 54.92), ("ML 2", "ANN", 55.02)],
}

# Weekly-only specialties (≤2/day)
WEEKLY_PLAN = {
    "Maternity":   [("Stat 2", "SARIMAX_weekly", 54.05)],
    "Psychiatry":  [("Stat 2", "SARIMAX_weekly", 77.46)],
}


def badge(mape: float) -> tuple[str, str, str]:
    if mape < 15:
        return ("operational", "Operational", "🟢")
    if mape < 30:
        return ("planning", "Planning", "🟡")
    return ("research", "Research preview", "🔴")


def specialty_dirname(spec: str, weekly: bool) -> str:
    suffix = "_weekly" if weekly else ""
    return f"{spec.lower()}{suffix}"


def alias_to_filename(alias: str) -> str:
    return alias.lower().replace(" ", "")


def fit_arima(y_train: pd.Series):
    from pmdarima import auto_arima
    return auto_arima(y_train.values, start_p=0, start_q=0, max_p=3, max_q=3,
                       d=1, seasonal=False, stepwise=True,
                       suppress_warnings=True, error_action="ignore",
                       information_criterion="aic", trace=False, random_state=42)


def fit_sarimax_weekly(y_weekly: pd.Series, X_weekly: pd.DataFrame | None):
    from pmdarima import auto_arima
    if X_weekly is not None and not X_weekly.empty:
        return auto_arima(y_weekly.values, X=X_weekly.values,
                           start_p=0, start_q=0, max_p=2, max_q=2, d=1,
                           seasonal=True, m=52,
                           start_P=0, start_Q=0, max_P=1, max_Q=1, D=1,
                           stepwise=True, suppress_warnings=True,
                           error_action="ignore",
                           information_criterion="aic",
                           random_state=42)
    return auto_arima(y_weekly.values, start_p=0, start_q=0, max_p=2, max_q=2,
                       d=1, seasonal=True, m=52,
                       start_P=0, start_Q=0, max_P=1, max_Q=1, D=1,
                       stepwise=True, suppress_warnings=True,
                       error_action="ignore",
                       information_criterion="aic", random_state=42)


def fit_xgb(X_train, y_train, params=None):
    from xgboost import XGBRegressor
    p = params or {"n_estimators": 300, "max_depth": 5,
                   "learning_rate": 0.05, "subsample": 1.0}
    m = XGBRegressor(**p, objective="reg:squarederror", random_state=42,
                     n_jobs=-1, verbosity=0)
    m.fit(X_train.values, y_train.values)
    return m


def fit_ann(X_train, y_train, params=None):
    import torch, torch.nn as nn, torch.optim as optim
    p = params or {"hidden_layers": 1, "units": 64, "dropout": 0.1,
                   "learning_rate": 0.0006, "batch_size": 32}
    torch.manual_seed(42)
    mean = X_train.mean(); std = X_train.std(ddof=0).replace(0, 1.0)
    Xs = ((X_train - mean) / std).values.astype(np.float32)
    y_mean = float(y_train.mean())
    y_std  = float(y_train.std() or 1.0)
    yn = (y_train.values.astype(np.float32) - y_mean) / y_std
    layers = []; last = Xs.shape[1]
    for _ in range(p["hidden_layers"]):
        layers += [nn.Linear(last, p["units"]), nn.ReLU(),
                   nn.Dropout(p["dropout"])]
        last = p["units"]
    layers.append(nn.Linear(last, 1))
    net = nn.Sequential(*layers)
    opt = optim.Adam(net.parameters(), lr=p["learning_rate"])
    loss_fn = nn.MSELoss()
    bs = p["batch_size"]; n = len(Xs)
    for epoch in range(60):
        order = np.random.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i+bs]
            xb = torch.from_numpy(Xs[ix])
            yb = torch.from_numpy(yn[ix])
            opt.zero_grad()
            out = net(xb).squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
    return {
        "model_state_dict": {k: v.cpu().numpy()
                              for k, v in net.state_dict().items()},
        "model_arch": {"in_dim": Xs.shape[1],
                        "hidden_layers": p["hidden_layers"],
                        "units": p["units"], "dropout": p["dropout"]},
        "feature_scaler": {"mean": mean.to_dict(), "std": std.to_dict()},
        "target_scaler":  {"mean": y_mean, "std": y_std},
    }


def build_y_for_specialty(g3: pd.DataFrame, specialty: str) -> pd.Series:
    col = f"spec_{specialty.lower()}"
    if col not in g3.columns:
        return pd.Series(dtype=float)
    return g3[col].astype(float)


def build_X_for_specialty(g3: pd.DataFrame, specialty: str):
    try:
        X, _ = build_task2_exogenous(g3, specialty, fit_scaler=True)
        return X
    except Exception:
        return None


def main() -> None:
    g3 = load_g3()
    splits = Splits.from_config()
    train_idx = splits.slice(g3, "train").index
    val_idx   = splits.slice(g3, "val").index
    fit_idx = train_idx.union(val_idx)
    print(f"Loaded G3: {g3.shape}, fit window {fit_idx[0].date()}-{fit_idx[-1].date()} ({len(fit_idx)} days)")

    catalogue = []
    headline_all = []

    # DAILY specialties
    for specialty, entries in DAILY_PLAN.items():
        spec_dir = DST / specialty_dirname(specialty, weekly=False)
        (spec_dir / "models").mkdir(parents=True, exist_ok=True)
        (spec_dir / "cards").mkdir(parents=True, exist_ok=True)
        (spec_dir / "metrics").mkdir(parents=True, exist_ok=True)

        avail = []
        for alias, scientific, mape in entries:
            fname = alias_to_filename(alias)
            print(f"\n[{specialty:14s}] {alias} ({scientific})  expected MAPE {mape:.1f}%")
            t0 = time.time()

            y_full = build_y_for_specialty(g3, specialty)
            if y_full.empty:
                print("  SKIP: target series not in G3"); continue
            y_fit = y_full.loc[y_full.index.intersection(fit_idx)]
            if alias == "Stat 1":
                model_obj = fit_arima(y_fit)
                bundle = {"family": "Statistical",
                           "scientific_name": scientific,
                           "fitted": model_obj,
                           "feature_names": [],
                           "feature_scaler": None,
                           "target_scaler": None,
                           "training_index_first": str(y_fit.index[0].date()),
                           "training_index_last":  str(y_fit.index[-1].date())}
            elif alias in ("ML 1", "ML 2"):
                X_full = build_X_for_specialty(g3, specialty)
                if X_full is None:
                    print("  SKIP: cannot build exogenous matrix"); continue
                X_fit = X_full.loc[X_full.index.intersection(fit_idx)]
                y_fit_ml = y_full.loc[X_fit.index]
                if alias == "ML 1":
                    model_obj = fit_xgb(X_fit, y_fit_ml)
                    bundle = {"family": "ML",
                               "scientific_name": scientific,
                               "fitted": model_obj,
                               "feature_names": list(X_fit.columns),
                               "feature_scaler": None,
                               "target_scaler": None}
                else:
                    ann = fit_ann(X_fit, y_fit_ml)
                    bundle = {"family": "ML",
                               "scientific_name": scientific,
                               "model_state_dict": ann["model_state_dict"],
                               "model_arch": ann["model_arch"],
                               "feature_names": list(X_fit.columns),
                               "feature_scaler": ann["feature_scaler"],
                               "target_scaler":  ann["target_scaler"]}
            else:
                print(f"  SKIP unknown alias {alias}"); continue

            # Persist
            pkl = spec_dir / "models" / f"{fname}.pkl"
            joblib.dump(bundle, pkl, compress=3)
            size_mb = round(pkl.stat().st_size / (1024*1024), 2)

            b_id, b_label, b_emoji = badge(mape)
            card = {
                "alias": alias,
                "family": bundle["family"],
                "task": "task2",
                "specialty": specialty,
                "resolution": "daily",
                "pickle_filename": f"{fname}.pkl",
                "pickle_size_mb": size_mb,
                "performance": {"val_MAPE": mape},
                "badge": b_id,
                "badge_label": b_label,
                "badge_emoji": b_emoji,
                "training_window": f"train+val ({y_fit.index[0].date()} to {y_fit.index[-1].date()}, {len(y_fit)} days)",
                "last_trained_utc": datetime.now(timezone.utc).isoformat(),
                "description": (f"Alias {alias} ({bundle['family']}) for "
                                 f"{specialty}. Badge: {b_label}."),
                "internal_only": {"scientific_name": scientific},
            }
            (spec_dir / "cards" / f"{fname}.json").write_text(
                json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

            avail.append({"alias": alias, "badge": b_id,
                           "val_MAPE": mape, "pickle": f"{fname}.pkl"})
            headline_all.append({
                "alias": alias, "task": "task2", "specialty": specialty,
                "resolution": "daily", "val_MAPE": mape,
                "badge": b_id, "badge_label": b_label, "badge_emoji": b_emoji,
            })
            print(f"  done in {time.time()-t0:.1f}s  -> {pkl.name} ({size_mb} MB) {b_emoji}")

        (spec_dir / "metrics" / "headline.json").write_text(
            json.dumps(avail, indent=2, ensure_ascii=False), encoding="utf-8")
        catalogue.append({"specialty": specialty,
                           "resolution": "daily",
                           "available_models": [a["alias"] for a in avail]})

    # WEEKLY specialties (Maternity, Psychiatry)
    for specialty, entries in WEEKLY_PLAN.items():
        spec_dir = DST / specialty_dirname(specialty, weekly=True)
        (spec_dir / "models").mkdir(parents=True, exist_ok=True)
        (spec_dir / "cards").mkdir(parents=True, exist_ok=True)
        (spec_dir / "metrics").mkdir(parents=True, exist_ok=True)

        avail = []
        y_full = build_y_for_specialty(g3, specialty)
        if y_full.empty:
            print(f"\n[{specialty:14s}] SKIP — no target column"); continue
        y_weekly = y_full.resample("W-MON").sum()
        y_fit = y_weekly.loc[y_weekly.index <= splits.val_end]

        for alias, scientific, mape in entries:
            fname = alias_to_filename(alias) + "_weekly"
            print(f"\n[{specialty:14s}] {alias} ({scientific}) weekly  expected MAPE {mape:.1f}%")
            t0 = time.time()
            model_obj = fit_sarimax_weekly(y_fit, None)
            bundle = {
                "family": "Statistical",
                "scientific_name": scientific,
                "fitted": model_obj,
                "feature_names": [],
                "feature_scaler": None,
                "target_scaler": None,
                "resolution": "weekly",
                "training_index_first": str(y_fit.index[0].date()),
                "training_index_last":  str(y_fit.index[-1].date()),
            }
            pkl = spec_dir / "models" / f"{fname}.pkl"
            joblib.dump(bundle, pkl, compress=3)
            size_mb = round(pkl.stat().st_size / (1024*1024), 2)
            b_id, b_label, b_emoji = badge(mape)

            card = {
                "alias": alias, "family": "Statistical",
                "task": "task2", "specialty": specialty,
                "resolution": "weekly",
                "pickle_filename": f"{fname}.pkl",
                "pickle_size_mb": size_mb,
                "performance": {"val_MAPE": mape},
                "badge": b_id, "badge_label": b_label, "badge_emoji": b_emoji,
                "training_window": f"train+val weekly ({y_fit.index[0].date()} to {y_fit.index[-1].date()})",
                "last_trained_utc": datetime.now(timezone.utc).isoformat(),
                "description": (f"Alias {alias} for {specialty} at WEEKLY "
                                 f"resolution (≤2 patients/day average makes "
                                 f"daily forecasting infeasible). Badge: {b_label}."),
                "internal_only": {"scientific_name": scientific},
            }
            (spec_dir / "cards" / f"{fname}.json").write_text(
                json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

            avail.append({"alias": alias, "badge": b_id,
                           "val_MAPE": mape, "pickle": f"{fname}.pkl"})
            headline_all.append({
                "alias": alias, "task": "task2", "specialty": specialty,
                "resolution": "weekly", "val_MAPE": mape,
                "badge": b_id, "badge_label": b_label, "badge_emoji": b_emoji,
            })
            print(f"  done in {time.time()-t0:.1f}s  -> {pkl.name} ({size_mb} MB) {b_emoji}")

        (spec_dir / "metrics" / "headline.json").write_text(
            json.dumps(avail, indent=2, ensure_ascii=False), encoding="utf-8")
        catalogue.append({"specialty": specialty,
                           "resolution": "weekly",
                           "available_models": [a["alias"] for a in avail]})

    # Master catalogue + headline
    (DST / "catalogue.json").write_text(
        json.dumps(catalogue, indent=2, ensure_ascii=False), encoding="utf-8")
    (DST / "headline_all.json").write_text(
        json.dumps(headline_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {DST/'catalogue.json'} (7 specialties)")
    print(f"Wrote {DST/'headline_all.json'} ({len(headline_all)} rows)")


if __name__ == "__main__":
    main()
