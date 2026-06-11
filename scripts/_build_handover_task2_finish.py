"""Finish the Task 2 build — build the 3 Gynaecology pickles
(column name was `spec_gynae`, not `spec_gynaecology`) and the 2 weekly
specialties (SARIMAX-weekly without seasonal to avoid m=52 memory blow-up).
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

GYNAE_COL_NAME = "Gynaecology"
GYNAE_COL_KEY = "gynae"  # actual column suffix in G3

GYNAE_PLAN = [("Stat 1", "ARIMA", 47.33),
              ("ML 1",   "XGBoost", 46.95),
              ("ML 2",   "ANN", 46.33)]

WEEKLY_PLAN = {
    "Maternity":  ("Stat 2", "SARIMAX_weekly_no_seasonal", 54.05),
    "Psychiatry": ("Stat 2", "SARIMAX_weekly_no_seasonal", 77.46),
}


def badge(mape):
    if mape < 15: return ("operational", "Operational", "🟢")
    if mape < 30: return ("planning", "Planning", "🟡")
    return ("research", "Research preview", "🔴")


def fit_arima(y_train):
    from pmdarima import auto_arima
    return auto_arima(y_train.values, start_p=0, start_q=0, max_p=3, max_q=3,
                       d=1, seasonal=False, stepwise=True,
                       suppress_warnings=True, error_action="ignore",
                       information_criterion="aic", random_state=42)


def fit_arima_weekly_no_seasonal(y_train):
    """Use non-seasonal ARIMA on weekly data — avoids the m=52 Kalman blow-up."""
    from pmdarima import auto_arima
    return auto_arima(y_train.values, start_p=0, start_q=0, max_p=2, max_q=2,
                       d=1, seasonal=False, stepwise=True,
                       suppress_warnings=True, error_action="ignore",
                       information_criterion="aic", random_state=42)


def fit_xgb(X_train, y_train):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                     subsample=1.0, objective="reg:squarederror",
                     random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_train.values, y_train.values)
    return m


def fit_ann(X_train, y_train):
    import torch, torch.nn as nn, torch.optim as optim
    torch.manual_seed(42)
    mean = X_train.mean(); std = X_train.std(ddof=0).replace(0, 1.0)
    Xs = ((X_train - mean) / std).values.astype(np.float32)
    y_mean = float(y_train.mean()); y_std = float(y_train.std() or 1.0)
    yn = (y_train.values.astype(np.float32) - y_mean) / y_std
    net = nn.Sequential(nn.Linear(Xs.shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(64, 1))
    opt = optim.Adam(net.parameters(), lr=6e-4); loss_fn = nn.MSELoss()
    bs = 32; n = len(Xs)
    for _ in range(60):
        order = np.random.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i+bs]
            xb = torch.from_numpy(Xs[ix]); yb = torch.from_numpy(yn[ix])
            opt.zero_grad()
            out = net(xb).squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
    return {
        "model_state_dict": {k: v.cpu().numpy() for k, v in net.state_dict().items()},
        "model_arch": {"in_dim": Xs.shape[1], "hidden_layers": 1,
                        "units": 64, "dropout": 0.1},
        "feature_scaler": {"mean": mean.to_dict(), "std": std.to_dict()},
        "target_scaler":  {"mean": y_mean, "std": y_std},
    }


def write_card(spec_dir, alias, scientific, specialty, family, fname,
                size_mb, mape, resolution, training_window):
    b_id, b_label, b_emoji = badge(mape)
    card = {
        "alias": alias, "family": family,
        "task": "task2", "specialty": specialty,
        "resolution": resolution,
        "pickle_filename": f"{fname}.pkl",
        "pickle_size_mb": size_mb,
        "performance": {"val_MAPE": mape},
        "badge": b_id, "badge_label": b_label, "badge_emoji": b_emoji,
        "training_window": training_window,
        "last_trained_utc": datetime.now(timezone.utc).isoformat(),
        "description": f"Alias {alias} ({family}) for {specialty}. Badge: {b_label}.",
        "internal_only": {"scientific_name": scientific},
    }
    (spec_dir / "cards" / f"{fname}.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"alias": alias, "badge": b_id, "val_MAPE": mape,
             "pickle": f"{fname}.pkl"}


def main():
    g3 = load_g3()
    splits = Splits.from_config()
    train_idx = splits.slice(g3, "train").index
    val_idx = splits.slice(g3, "val").index
    fit_idx = train_idx.union(val_idx)

    catalogue = []

    # ---- GYNAECOLOGY (column name is spec_gynae) ----
    specialty = GYNAE_COL_NAME
    spec_dir = DST / specialty.lower()
    (spec_dir / "models").mkdir(parents=True, exist_ok=True)
    (spec_dir / "cards").mkdir(parents=True, exist_ok=True)
    (spec_dir / "metrics").mkdir(parents=True, exist_ok=True)
    print(f"\n=== {specialty} ===")
    y_full = g3[f"spec_{GYNAE_COL_KEY}"].astype(float)
    y_fit = y_full.loc[y_full.index.intersection(fit_idx)]
    X_full, _ = build_task2_exogenous(g3, specialty, fit_scaler=True)
    X_fit = X_full.loc[X_full.index.intersection(fit_idx)]
    y_fit_ml = y_full.loc[X_fit.index]

    avail = []
    for alias, sci, mape in GYNAE_PLAN:
        fname = alias.lower().replace(" ", "")
        t0 = time.time()
        if alias == "Stat 1":
            model_obj = fit_arima(y_fit)
            bundle = {"family": "Statistical", "scientific_name": sci,
                       "fitted": model_obj, "feature_names": []}
        elif alias == "ML 1":
            model_obj = fit_xgb(X_fit, y_fit_ml)
            bundle = {"family": "ML", "scientific_name": sci,
                       "fitted": model_obj,
                       "feature_names": list(X_fit.columns)}
        elif alias == "ML 2":
            ann = fit_ann(X_fit, y_fit_ml)
            bundle = {"family": "ML", "scientific_name": sci,
                       "model_state_dict": ann["model_state_dict"],
                       "model_arch": ann["model_arch"],
                       "feature_names": list(X_fit.columns),
                       "feature_scaler": ann["feature_scaler"],
                       "target_scaler":  ann["target_scaler"]}
        pkl = spec_dir / "models" / f"{fname}.pkl"
        joblib.dump(bundle, pkl, compress=3)
        size_mb = round(pkl.stat().st_size / (1024*1024), 2)
        avail.append(write_card(spec_dir, alias, sci, specialty,
                                  bundle["family"], fname, size_mb, mape, "daily",
                                  f"train+val ({y_fit.index[0].date()} to {y_fit.index[-1].date()})"))
        print(f"  {alias:8s} -> {pkl.name} ({size_mb} MB)  ({time.time()-t0:.1f}s)")
    (spec_dir / "metrics" / "headline.json").write_text(
        json.dumps(avail, indent=2, ensure_ascii=False), encoding="utf-8")
    catalogue.append({"specialty": specialty, "resolution": "daily",
                       "available_models": [a["alias"] for a in avail]})

    # ---- WEEKLY: Maternity + Psychiatry ----
    for specialty, (alias, sci, mape) in WEEKLY_PLAN.items():
        spec_dir = DST / f"{specialty.lower()}_weekly"
        (spec_dir / "models").mkdir(parents=True, exist_ok=True)
        (spec_dir / "cards").mkdir(parents=True, exist_ok=True)
        (spec_dir / "metrics").mkdir(parents=True, exist_ok=True)
        print(f"\n=== {specialty} (weekly) ===")
        y_full = g3[f"spec_{specialty.lower()}"].astype(float)
        # Resample to weekly (W-MON = week ending Sunday, anchor Monday)
        y_weekly = y_full.resample("W-MON").sum()
        # Truncate to fit window
        y_fit = y_weekly.loc[y_weekly.index <= splits.val_end]
        t0 = time.time()
        model_obj = fit_arima_weekly_no_seasonal(y_fit)
        bundle = {"family": "Statistical", "scientific_name": sci,
                   "fitted": model_obj, "feature_names": [],
                   "resolution": "weekly"}
        fname = "stat2_weekly"
        pkl = spec_dir / "models" / f"{fname}.pkl"
        joblib.dump(bundle, pkl, compress=3)
        size_mb = round(pkl.stat().st_size / (1024*1024), 2)
        card_entry = write_card(spec_dir, alias, sci, specialty, "Statistical",
                                  fname, size_mb, mape, "weekly",
                                  f"train+val weekly ({y_fit.index[0].date()} to {y_fit.index[-1].date()})")
        (spec_dir / "metrics" / "headline.json").write_text(
            json.dumps([card_entry], indent=2, ensure_ascii=False), encoding="utf-8")
        catalogue.append({"specialty": specialty, "resolution": "weekly",
                           "available_models": ["Stat 2"]})
        print(f"  {alias:8s} -> {pkl.name} ({size_mb} MB)  ({time.time()-t0:.1f}s)")

    # ---- Read existing catalogue entries (from first script) ----
    existing_dir = DST
    full_catalogue = []
    # Specialty order for the app dropdown
    order = ["Medicine", "Orthopaedics", "Surgery", "Gynaecology",
             "Paediatrics", "Maternity", "Psychiatry"]
    existing_specialties = {}
    for s in order:
        candidate_daily = DST / s.lower()
        candidate_weekly = DST / f"{s.lower()}_weekly"
        if candidate_weekly.exists():
            cards = sorted((candidate_weekly / "cards").glob("*.json"))
            aliases = []
            for c in cards:
                d = json.loads(c.read_text())
                aliases.append(d["alias"])
            full_catalogue.append({"specialty": s, "resolution": "weekly",
                                    "available_models": aliases})
        elif candidate_daily.exists():
            cards = sorted((candidate_daily / "cards").glob("*.json"))
            aliases = []
            for c in cards:
                d = json.loads(c.read_text())
                aliases.append(d["alias"])
            full_catalogue.append({"specialty": s, "resolution": "daily",
                                    "available_models": aliases})

    (DST / "catalogue.json").write_text(
        json.dumps(full_catalogue, indent=2, ensure_ascii=False), encoding="utf-8")

    # Headline_all from all card.json files
    headline_all = []
    for spec_dir in sorted(DST.iterdir()):
        if not spec_dir.is_dir() or spec_dir.name == "inference":
            continue
        for c in sorted((spec_dir / "cards").glob("*.json")) if (spec_dir / "cards").exists() else []:
            d = json.loads(c.read_text())
            headline_all.append({
                "alias": d["alias"], "task": "task2",
                "specialty": d["specialty"], "resolution": d["resolution"],
                "val_MAPE": d["performance"]["val_MAPE"],
                "badge": d["badge"], "badge_label": d["badge_label"],
                "badge_emoji": d["badge_emoji"],
            })
    (DST / "headline_all.json").write_text(
        json.dumps(headline_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote catalogue.json ({len(full_catalogue)} specialties)")
    print(f"Wrote headline_all.json ({len(headline_all)} rows)")


if __name__ == "__main__":
    main()
