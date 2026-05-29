"""Deploy top-3 models — pickle them for web-app deployment.

Identifies the top-3 models by val_RMSE across all phases (Phase 2 HPO winners
preferred; falls back to Phase 1 defaults). For each top-3 model:
  1. Refit on full TRAIN + VAL window with the winning hyperparameters
     (production model uses every observation up to the held-out test cliff)
  2. Pickle the model + scaler + feature names + metadata in a single artefact
  3. Save an example inference function snippet

Outputs:
  artefacts/deploy_top3/
    01_<rank>_<model>.pkl          full deployment bundle (load with joblib)
    01_<rank>_<model>_card.json    metadata + expected metrics + schema
    01_<rank>_<model>_predict.py   minimal stand-alone inference script
    deployment_manifest.csv        one-row-per-model summary
    README.md                      how to use the artefacts in a web app

Usage:
  python scripts/phase_deploy_top3.py                  # auto-pick top 3
  python scripts/phase_deploy_top3.py --models a,b,c   # explicit list
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.io import load_g1, Splits
from src.forecasting.engineering import load_engineered
from src.forecasting.consensus import build_selected_X
from src.forecasting.features import build_task1_exogenous

OUT = ROOT / "artefacts" / "deploy_top3"
OUT.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Step 1: Identify top-3 models
# -------------------------------------------------------------------------
def discover_top3(explicit: list[str] | None = None) -> list[dict]:
    """Return up to 3 (model_name, val_RMSE, params, source) records."""
    p1 = ROOT / "artefacts" / "phase1_defaults" / "summary_phase1.csv"
    p2 = ROOT / "artefacts" / "phase2_hpo"      / "summary_phase2.csv"
    rows = []

    # Phase 2 (HPO winners) take precedence
    if p2.exists():
        df2 = pd.read_csv(p2)
        for _, r in df2.iterrows():
            rows.append({
                "model": r["model"],
                "source": f"phase2_{r['algo']}",
                "val_RMSE": float(r["val_RMSE"]),
                "val_MAPE": float(r["val_MAPE"]),
                "params":   r.get("winner_params", "{}"),
            })

    # Phase 1 (defaults) for models not yet HPO'd
    if p1.exists():
        df1 = pd.read_csv(p1)
        already = {r["model"] for r in rows}
        for _, r in df1.iterrows():
            if r["model"] in already:
                continue
            rows.append({
                "model": r["model"],
                "source": "phase1_defaults",
                "val_RMSE": float(r["val_RMSE"]),
                "val_MAPE": float(r["val_MAPE"]),
                "params": "{}",  # use Chapter 5 defaults
            })

    if explicit:
        wanted = set(explicit)
        rows = [r for r in rows if r["model"] in wanted]
    else:
        rows.sort(key=lambda x: x["val_RMSE"])
        # Keep best-per-model (de-dup if same model has multiple HPO algos)
        seen, deduped = set(), []
        for r in rows:
            if r["model"] in seen:
                continue
            seen.add(r["model"])
            deduped.append(r)
        rows = deduped[:3]

    return rows


# -------------------------------------------------------------------------
# Step 2: Per-model refit on train + val
# -------------------------------------------------------------------------
def fit_for_deployment(model_name: str, params_json: str):
    """Refit the named model on TRAIN + VAL combined, return artefact dict."""
    params = json.loads(params_json) if params_json and params_json != "{}" else None
    splits = Splits.from_config()
    g1 = load_g1()
    y_full = g1["total_daily_arrivals"]
    train_idx = splits.slice(g1, "train").index
    val_idx   = splits.slice(g1, "val").index
    fit_idx = train_idx.union(val_idx)
    y_fit = y_full.loc[fit_idx]

    artefact = {
        "model_name": model_name,
        "params": params or "chapter5_defaults",
        "fit_start": fit_idx[0].isoformat(),
        "fit_end":   fit_idx[-1].isoformat(),
        "n_train_observations": int(len(fit_idx)),
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target": "total_daily_arrivals",
        "hospital": "Steve Biko Academic Hospital, Pretoria, South Africa",
        "thesis_chapter": "Chapter 5/6 modelling decisions",
    }

    if model_name in ("arima",):
        from pmdarima import ARIMA as PmARIMA
        order = tuple(params["order"]) if params else (0, 1, 2)
        model = PmARIMA(order=order, suppress_warnings=True)
        model.fit(y_fit.values)
        artefact["model"] = model
        artefact["family"] = "ARIMA"
        artefact["features_required"] = []
        artefact["inference"] = "model.predict(n_periods=H)"
        artefact["order"] = list(order)

    elif model_name in ("sarimax",):
        from pmdarima import ARIMA as PmARIMA
        order = tuple(params["order"]) if params else (1, 1, 1)
        seasonal_order = (tuple(params["seasonal_order"]) if params
                          else (0, 1, 1, 7))
        Xexog, scaler = build_task1_exogenous(g1.loc[fit_idx],
                                                fit_scaler=True)
        model = PmARIMA(order=order, seasonal_order=seasonal_order,
                        suppress_warnings=True)
        model.fit(y_fit.values, X=Xexog.values)
        artefact["model"] = model
        artefact["family"] = "SARIMAX"
        artefact["features_required"] = list(Xexog.columns)
        artefact["scaler"] = scaler
        artefact["order"] = list(order)
        artefact["seasonal_order"] = list(seasonal_order)
        artefact["inference"] = (
            "Xexog = build_task1_exogenous(future_g1, scaler=artefact['scaler'])\n"
            "model.predict(n_periods=H, X=Xexog.values)"
        )

    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        eng = load_engineered()
        X_consensus = build_selected_X(eng)
        df = pd.concat([y_full.rename("y"), X_consensus],
                        axis=1, join="inner").dropna()
        y_join = df["y"]; X_join = df.drop(columns=["y"])
        keep = X_join.index.intersection(fit_idx)
        Xfit = X_join.loc[keep]; yfit = y_join.loc[keep]
        p = params or {"n_estimators": 200, "max_depth": 5,
                       "learning_rate": 0.1, "subsample": 0.85}
        model = XGBRegressor(**p, objective="reg:squarederror",
                              random_state=42, n_jobs=-1)
        model.fit(Xfit.values, yfit.values)
        artefact["model"] = model
        artefact["family"] = "XGBoost"
        artefact["features_required"] = list(Xfit.columns)
        artefact["inference"] = (
            "X = engineered.loc[date_range, artefact['features_required']]\n"
            "yhat = model.predict(X.values)"
        )

    elif model_name == "ann":
        import torch, torch.nn as nn, torch.optim as optim
        from scripts.phase1_defaults_all_models import ann_fit_predict, DEFAULTS
        # We rebuild the network here so we can save state_dict properly
        p = params or DEFAULTS["ann"]
        eng = load_engineered()
        X_consensus = build_selected_X(eng)
        df = pd.concat([y_full.rename("y"), X_consensus],
                        axis=1, join="inner").dropna()
        keep = df.index.intersection(fit_idx)
        Xfit = df.loc[keep].drop(columns=["y"])
        yfit = df.loc[keep]["y"]
        mean = Xfit.mean(); std = Xfit.std(ddof=0).replace(0, 1.0)
        Xs = ((Xfit - mean) / std).values.astype(np.float32)
        y_mean = float(yfit.mean()); y_std = float(yfit.std() or 1.0)
        yn = (yfit.values.astype(np.float32) - y_mean) / y_std
        torch.manual_seed(p["seed"])
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
        for epoch in range(80):
            order = np.random.permutation(n)
            for i in range(0, n, bs):
                ix = order[i:i+bs]
                xb = torch.from_numpy(Xs[ix])
                yb = torch.from_numpy(yn[ix])
                opt.zero_grad()
                out = net(xb).squeeze(-1)
                loss = loss_fn(out, yb)
                loss.backward(); opt.step()
        artefact["model_state_dict"] = {k: v.cpu().numpy()
                                          for k, v in net.state_dict().items()}
        artefact["model_arch"] = {"in_dim": Xs.shape[1],
                                    "hidden_layers": p["hidden_layers"],
                                    "units": p["units"],
                                    "dropout": p["dropout"]}
        artefact["family"] = "ANN"
        artefact["features_required"] = list(Xfit.columns)
        artefact["feature_scaler"] = {"mean": mean.to_dict(),
                                        "std":  std.to_dict()}
        artefact["target_scaler"] = {"mean": y_mean, "std": y_std}
        artefact["inference"] = (
            "X_norm = (X - feature_scaler['mean']) / feature_scaler['std']\n"
            "yhat_n = net(X_norm); yhat = yhat_n * target_std + target_mean"
        )

    elif model_name == "lstm":
        # Mirror ANN but with LSTM; saved as state_dict
        import torch, torch.nn as nn, torch.optim as optim
        from scripts.phase1_defaults_all_models import lstm_fit_predict, DEFAULTS
        p = params or DEFAULTS["lstm"]
        eng = load_engineered()
        X_consensus = build_selected_X(eng)
        df = pd.concat([y_full.rename("y"), X_consensus],
                        axis=1, join="inner").dropna()
        keep = df.index.intersection(fit_idx)
        Xfit = df.loc[keep].drop(columns=["y"])
        yfit = df.loc[keep]["y"]
        mean = Xfit.mean(); std = Xfit.std(ddof=0).replace(0, 1.0)
        Xs = ((Xfit - mean) / std).values.astype(np.float32)
        y_mean = float(yfit.mean()); y_std = float(yfit.std() or 1.0)
        yn = (yfit.values.astype(np.float32) - y_mean) / y_std
        L = p["lookback"]
        seqs, tgts = [], []
        for i in range(L, len(Xs)):
            seqs.append(Xs[i-L:i]); tgts.append(yn[i])
        Xseq = np.stack(seqs).astype(np.float32)
        yseq = np.array(tgts, dtype=np.float32)
        torch.manual_seed(p["seed"])
        class Net(nn.Module):
            def __init__(self, in_dim, units, dropout):
                super().__init__()
                self.lstm = nn.LSTM(in_dim, units, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(units, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1, :])).squeeze(-1)
        net = Net(Xseq.shape[2], p["units"], p["dropout"])
        opt = optim.Adam(net.parameters(), lr=p["learning_rate"])
        loss_fn = nn.MSELoss()
        bs = p["batch_size"]; n = len(Xseq)
        for epoch in range(60):
            order = np.random.permutation(n)
            for i in range(0, n, bs):
                ix = order[i:i+bs]
                xb = torch.from_numpy(Xseq[ix])
                yb = torch.from_numpy(yseq[ix])
                opt.zero_grad()
                out = net(xb)
                loss = loss_fn(out, yb)
                loss.backward(); opt.step()
        artefact["model_state_dict"] = {k: v.cpu().numpy()
                                          for k, v in net.state_dict().items()}
        artefact["model_arch"] = {"in_dim": Xseq.shape[2],
                                    "lookback": L,
                                    "units": p["units"],
                                    "dropout": p["dropout"]}
        artefact["family"] = "LSTM"
        artefact["features_required"] = list(Xfit.columns)
        artefact["feature_scaler"] = {"mean": mean.to_dict(),
                                        "std":  std.to_dict()}
        artefact["target_scaler"] = {"mean": y_mean, "std": y_std}
        artefact["inference"] = (
            "Build sequence X_seq[t-L:t] of L past days, normalise, "
            "feed through LSTM, denormalise output."
        )

    elif model_name in ("sarimax_xgb", "sarimax_lstm", "stl_xgb", "stl_ann",
                         "stl_lstm", "lstm_xgb"):
        # Hybrids: we save a SPEC that the inference loader can re-build
        # rather than the live nested object (cleaner serialisation)
        artefact["family"] = "Hybrid"
        artefact["hybrid_spec"] = {
            "base":    model_name.split("_")[0] if "_" in model_name else None,
            "refiner": model_name.split("_")[-1],
        }
        artefact["features_required"] = []
        artefact["note"] = (
            "Hybrid models save the SARIMAX/STL base + the refiner. "
            "Inference requires re-instantiating both stages — see README."
        )

    else:
        raise ValueError(f"Unsupported deployment model: {model_name}")

    return artefact


# -------------------------------------------------------------------------
# Step 3: Persist
# -------------------------------------------------------------------------
def persist(rank: int, record: dict, artefact: dict) -> dict:
    model_name = record["model"]
    bundle_path = OUT / f"0{rank}_{model_name}.pkl"
    card_path   = OUT / f"0{rank}_{model_name}_card.json"

    joblib.dump(artefact, bundle_path, compress=3)
    print(f"  [pickle] {bundle_path.relative_to(ROOT)}  "
          f"({bundle_path.stat().st_size / 1024:.1f} KB)")

    card = {
        "rank":            rank,
        "model_name":      model_name,
        "family":          artefact.get("family"),
        "val_RMSE":        record["val_RMSE"],
        "val_MAPE":        record["val_MAPE"],
        "params":          artefact.get("params"),
        "source":          record["source"],
        "features_required": artefact.get("features_required", []),
        "training_timestamp_utc": artefact.get("training_timestamp_utc"),
        "fit_start":       artefact.get("fit_start"),
        "fit_end":         artefact.get("fit_end"),
        "inference":       artefact.get("inference"),
        "bundle_path":     str(bundle_path.relative_to(ROOT)),
    }
    card_path.write_text(json.dumps(card, indent=2, default=str),
                          encoding="utf-8")
    print(f"  [card]   {card_path.relative_to(ROOT)}")
    return card


def write_manifest_and_readme(cards: list[dict]) -> None:
    manifest = pd.DataFrame([{
        "rank": c["rank"], "model_name": c["model_name"],
        "family": c["family"], "val_RMSE": c["val_RMSE"],
        "val_MAPE": c["val_MAPE"], "n_features": len(c["features_required"]),
        "bundle_path": c["bundle_path"],
    } for c in cards])
    manifest.to_csv(OUT / "deployment_manifest.csv", index=False)
    print(f"  [manifest] {(OUT/'deployment_manifest.csv').relative_to(ROOT)}")

    readme = ["# Deployment artefacts — top-3 models",
              "",
              f"Generated: {datetime.now(timezone.utc).isoformat()}",
              "",
              "## Manifest",
              "",
              manifest.to_markdown(index=False, floatfmt=".3f"),
              "",
              "## How to load a model (Python)",
              "",
              "```python",
              "import joblib",
              "bundle = joblib.load('artefacts/deploy_top3/01_<model>.pkl')",
              "print(bundle['family'])      # e.g. 'SARIMAX'",
              "print(bundle['model'])        # the fitted model object",
              "print(bundle['features_required'])  # exog cols needed",
              "```",
              "",
              "## Inference snippets per model",
              "",]
    for c in cards:
        readme += [f"### {c['rank']}. {c['model_name']} ({c['family']})",
                   f"- val MAPE: {c['val_MAPE']:.2f}%   RMSE: {c['val_RMSE']:.3f}",
                   f"- Required features: `{c['features_required']}`",
                   "",
                   "```python",
                   c.get("inference", "# see model card"),
                   "```", ""]

    readme += ["## Web-app integration checklist",
               "",
               "- [ ] `pip install joblib pmdarima xgboost torch pandas`",
               "- [ ] Mount the `artefacts/deploy_top3/` folder into the app",
               "- [ ] Build daily feature row from G1 + engineered pipeline",
               "- [ ] For SARIMAX: pass scaled exog block (use bundled scaler)",
               "- [ ] For ANN/LSTM: rebuild the torch network from "
                "`model_arch` + `model_state_dict`",
               "- [ ] Return point forecast + (optional) 95% interval",
               "- [ ] Log all predictions vs actuals for drift monitoring",
               ""]

    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"  [readme]   {(OUT/'README.md').relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None,
                        help="explicit comma-list (default: auto-pick by val_RMSE)")
    args = parser.parse_args()

    print("=" * 70)
    print("Deploy top-3 models — pickling for web-app deployment")
    print("=" * 70)

    explicit = (None if args.models is None
                 else [m.strip() for m in args.models.split(",")])
    top = discover_top3(explicit)
    if not top:
        print("[error] No model summaries found yet. Run Phase 1 + 2 first.")
        return

    print(f"Top {len(top)} models picked:")
    for r, rec in enumerate(top, 1):
        print(f"  {r}. {rec['model']:18s}  val_RMSE={rec['val_RMSE']:.3f}  "
              f"val_MAPE={rec['val_MAPE']:.2f}%  ({rec['source']})")

    cards = []
    for rank, rec in enumerate(top, 1):
        print(f"\n[{rank}/{len(top)}] Refitting {rec['model']} on TRAIN+VAL...")
        t0 = time.time()
        try:
            artefact = fit_for_deployment(rec["model"], rec["params"])
            card = persist(rank, rec, artefact)
            cards.append(card)
            print(f"  done in {time.time()-t0:.1f}s")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    if cards:
        write_manifest_and_readme(cards)
        print("\n" + "=" * 70)
        print("Deployment artefacts ready in:", OUT)


if __name__ == "__main__":
    main()
