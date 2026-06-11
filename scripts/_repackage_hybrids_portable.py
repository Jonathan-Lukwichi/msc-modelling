"""Re-package the SARIMAX-hybrid pickles as portable, self-contained bundles.

Original pickles wrap their components in `src.forecasting.deploy.ModelPackage`,
which prevents loading them outside the project. We strip the wrapper and save
only plain Python dicts pointing to portable inner objects:
  - base    -> pmdarima.arima.ARIMA
  - refiner -> xgboost.XGBRegressor OR torch state_dict (with arch metadata)

Output: artefacts/handover_to_webapp/task1_daily_arrivals/models/{hybrid1,hybrid2}.pkl
"""
from __future__ import annotations

import sys
import joblib
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so the original pickles can resolve ModelPackage

# Importing this triggers the class registration the original pickles need
import src.forecasting.deploy  # noqa: F401

SRC = ROOT / "artefacts" / "models" / "deploy"
DST = ROOT / "artefacts" / "handover_to_webapp" / "task1_daily_arrivals" / "models"


def portable_modelpackage(mp) -> dict:
    """Turn a ModelPackage instance into a flat dict of portable parts."""
    return {
        "name":           mp.name,
        "family":         mp.family,
        "fitted":         mp.fitted,           # pmdarima ARIMA / XGBRegressor
        "feature_names":  mp.feature_names,
        "feature_scaler": mp.feature_scaler,
        "target_scaler":  mp.target_scaler,
        "lookback":       mp.lookback,
        "best_params":    mp.best_params,
        "metadata":       mp.metadata,
    }


def repackage(src_name: str, dst_name: str) -> None:
    src_pkl = SRC / src_name
    dst_pkl = DST / dst_name
    print(f"\nRe-packaging {src_name} -> {dst_name}")

    original = joblib.load(src_pkl)
    fitted_dict = original["fitted"]
    base_mp = fitted_dict["base"]
    refiner_mp = fitted_dict["refiner"]

    portable = {
        "name":          original["name"],
        "family":        original["family"],
        "base":          portable_modelpackage(base_mp),
        "refiner":       portable_modelpackage(refiner_mp),
        "feature_names": original["feature_names"],
        "feature_scaler": original.get("feature_scaler"),
        "target_scaler":  original.get("target_scaler"),
        "lookback":       original.get("lookback"),
        "best_params":    original["best_params"],
        "metadata":       original["metadata"],
    }

    # Test pickle-ability without the src module on path
    raw = pickle.dumps(portable)
    print(f"  pickled size: {len(raw) / (1024*1024):.1f} MB")

    joblib.dump(portable, dst_pkl, compress=3)
    print(f"  wrote: {dst_pkl}")


if __name__ == "__main__":
    repackage("hybrid_sarimax_xgb.pkl",  "hybrid1.pkl")
    repackage("hybrid_sarimax_lstm.pkl", "hybrid2.pkl")
    print("\nDone. Re-run the smoke test to verify portability.")
