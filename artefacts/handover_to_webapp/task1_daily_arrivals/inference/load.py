"""Task 1 model loader — uniform interface for all 6 aliases.

Usage:
    from task1_daily_arrivals.inference.load import load_model
    bundle = load_model("Hybrid 1")
    bundle["fitted"]    # the fitted model (pmdarima ARIMA, XGBRegressor, etc.)
    bundle["card"]      # the JSON card with metadata, badge, etc.
"""
from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent.parent
MODELS = THIS / "models"
CARDS = THIS / "cards"

ALIAS_TO_FILENAME = {
    "Stat 1":   "stat1",
    "Stat 2":   "stat2",
    "ML 1":     "ml1",
    "ML 2":     "ml2",
    "Hybrid 1": "hybrid1",
    "Hybrid 2": "hybrid2",
}

VALID_ALIASES = set(ALIAS_TO_FILENAME.keys())


def load_model(alias: str) -> dict[str, Any]:
    """Load a Task 1 model by its alias (e.g. "Hybrid 1").

    Returns a dict:
        {
          "alias": str,
          "fitted": <fitted model object>,
          "card": dict (the metadata card),
          "scientific_name": str (for internal audit only — DO NOT expose in UI)
        }
    """
    if alias not in VALID_ALIASES:
        raise ValueError(
            f"Unknown alias: {alias!r}. Valid Task 1 aliases: "
            f"{sorted(VALID_ALIASES)}"
        )
    fname = ALIAS_TO_FILENAME[alias]
    pkl = MODELS / f"{fname}.pkl"
    card_path = CARDS / f"{fname}.json"

    if not pkl.exists():
        raise FileNotFoundError(f"Pickle missing: {pkl}")
    if not card_path.exists():
        raise FileNotFoundError(f"Card missing: {card_path}")

    bundle = joblib.load(pkl)
    card = json.loads(card_path.read_text(encoding="utf-8"))

    return {
        "alias": alias,
        "fitted": bundle,
        "card": card,
        "scientific_name": card["internal_only"]["scientific_name"],
    }


def list_models() -> list[str]:
    """Return all available Task 1 aliases (for filling dropdowns)."""
    return sorted(VALID_ALIASES)
