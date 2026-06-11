"""Task 2 model loader — uniform interface across all specialties.

Usage:
    from task2_specialties.inference.load import load_model, available_aliases
    bundle = load_model("Medicine", "Stat 1")
    bundle["fitted"]    # the fitted model (ARIMA, XGBoost, etc.)
    bundle["card"]      # JSON card with badge, performance, etc.
"""
from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent.parent
ROOT = THIS  # task2_specialties/

ALIAS_TO_FILESTEM = {
    "Stat 1":   "stat1",
    "Stat 2":   "stat2",
    "ML 1":     "ml1",
    "ML 2":     "ml2",
}

WEEKLY_SPECIALTIES = {"Maternity", "Psychiatry"}


def _specialty_dirname(specialty: str) -> str:
    """Map specialty name to folder name in task2_specialties/."""
    base = specialty.lower()
    if specialty in WEEKLY_SPECIALTIES:
        return f"{base}_weekly"
    return base


def _filename_for(alias: str, specialty: str) -> str:
    stem = ALIAS_TO_FILESTEM[alias]
    if specialty in WEEKLY_SPECIALTIES:
        return f"{stem}_weekly"
    return stem


def load_catalogue() -> list[dict]:
    """Return the master catalogue.json (specialty -> available aliases)."""
    return json.loads((ROOT / "catalogue.json").read_text(encoding="utf-8"))


def available_aliases(specialty: str) -> list[str]:
    """Return the list of aliases trained for this specialty (for UI dropdown filter)."""
    cat = load_catalogue()
    for entry in cat:
        if entry["specialty"] == specialty:
            return entry["available_models"]
    raise ValueError(f"Unknown specialty: {specialty!r}")


def load_model(specialty: str, alias: str) -> dict[str, Any]:
    """Load a Task 2 model.

    Returns a dict:
        {
          "alias": str,
          "specialty": str,
          "resolution": "daily" or "weekly",
          "fitted": <model bundle dict>,
          "card": <metadata card>,
        }
    """
    if alias not in ALIAS_TO_FILESTEM:
        raise ValueError(f"Unknown alias: {alias!r}. "
                         f"Valid: {sorted(ALIAS_TO_FILESTEM)}")

    avail = available_aliases(specialty)
    if alias not in avail:
        raise ValueError(
            f"{alias!r} is not deployed for {specialty!r}. "
            f"Available for this specialty: {avail}"
        )

    spec_dir = ROOT / _specialty_dirname(specialty)
    fname = _filename_for(alias, specialty)
    pkl_path = spec_dir / "models" / f"{fname}.pkl"
    card_path = spec_dir / "cards" / f"{fname}.json"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle missing: {pkl_path}")
    if not card_path.exists():
        raise FileNotFoundError(f"Card missing: {card_path}")

    bundle = joblib.load(pkl_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))

    return {
        "alias": alias,
        "specialty": specialty,
        "resolution": "weekly" if specialty in WEEKLY_SPECIALTIES else "daily",
        "fitted": bundle,
        "card": card,
    }
