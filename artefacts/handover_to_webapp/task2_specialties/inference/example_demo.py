"""Task 2 smoke test — load every deployed (specialty, alias) and print card.

    python task2_specialties/inference/example_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task2_specialties.inference.load import (
    load_catalogue, load_model,
)


def main() -> None:
    print("=" * 70)
    print("Task 2 — Per-Specialty ED Arrivals — Deployment Smoke Test")
    print("=" * 70)

    catalogue = load_catalogue()
    print(f"\n{len(catalogue)} specialties in catalogue")

    total_models = 0
    for entry in catalogue:
        specialty = entry["specialty"]
        res = entry["resolution"]
        avail = entry["available_models"]
        print(f"\n  [{specialty} — {res}] available: {avail}")
        for alias in avail:
            try:
                bundle = load_model(specialty, alias)
                card = bundle["card"]
                badge = f"{card['badge_emoji']} {card['badge_label']}"
                mape = card["performance"]["val_MAPE"]
                print(f"     {alias:9s} loaded   val MAPE {mape:>5.1f}%  {badge}")
                total_models += 1
            except Exception as exc:
                print(f"     {alias:9s} FAILED  {exc}")

    print("\n" + "=" * 70)
    print(f"Total Task 2 models loaded: {total_models}")
    print("=" * 70)


if __name__ == "__main__":
    main()
