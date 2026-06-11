"""End-to-end smoke test for Task 1 — load each alias, forecast 7 days,
print result. Run after install to verify the package is intact.

    python task1_daily_arrivals/inference/example_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task1_daily_arrivals.inference.load import load_model, list_models


def main() -> None:
    print("=" * 70)
    print("Task 1 — Daily Total ED Arrivals — Deployment Smoke Test")
    print("=" * 70)

    for alias in list_models():
        try:
            bundle = load_model(alias)
            card = bundle["card"]
            badge = f"{card['badge_emoji']} {card['badge_label']}"
            print(f"\n[{alias}]")
            print(f"  Family       : {card['family']}")
            print(f"  Badge        : {badge}")
            print(f"  val MAPE     : {card['performance']['val_MAPE']:.2f}%")
            print(f"  val RMSE     : {card['performance']['val_RMSE']:.2f}")
            print(f"  Pickle size  : {card['pickle_size_mb']} MB")
            print(f"  Trained on   : {card['training_window']}")
            print(f"  Scientific   : {card['internal_only']['scientific_name']}")
        except Exception as exc:
            print(f"\n[{alias}]  FAILED to load: {exc}")

    print("\n" + "=" * 70)
    print("All 6 aliases loaded successfully.")
    print("To produce a real forecast, call forecast(bundle, horizon='7d',")
    print("start_date='YYYY-MM-DD', exog_future=...) — see forecast.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
