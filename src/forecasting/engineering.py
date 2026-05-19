"""§3.4.2 feature engineering: load upstream pre-computed engineered matrix.

The full §3.4.2 recipe (cyclical sin/cos, Fourier harmonics, lags, rolling stats,
interactions) has already been executed in the EDA pipeline and lives at the
path under upstream_artefacts.engineered_features in configs/paths.yaml.

This module loads that matrix and verifies it covers the date range we need.
If the upstream file is missing or its shape is wrong, the caller should
regenerate from scratch per CHAPTER_6_PLAN.md §9.4.

Used by: XGBoost, ANN, LSTM, and the 6 hybrids (NOT by SARIMAX / ARIMA / NB GLM —
those use the §5.2.5 raw 10 inventory directly via features.build_task1_exogenous).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_engineered() -> pd.DataFrame:
    """Load the §3.4.2 engineered matrix from the upstream artefact.

    Returns a DataFrame indexed by date with 100 columns including:
      - cyclical sin/cos (dow, month, doy)
      - Fourier annual + weekly harmonics
      - lag features arrivals_lag_{1,2,3,7,14,21,28,30}
      - rolling stats at windows 7/14/30
      - calendar binaries (23)
    """
    paths = yaml.safe_load((CONFIG_DIR / "paths.yaml").read_text())
    path = paths["upstream_artefacts"]["engineered_features"]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True).set_index("date")
    return df


def validate_engineered(df: pd.DataFrame) -> dict:
    """Sanity-check the engineered matrix against §3.4.2 expectations.

    Returns a dict of {check_name: bool} for the verification harness.
    """
    cyclical_cols = [c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")]
    fourier_cols = [c for c in df.columns if c.startswith("fourier_")]
    lag_cols = [c for c in df.columns if "lag_" in c]
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    return {
        "row_count_2440": len(df) == 2440,
        "col_count_in_50_100": 50 <= df.shape[1] <= 100,
        "has_cyclical": len(cyclical_cols) >= 4,
        "has_fourier_harmonics": len(fourier_cols) >= 12,
        "has_lags": len(lag_cols) >= 4,
        "has_rolling": len(rolling_cols) >= 4,
        "date_range_min": df.index.min(),
        "date_range_max": df.index.max(),
        "col_count": df.shape[1],
        "lag_count": len(lag_cols),
        "rolling_count": len(rolling_cols),
    }


if __name__ == "__main__":
    df = load_engineered()
    print(f"Engineered matrix: {df.shape}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print("\nValidation results:")
    for k, v in validate_engineered(df).items():
        print(f"  {k}: {v}")
