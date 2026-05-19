"""Feature matrix construction.

Task 1: the §5.2.5 raw 10 inventory (categorical + 2 continuous + 7 calendar binaries).
        Used directly by SARIMA, ARIMA-X, NB GLM — no engineering, no consensus.

Task 2: per-specialty exogenous blocks per §5.3.3, with Surgery sign-reversal columns.

The downstream ML models (XGBoost / ANN / LSTM / hybrids) load
artefacts/engineered + consensus selection separately — see engineering.py
and consensus.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


@dataclass
class StandardScaler:
    """Minimal scaler that fits on a training fold and applies elsewhere."""
    mean_: pd.Series
    std_: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: Sequence[str]) -> "StandardScaler":
        sub = df[list(cols)]
        return cls(mean_=sub.mean(), std_=sub.std(ddof=0).replace(0, 1.0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.mean_.index:
            if col in out.columns:
                out[col] = (out[col] - self.mean_[col]) / self.std_[col]
        return out


def _encode_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """6 dummies (Mon as reference). day_of_week column in G1 is 0-6."""
    dow = df["day_of_week"].astype(int)
    out = pd.DataFrame(index=df.index)
    for d in range(1, 7):
        out[f"dow_{d}"] = (dow == d).astype(int)
    return out


def build_task1_exogenous(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Build the §5.2.5 raw 10 feature block.

    Returns (X, scaler). If fit_scaler=True the scaler is fit on df; otherwise
    pass an externally-fit scaler to apply to val/test/etc.
    """
    cfg = _load_yaml("features_task1.yaml")
    continuous = cfg["continuous"]
    binaries = cfg["binary"]

    dow_dummies = _encode_day_of_week(df)
    continuous_block = df[continuous].copy()
    binary_block = df[binaries].astype(int)

    if fit_scaler:
        scaler = StandardScaler.fit(df, continuous)
    if scaler is None:
        raise ValueError("Pass fit_scaler=True or supply a fitted scaler")

    continuous_scaled = scaler.transform(continuous_block)
    X = pd.concat([dow_dummies, continuous_scaled, binary_block], axis=1)
    X.index = df.index
    return X, scaler


def build_task2_exogenous(
    df: pd.DataFrame,
    specialty: str,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Per-specialty exogenous block per §5.3.3.

    Always includes day-of-week dummies + 7 §5.2.5 calendar binaries. Weather
    columns and Surgery sign-reversal columns come from configs/features_task2.yaml.
    """
    task1_cfg = _load_yaml("features_task1.yaml")
    task2_cfg = _load_yaml("features_task2.yaml")
    binaries = task1_cfg["binary"]

    if specialty in task2_cfg["daily_specialties"]:
        spec_cfg = task2_cfg["daily_specialties"][specialty]
    elif specialty in task2_cfg["weekly_specialties"]:
        spec_cfg = task2_cfg["weekly_specialties"][specialty]
    else:
        raise KeyError(f"Unknown specialty: {specialty!r}")

    weather_cols = spec_cfg.get("weather", []) or []
    interaction_cols = spec_cfg.get("interactions", []) or []

    dow_dummies = _encode_day_of_week(df)
    binary_block = df[binaries].astype(int)

    if weather_cols:
        if fit_scaler:
            scaler = StandardScaler.fit(df, weather_cols)
        if scaler is None:
            raise ValueError("Pass fit_scaler=True or supply a fitted scaler for weather")
        weather_block = scaler.transform(df[weather_cols].copy())
    else:
        weather_block = pd.DataFrame(index=df.index)

    interaction_block = pd.DataFrame(index=df.index)
    for col in interaction_cols:
        # For Surgery these are duplicate columns; renamed to make their role explicit
        interaction_block[f"{specialty.lower()}_{col}"] = df[col].astype(int)

    X = pd.concat([dow_dummies, weather_block, binary_block, interaction_block], axis=1)
    X.index = df.index
    return X, scaler


def task1_share_target(df_g1: pd.DataFrame, df_g3: pd.DataFrame, specialty_col: str) -> pd.Series:
    """Specialty share-of-header target per §4.4.4.

    Reconstruction at delivery time: predicted_count = predicted_share * task1_forecast.
    """
    aligned = df_g3.join(df_g1["total_daily_arrivals"], how="inner")
    share = aligned[specialty_col] / aligned["total_daily_arrivals"]
    share.name = f"{specialty_col}_share"
    return share


if __name__ == "__main__":
    from .io import load_g1, Splits

    splits = Splits.from_config()
    g1 = load_g1()
    train_df = splits.slice(g1, "train")
    val_df = splits.slice(g1, "val")

    X_train, scaler = build_task1_exogenous(train_df, fit_scaler=True)
    X_val, _ = build_task1_exogenous(val_df, scaler=scaler)

    print(f"Train exogenous X: {X_train.shape}")
    print(f"Val exogenous X:   {X_val.shape}")
    print(f"Columns: {list(X_train.columns)}")
    print(f"\nMean of standardised columns on val (sanity check, should be near 0):")
    print(X_val[scaler.mean_.index].mean().round(3))
