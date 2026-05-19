"""Data loading and split freezing.

Reads the joined G1..G4 CSVs produced by the Chapter 5 EDA pipeline and exposes
fixed train / validation / test splits per §5.5.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(name: str) -> dict:
    local = CONFIG_DIR / name.replace(".yaml", ".local.yaml")
    if local.exists():
        return yaml.safe_load(local.read_text())
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def load_paths() -> dict:
    return _load_yaml("paths.yaml")


def load_split_config() -> dict:
    return _load_yaml("split.yaml")


@dataclass(frozen=True)
class Splits:
    """Fixed boundaries from Chapter 5 §5.5.2."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    pre_covid_start: pd.Timestamp
    pre_covid_end: pd.Timestamp

    @classmethod
    def from_config(cls) -> "Splits":
        cfg = load_split_config()
        return cls(
            train_start=pd.Timestamp(cfg["train"]["start"]),
            train_end=pd.Timestamp(cfg["train"]["end"]),
            val_start=pd.Timestamp(cfg["validation"]["start"]),
            val_end=pd.Timestamp(cfg["validation"]["end"]),
            test_start=pd.Timestamp(cfg["test"]["start"]),
            test_end=pd.Timestamp(cfg["test"]["end"]),
            pre_covid_start=pd.Timestamp(cfg["pre_covid"]["start"]),
            pre_covid_end=pd.Timestamp(cfg["pre_covid"]["end"]),
        )

    def label(self, date: pd.Timestamp) -> str:
        """Return 'train', 'val', 'test', 'pre_covid', or 'during_covid'."""
        if self.train_start <= date <= self.train_end:
            return "train"
        if self.val_start <= date <= self.val_end:
            return "val"
        if self.test_start <= date <= self.test_end:
            return "test"
        if self.pre_covid_start <= date <= self.pre_covid_end:
            return "pre_covid"
        return "during_covid"

    def slice(self, df: pd.DataFrame, block: str) -> pd.DataFrame:
        bounds = {
            "train": (self.train_start, self.train_end),
            "val": (self.val_start, self.val_end),
            "test": (self.test_start, self.test_end),
            "pre_covid": (self.pre_covid_start, self.pre_covid_end),
        }
        if block not in bounds:
            raise ValueError(f"Unknown block: {block!r}")
        start, end = bounds[block]
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])
        mask = (idx >= start) & (idx <= end)
        return df.loc[mask].copy()


def _read_csv_with_date(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    return df


def load_g1(filter_zero_days: bool = True) -> pd.DataFrame:
    """Load G1 daily demand. Filters the 17 MCAR is_zero_day == 1 rows per §4.4.1."""
    paths = load_paths()
    df = _read_csv_with_date(paths["inputs"]["g1_daily"])
    if filter_zero_days and "is_zero_day" in df.columns:
        df = df[df["is_zero_day"] == 0].copy()
    return df


def load_g2() -> pd.DataFrame:
    """Load G2 hourly demand."""
    paths = load_paths()
    return _read_csv_with_date(paths["inputs"]["g2_hourly"])


def load_g3(filter_zero_days: bool = True) -> pd.DataFrame:
    """Load G3 clinical daily (per-specialty)."""
    paths = load_paths()
    df = _read_csv_with_date(paths["inputs"]["g3_clinical_daily"])
    if filter_zero_days and "is_zero_day" in df.columns:
        df = df[df["is_zero_day"] == 0].copy()
    return df


def load_g4() -> pd.DataFrame:
    """Load G4 clinical hourly (per-specialty hourly)."""
    paths = load_paths()
    return _read_csv_with_date(paths["inputs"]["g4_clinical_hourly"])


def get_split_frames(
    df: pd.DataFrame, splits: Optional[Splits] = None
) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} DataFrames sliced from df."""
    splits = splits or Splits.from_config()
    return {block: splits.slice(df, block) for block in ("train", "val", "test")}


def verify_split_sizes(df: pd.DataFrame, splits: Optional[Splits] = None) -> dict[str, int]:
    """Check the train / val / test row counts against §5.5.2 expectations."""
    splits = splits or Splits.from_config()
    cfg = load_split_config()
    counts = {block: len(splits.slice(df, block)) for block in ("train", "val", "test")}
    expected = {
        "train": cfg["train"]["expected_days"],
        "val": cfg["validation"]["expected_days"],
        "test": cfg["test"]["expected_days"],
    }
    for block, actual in counts.items():
        if actual != expected[block]:
            print(
                f"  WARNING: {block} count {actual} != expected {expected[block]} "
                f"(check is_zero_day filter and date coverage)"
            )
    return counts


if __name__ == "__main__":
    splits = Splits.from_config()
    print("Splits loaded from configs/split.yaml:")
    print(f"  train  {splits.train_start.date()} -> {splits.train_end.date()}")
    print(f"  val    {splits.val_start.date()} -> {splits.val_end.date()}")
    print(f"  test   {splits.test_start.date()} -> {splits.test_end.date()}")

    g1 = load_g1()
    print(f"\nG1 loaded: {g1.shape[0]} rows x {g1.shape[1]} cols (zero days filtered)")
    print(f"  date range: {g1.index.min().date()} -> {g1.index.max().date()}")
    counts = verify_split_sizes(g1, splits)
    print(f"\nSplit counts (post zero-day filter): {counts}")
