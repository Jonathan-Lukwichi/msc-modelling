"""Tests for src/forecasting/leaderboard.py (Prompt 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.leaderboard import (
    CANONICAL_SCHEMA,
    append_row,
    load_leaderboard,
    to_latex,
    per_quarter_table,
)


def test_append_then_load_roundtrip(tmp_path: Path):
    p = tmp_path / "lb.parquet"
    append_row(
        p, model="XGBoost", family="ml", criterion="rmse",
        val_metrics={"MAPE": 11.99, "RMSE": 9.35},
        test_metrics={"MAPE": 12.63, "RMSE": 10.30, "MASE": 0.87,
                       "Winkler_80": 48.2, "Coverage_80": 0.81},
        per_horizon=pd.DataFrame({"horizon": [1, 3, 7], "MAPE": [14.3, 11.6, 10.8]}),
        source_csv="artefacts/metrics/xgboost_rmse_metrics.csv",
    )
    df = load_leaderboard(p)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["model"] == "XGBoost"
    assert row["val_mape"] == pytest.approx(11.99)
    assert row["test_winkler_80"] == pytest.approx(48.2)
    assert row["h1_mape"] == pytest.approx(14.3)
    assert row["h7_mape"] == pytest.approx(10.8)


def test_upsert_overwrites_same_model_criterion(tmp_path: Path):
    p = tmp_path / "lb.parquet"
    append_row(p, model="ANN", family="dl", criterion="mape",
                 val_metrics={"MAPE": 12.0}, test_metrics={"MAPE": 13.2})
    append_row(p, model="ANN", family="dl", criterion="mape",
                 val_metrics={"MAPE": 11.9}, test_metrics={"MAPE": 13.1})
    df = load_leaderboard(p)
    assert len(df) == 1
    assert df.iloc[0]["val_mape"] == pytest.approx(11.9)


def test_sort_by_test_mape(tmp_path: Path):
    p = tmp_path / "lb.parquet"
    append_row(p, model="A", family="ml", criterion="mape",
                 test_metrics={"MAPE": 15.0})
    append_row(p, model="B", family="ml", criterion="mape",
                 test_metrics={"MAPE": 10.0})
    append_row(p, model="C", family="ml", criterion="mape",
                 test_metrics={"MAPE": 12.0})
    df = load_leaderboard(p)
    assert list(df["model"]) == ["B", "C", "A"]


def test_to_latex_renders_columns():
    df = pd.DataFrame({
        "model": ["ARIMA", "XGBoost"],
        "family": ["parametric", "ml"],
        "val_mape": [12.0, 11.99],
        "test_mape": [13.5, 12.63],
        "test_mase": [0.92, 0.87],
        "test_winkler_80": [52.4, 48.2],
        "test_coverage_80": [0.79, 0.81],
    })
    out = to_latex(df, caption="t", label="t")
    assert r"\begin{longtable}" in out
    assert "ARIMA" in out
    assert "11.99" in out


def test_per_quarter_table_drift_sensitivity(tmp_path: Path):
    csv = tmp_path / "tpq.csv"
    pd.DataFrame({
        "model": ["A", "A", "A", "B", "B", "B"],
        "quarter": ["2025Q1", "2025Q2", "2025Q3", "2025Q1", "2025Q2", "2025Q3"],
        "MAPE": [10.0, 11.0, 12.0, 13.0, 13.5, 13.8],
    }).to_csv(csv, index=False)
    out = per_quarter_table(csv)
    assert "drift_sensitivity" in out.columns
    # A's drift = 12 - 10 = 2, B's = 13.8 - 13 = 0.8
    assert out.index[0] == "A"
    assert out.iloc[0]["drift_sensitivity"] == pytest.approx(2.0)


def test_schema_field_count():
    """If someone changes the schema, this test breaks loudly."""
    assert len(CANONICAL_SCHEMA.names) == 17
