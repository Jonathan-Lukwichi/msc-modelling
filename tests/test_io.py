"""Smoke tests for src/forecasting/io.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.io import (
    Splits, load_g1, load_g3, load_split_config, verify_split_sizes,
)
from conftest import requires_real_data


def test_splits_dates_match_chapter5():
    """§5.5.2 split dates are settled and must not drift."""
    s = Splits.from_config()
    assert s.train_start == pd.Timestamp("2022-03-01")
    assert s.train_end == pd.Timestamp("2024-06-30")
    assert s.val_start == pd.Timestamp("2024-07-01")
    assert s.val_end == pd.Timestamp("2024-12-31")
    assert s.test_start == pd.Timestamp("2025-01-01")
    assert s.test_end == pd.Timestamp("2026-01-31")


@requires_real_data
def test_g1_load_and_filter():
    """G1 loads, has total_daily_arrivals, and zero-day filter drops 17 rows (§4.4.1)."""
    g1_all = load_g1(filter_zero_days=False)
    g1_filtered = load_g1(filter_zero_days=True)
    assert "total_daily_arrivals" in g1_all.columns
    assert "is_zero_day" in g1_all.columns
    dropped = len(g1_all) - len(g1_filtered)
    # §4.4.1 says 17 MCAR zero days. Allow exact match.
    assert dropped == 17, f"Expected 17 zero days dropped, got {dropped}"


@requires_real_data
def test_g1_split_counts_match_calendar_window():
    """val=184 and test=396 should hold post-filter (no zero days in those blocks).
    train may differ from 853 by however many zero days fall in that window."""
    g1 = load_g1()
    counts = verify_split_sizes(g1)
    assert counts["val"] == 184
    assert counts["test"] == 396
    assert 840 <= counts["train"] <= 853


@requires_real_data
def test_g3_loads_with_specialty_columns():
    """G3 should expose per-specialty count columns."""
    g3 = load_g3()
    cols_lower = [c.lower() for c in g3.columns]
    # Should have at least medicine and surgery counts
    expected_substrings = ("medic", "surg", "ortho", "paed", "gyn")
    found = [s for s in expected_substrings
             if any(s in c for c in cols_lower)]
    assert len(found) >= 3, f"Too few specialty columns found: {found}"
