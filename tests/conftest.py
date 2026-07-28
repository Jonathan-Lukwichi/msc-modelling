"""Shared pytest fixtures and markers.

A handful of tests (test_io.py, test_features.py) need the raw G1-G4
hospital CSVs, which live outside this repo at a machine-specific
absolute path configured in configs/paths.local.yaml (gitignored, never
committed -- see README "Data access"). Those files are confidential and
cannot exist on a CI runner or a fresh clone that hasn't been pointed at
a copy of the dataset.

Rather than let the whole test session crash with a FileNotFoundError
the moment one of those tests imports src.forecasting.io, we probe once
at collection time and skip (not fail) the data-dependent tests with a
clear reason when the data isn't reachable. The remaining ~38 tests
(pure-function tests on synthetic arrays: metrics, CV folds, rolling
forecaster, leaderboard schema, OOF hybrid, drift weighting, ACI) are
unaffected and always run, in CI or anywhere else.
"""
from __future__ import annotations

import pytest


def _real_data_available() -> bool:
    try:
        from src.forecasting.io import load_g1
        load_g1()
        return True
    except Exception:
        return False


REAL_DATA_AVAILABLE = _real_data_available()

requires_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason=(
        "Requires the raw G1-G4 hospital CSVs referenced by "
        "configs/paths.local.yaml. Not available in this environment "
        "(e.g. CI, or a fresh clone without a local copy of the "
        "confidential dataset). See README 'Data access'."
    ),
)
