"""§3.4.3 Algorithm 1: load upstream pre-computed four-method consensus selection.

Algorithm 1 (Dummy / RF permutation / Lasso / GBM gain) has already been executed
in the EDA pipeline. This module loads the result and exposes the retained
feature list.

Per the audit recorded in CHAPTER_6_PLAN.md §10.6: of the §5.2.5 raw 10 features,
only `day_of_week` and `is_long_weekend` survive Algorithm 1 on the engineered
space. The other 8 calendar binaries are absorbed by `arrivals_lag_{7,14,21,28}`
and `rolling_{mean,std}_{7,14,30}d` — those are the features that dominate the
voting. This divergence is expected and is the design intent of the two-pipeline
architecture (parametric baselines use raw 10; ML models use the 23 consensus).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_consensus() -> pd.DataFrame:
    """Load the upstream consensus selection table.

    Columns: Feature, Dummy, RF_Perm, Lasso, GBM, Total, Consensus.
    Consensus == 1 iff Total >= 2 (the §3.4.3 vote threshold).
    """
    paths = yaml.safe_load((CONFIG_DIR / "paths.yaml").read_text())
    path = paths["upstream_artefacts"]["consensus_selection"]
    return pd.read_csv(path)


def retained_features() -> list[str]:
    """Return the alphabetically-sorted list of features with vote count >= 2."""
    sel = load_consensus()
    return sorted(sel[sel["Consensus"] == 1]["Feature"].tolist())


def build_selected_X(engineered: pd.DataFrame) -> pd.DataFrame:
    """Subset the engineered matrix to the consensus-retained features only."""
    keep = retained_features()
    available = [c for c in keep if c in engineered.columns]
    missing = [c for c in keep if c not in engineered.columns]
    if missing:
        print(f"WARNING: {len(missing)} consensus-retained features not in "
              f"engineered matrix: {missing}")
    return engineered[available].copy()


def audit_raw10_survival() -> dict[str, bool]:
    """Check whether each §5.2.5 raw 10 feature survives the consensus."""
    retained = set(retained_features())
    raw10 = [
        "day_of_week", "temp_mean_C", "wind_max_kmh",
        "is_weekend", "is_long_weekend", "is_public_holiday",
        "is_school_holiday", "is_festive_season",
        "is_winter_holiday", "is_near_holiday",
    ]
    return {f: (f in retained) for f in raw10}


if __name__ == "__main__":
    from .engineering import load_engineered

    sel = load_consensus()
    print(f"Consensus table: {sel.shape}")
    print(f"Retained (vote >= 2): {sel['Consensus'].sum()} of {len(sel)}")
    vote_dist = sel["Total"].value_counts().sort_index()
    print(f"Vote distribution: {dict(vote_dist)}")

    print("\n§5.2.5 raw 10 audit:")
    for f, kept in audit_raw10_survival().items():
        print(f"  {'KEPT' if kept else 'OUT '} {f}")

    eng = load_engineered()
    X = build_selected_X(eng)
    print(f"\nSelected feature matrix shape: {X.shape}")
