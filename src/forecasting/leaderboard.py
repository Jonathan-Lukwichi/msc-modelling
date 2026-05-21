"""Canonical leaderboard writer (Prompt 2 of Ch6 refactor).

Reconciles the heterogeneous per-script `*_metrics.csv` files into the
single parquet table `artefacts/leaderboard_canonical.parquet` whose
schema was frozen in Prompt 0.

Provides:
  - append_row    : upsert a (model, criterion) row
  - load_leaderboard : read back as a DataFrame sorted by test_mape
  - to_latex      : LaTeX longtable for chap6.tex
  - per_quarter_table : OOD honesty per-model breakdown
  - reconcile_metrics_csvs : ingest legacy CSVs into the canonical table

Why parquet:
  - Strongly typed (vs lossy CSV); roundtrip-safe for floats and timestamps.
  - Append-friendly with pyarrow.parquet.write_to_dataset (one file per
    write, but we keep it simple here and rewrite the whole table on each
    update; the table is small).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CANONICAL_SCHEMA = pa.schema([
    ("model", pa.string()),
    ("family", pa.string()),
    ("criterion", pa.string()),
    ("seed", pa.int64()),
    ("val_mape", pa.float64()),
    ("val_rmse", pa.float64()),
    ("val_mase", pa.float64()),
    ("test_mape", pa.float64()),
    ("test_rmse", pa.float64()),
    ("test_mase", pa.float64()),
    ("test_winkler_80", pa.float64()),
    ("test_coverage_80", pa.float64()),
    ("h1_mape", pa.float64()),
    ("h3_mape", pa.float64()),
    ("h7_mape", pa.float64()),
    ("source_csv", pa.string()),
    ("timestamp", pa.timestamp("us")),
])


# ---------------------------------------------------------------------------
# Core read/write
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pad missing columns with NaN/None so the schema matches."""
    for name in CANONICAL_SCHEMA.names:
        if name not in df.columns:
            df[name] = None
    return df[CANONICAL_SCHEMA.names]


def load_leaderboard(parquet_path: str | Path) -> pd.DataFrame:
    """Read the canonical leaderboard, sorted by test_mape ascending."""
    path = Path(parquet_path)
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_SCHEMA.names)
    df = pq.read_table(path).to_pandas()
    if "test_mape" in df.columns and len(df) > 0:
        df = df.sort_values("test_mape", ascending=True, kind="mergesort")
    return df.reset_index(drop=True)


def append_row(
    parquet_path: str | Path,
    *,
    model: str,
    family: str,
    criterion: str,
    seed: int = 42,
    val_metrics: Optional[dict] = None,
    test_metrics: Optional[dict] = None,
    per_horizon: Optional[pd.DataFrame] = None,
    source_csv: str = "",
) -> None:
    """Upsert a single (model, criterion) row into the canonical leaderboard.

    val_metrics  : dict with keys {"MAPE","RMSE","MASE"} (any missing -> NaN).
    test_metrics : dict with keys {"MAPE","RMSE","MASE","Winkler_80",
                                       "Coverage_80"} (any missing -> NaN).
    per_horizon  : DataFrame with columns 'horizon' (1..7) and 'MAPE'.
    """
    vm = val_metrics or {}
    tm = test_metrics or {}

    h_map = {1: np.nan, 3: np.nan, 7: np.nan}
    if per_horizon is not None and len(per_horizon) > 0:
        for h in (1, 3, 7):
            sel = per_horizon.loc[per_horizon["horizon"] == h, "MAPE"]
            if len(sel) > 0:
                h_map[h] = float(sel.iloc[0])

    new_row = pd.DataFrame([{
        "model": model,
        "family": family,
        "criterion": criterion,
        "seed": int(seed),
        "val_mape": float(vm.get("MAPE", np.nan)),
        "val_rmse": float(vm.get("RMSE", np.nan)),
        "val_mase": float(vm.get("MASE", np.nan)),
        "test_mape": float(tm.get("MAPE", np.nan)),
        "test_rmse": float(tm.get("RMSE", np.nan)),
        "test_mase": float(tm.get("MASE", np.nan)),
        "test_winkler_80": float(tm.get("Winkler_80", np.nan)),
        "test_coverage_80": float(tm.get("Coverage_80", np.nan)),
        "h1_mape": h_map[1],
        "h3_mape": h_map[3],
        "h7_mape": h_map[7],
        "source_csv": source_csv,
        "timestamp": pd.Timestamp.utcnow().to_pydatetime(),
    }])

    existing = load_leaderboard(parquet_path)
    if len(existing) > 0:
        mask = (existing["model"] == model) & (existing["criterion"] == criterion)
        existing = existing[~mask].copy()
    # Avoid the pandas-2.x "concat with empty / all-NA" FutureWarning by
    # only concatenating when the existing frame has rows.
    if existing.empty:
        combined = new_row.copy()
    else:
        combined = pd.concat([existing, new_row], ignore_index=True)
    combined = _ensure_columns(combined)

    # Coerce dtypes -> arrow schema
    combined["seed"] = (
        pd.to_numeric(combined["seed"], errors="coerce").fillna(0).astype("int64")
    )
    for col, t in zip(CANONICAL_SCHEMA.names, CANONICAL_SCHEMA.types):
        if pa.types.is_floating(t):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
        elif pa.types.is_string(t):
            combined[col] = combined[col].astype("string").fillna("")
        elif pa.types.is_timestamp(t):
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    table = pa.Table.from_pandas(combined, schema=CANONICAL_SCHEMA,
                                    preserve_index=False)
    path = Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def to_latex(
    df: pd.DataFrame,
    columns: Iterable[str] = (
        "model", "family", "val_mape", "test_mape",
        "test_mase", "test_winkler_80", "test_coverage_80",
    ),
    float_fmt: str = "{:.2f}",
    caption: str = "Canonical leaderboard.",
    label: str = "tab:leaderboard",
) -> str:
    """Render the leaderboard as a LaTeX longtable for chap6.tex."""
    sub = df[list(columns)].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(
                lambda v: "" if pd.isna(v) else float_fmt.format(v)
            )
    header = " & ".join(c.replace("_", r"\_") for c in columns) + r" \\"
    body_rows = " \\\\\n".join(
        " & ".join(str(v) for v in row) for row in sub.values.tolist()
    )
    return (
        r"\begin{longtable}{" + "l" * len(list(columns)) + r"}" + "\n"
        + r"\caption{" + caption + r"} \label{" + label + r"} \\" + "\n"
        + r"\toprule" + "\n" + header + "\n" + r"\midrule" + "\n"
        + r"\endhead" + "\n"
        + body_rows + r" \\" + "\n"
        + r"\bottomrule" + "\n" + r"\end{longtable}" + "\n"
    )


# ---------------------------------------------------------------------------
# OOD honesty table
# ---------------------------------------------------------------------------

def per_quarter_table(
    test_per_quarter_csv: str | Path,
) -> pd.DataFrame:
    """Read artefacts/metrics/test_per_quarter.csv and add drift_sensitivity.

    drift_sensitivity = max(quarterly_MAPE) - min(quarterly_MAPE) per model.
    Returns DataFrame sorted by drift_sensitivity descending.
    """
    df = pd.read_csv(test_per_quarter_csv)
    if "MAPE" not in df.columns or "model" not in df.columns:
        raise ValueError(
            f"{test_per_quarter_csv} must have 'model' and 'MAPE' columns."
        )
    pivot = df.pivot_table(
        index="model", columns="quarter", values="MAPE", aggfunc="first",
    )
    pivot["drift_sensitivity"] = pivot.max(axis=1) - pivot.min(axis=1)
    return pivot.sort_values("drift_sensitivity", ascending=False)


# ---------------------------------------------------------------------------
# Legacy CSV ingestion
# ---------------------------------------------------------------------------

LEGACY_FAMILY_MAP = {
    "arima": "parametric",
    "sarimax": "parametric",
    "nbglm": "parametric",
    "nb-glm": "parametric",
    "xgboost": "ml",
    "ann": "dl",
    "lstm": "dl",
    "naive_yest": "naive",
    "naive_seasonal": "naive",
    "naive_dow7": "naive",
    "dow_mean": "naive",
    "hybrid_sarimax_xgb": "hybrid",
    "hybrid_sarimax_lstm": "hybrid",
    "hybrid_lstm_xgb": "hybrid",
    "hybrid_stl_xgb": "hybrid",
    "hybrid_stl_ann": "hybrid",
    "hybrid_stl_lstm": "hybrid",
    "quantile_xgb": "uq",
    "conformal": "uq",
}


def family_of(model_name: str) -> str:
    key = model_name.lower().strip()
    return LEGACY_FAMILY_MAP.get(key, "other")


def reconcile_metrics_csvs(
    metrics_dir: str | Path = "artefacts/metrics",
    parquet_path: str | Path = "artefacts/leaderboard_canonical.parquet",
) -> pd.DataFrame:
    """Walk artefacts/metrics/ and upsert each model's row.

    Picks up files of the form `{model}_metrics.csv` and `{model}_rmse_metrics.csv`.
    Returns the resulting leaderboard DataFrame.
    """
    metrics_dir = Path(metrics_dir)
    parquet_path = Path(parquet_path)

    candidate_csvs = sorted(metrics_dir.glob("*_metrics.csv"))
    for csv_path in candidate_csvs:
        stem = csv_path.stem
        if stem.endswith("_rmse_metrics"):
            model_name = stem[: -len("_rmse_metrics")]
            criterion = "rmse"
        elif stem.endswith("_metrics"):
            model_name = stem[: -len("_metrics")]
            criterion = "mape"
        else:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        # Heuristic readers — accept either "block,MAPE,MAE,RMSE,R2" rows
        # or wide-form columns. Skip when we cannot find a val/test pair.
        if "block" in df.columns and "MAPE" in df.columns:
            val_row = df[df["block"] == "val"].head(1)
            test_row = df[df["block"] == "test"].head(1)
            vm = val_row[["MAPE", "RMSE"]].iloc[0].to_dict() if not val_row.empty else {}
            tm = test_row[["MAPE", "RMSE"]].iloc[0].to_dict() if not test_row.empty else {}
            if not vm and not tm:
                continue
            append_row(
                parquet_path,
                model=model_name, family=family_of(model_name),
                criterion=criterion,
                val_metrics=vm, test_metrics=tm,
                source_csv=str(csv_path.relative_to(metrics_dir.parent.parent)
                                if metrics_dir.parent.parent in csv_path.parents
                                else csv_path),
            )
    return load_leaderboard(parquet_path)


__all__ = [
    "CANONICAL_SCHEMA",
    "load_leaderboard",
    "append_row",
    "to_latex",
    "per_quarter_table",
    "family_of",
    "reconcile_metrics_csvs",
]
