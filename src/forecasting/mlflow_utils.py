"""Thin MLflow wrapper for the modelling pipeline's experiment tracking.

Every training script that adopts this wrapper logs, per run:
  - params:  the hyperparameters actually used (dict, any JSON-safe values)
  - metrics: val/test MAPE, RMSE, MAE, MASE (float, any subset)
  - tags:    model family, HPO method, git commit (best-effort)
  - artefacts (optional): the prediction CSVs, so a run's exact output
    is attached to its params/metrics in one place

Design choice: MLflow is optional (see requirements-optional.txt). This
module degrades to a no-op context manager when the mlflow package
isn't installed, so scripts that import it don't gain a hard
dependency -- `python scripts/30_random_forest_baseline.py` still runs
on a machine that never installed the optional tracking stack, it just
skips logging with a one-line notice.

Usage:
    from src.forecasting.mlflow_utils import log_run

    with log_run("random_forest", params=best_params) as run:
        ... fit + predict ...
        run.log_metrics({"val_MAPE": 12.73, "test_MAPE": 14.26})
        run.log_artefact(ROOT / "artefacts" / "predictions" / "random_forest.csv")
"""
from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any

# MLflow >=3.0 deprecated the plain filesystem tracking backend (the
# `file:./mlruns` URI this module uses) in favour of a SQLite/database
# backend, and raises MlflowException instead of a warning unless this
# opt-out is set. A local file store is the right choice for a thesis
# repo -- no server process, works identically on Windows/Mac/Linux and
# inside the Docker image, diff-friendly enough for a small number of
# runs -- so we opt out deliberately rather than add a SQLite file
# dependency. Must be set BEFORE importing mlflow.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"file:{(ROOT / 'mlruns').as_posix()}"
EXPERIMENT_NAME = "msc-modelling-ch6"


def _git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class _NoOpRun:
    """Returned when mlflow isn't installed -- every call is a silent no-op
    so calling code doesn't need an `if mlflow_available` branch."""

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artefact(self, path: Path) -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        pass


class _MlflowRun:
    """Thin wrapper around an active mlflow run, exposing the subset of
    the API this project actually uses."""

    def __init__(self, active_run):
        self._run = active_run

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow rejects values it can't stringify meaningfully for some
        # types (e.g. numpy scalars); cast defensively.
        safe = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                 for k, v in params.items()}
        mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        # `v == v` filters NaN (NaN != NaN); MLflow's log_metrics errors on NaN.

    def log_artefact(self, path: Path) -> None:
        if Path(path).exists():
            mlflow.log_artifact(str(path))

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)


@contextlib.contextmanager
def log_run(model_name: str, *, params: dict[str, Any] | None = None,
            family: str = "", criterion: str = "rmse"):
    """Context manager: opens an MLflow run tagged with the model name,
    family, HPO criterion, and current git commit; logs `params` up
    front if given. Yields an object with .log_metrics/.log_artefact/
    .set_tag, all safe no-ops if mlflow isn't installed.

    Example:
        with log_run("random_forest", params=best_params, family="ml") as run:
            ...
            run.log_metrics({"val_MAPE": 12.73, "test_MAPE": 14.26})
    """
    if not _MLFLOW_AVAILABLE:
        print(f"  [mlflow_utils] mlflow not installed -- skipping experiment "
              f"tracking for '{model_name}' (pip install mlflow to enable).")
        yield _NoOpRun()
        return

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name) as active_run:
        run = _MlflowRun(active_run)
        run.set_tag("family", family)
        run.set_tag("criterion", criterion)
        run.set_tag("git_commit", _git_commit_short())
        if params:
            run.log_params(params)
        yield run
