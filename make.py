"""Cross-platform one-command runbook for the MSc modelling repo.

A non-developer can reproduce every numeric claim with these commands:

    python make.py setup     # install core dependencies (one-time, fast)
    python make.py verify    # confirm the data plumbing works
    python make.py test      # run the unit-test suite (50 tests)
    python make.py crossval  # cross-validate every numeric claim
    python make.py pipeline  # full modelling pipeline (~6-10 hours)

Reproducibility platform commands (see README "Reproducibility
platform" section for the full picture and its honest limits):

    python make.py docker-build     # build the Docker image
    python make.py docker-crossval  # run crossval INSIDE the container
    python make.py dvc-dag          # print the DVC pipeline graph
    python make.py dvc-repro        # re-run any stale DVC stage
    python make.py mlflow-ui        # browse logged experiment runs

Run `python make.py help` for the full list.

Why a Python script instead of a Makefile: Make is not pre-installed on
Windows, and this repo's primary user is on Windows. Python is.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # use whichever Python invoked this script


def _sh(cmd: list[str], cwd: Path = ROOT, allow_fail: bool = False) -> int:
    """Run a subprocess, stream stdout/stderr, return the exit code."""
    print(f"\n$ {' '.join(cmd)}")
    # Force UTF-8 stdio so scripts that print Greek/maths symbols (delta,
    # gamma, plus/minus) do not crash on Windows cp1252 consoles.
    import os
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0 and not allow_fail:
        print(f"  -> exit code {proc.returncode}")
        sys.exit(proc.returncode)
    return proc.returncode


def cmd_setup():
    """Install CORE dependencies only (requirements-core.txt) -- fast."""
    print("Installing core dependencies (pipeline + tests + crossval).")
    print("For the reproducibility platform stack (DVC/MLflow/Hydra) or the "
          "Nixtla/DeepAR extras, run: python make.py setup-full")
    _sh([PY, "-m", "pip", "install", "-r", "requirements-core.txt"])


def cmd_setup_full():
    """Install core + optional dependencies (DVC, MLflow, Hydra, Nixtla, DeepAR)."""
    _sh([PY, "-m", "pip", "install", "-r", "requirements-core.txt"])
    _sh([PY, "-m", "pip", "install", "-r", "requirements-optional.txt"])


def cmd_verify():
    """Quick sanity checks: imports, paths, splits."""
    print("Verifying data plumbing and split configuration...")
    _sh([PY, "-c",
         "from src.forecasting.io import load_g1, Splits; "
         "splits = Splits.from_config(); g1 = load_g1(); "
         "print(f'OK: G1 loaded {len(g1)} rows; splits train/val/test = '"
         "f'{len(splits.slice(g1, \"train\"))}/{len(splits.slice(g1, \"val\"))}'"
         "f'/{len(splits.slice(g1, \"test\"))}')"])


def cmd_test():
    """Run the unit-test suite. 50 tests; should finish in ~30 seconds."""
    _sh([PY, "-m", "pytest", "tests/", "-q", "--no-header"])


def cmd_crossval():
    """Cross-validate every numeric claim in chap6/7/8 against the on-disk artefacts.

    This is the audit script -- it ingests the parquet leaderboard, the
    per-model metrics CSVs, and the ACI / drift-aware tables, and prints
    PASS / FAIL / WARN per claim. Target: 57 PASS / 0 FAIL / 0 WARN.
    """
    _sh([PY, "scripts/27_cross_validate_claims.py"])


def cmd_fill_gaps():
    """Compute MASE + per-horizon XGBoost + missing test MAPE rows.

    Updates artefacts/leaderboard_canonical.parquet in place. Idempotent.
    """
    _sh([PY, "scripts/28_fill_gaps.py"])


def cmd_leaderboard():
    """Rebuild the canonical leaderboard from all *_metrics.csv files."""
    _sh([PY, "-c",
         "from src.forecasting.leaderboard import reconcile_metrics_csvs; "
         "df = reconcile_metrics_csvs(); "
         "print(df[['model','family','criterion','val_mape','test_mape',"
         "'test_mase']].head(10).to_string(index=False))"])


def cmd_pipeline():
    """Run the full modelling pipeline (LONG - several hours)."""
    print("Running the full modelling pipeline. This will take ~6-10 hours.")
    print("Each script writes its outputs to artefacts/{predictions,metrics,figures}/.")
    print("If interrupted, re-run individual scripts by hand; they are idempotent.")
    stages = [
        ("Naive baselines",          "01_reference_floor.py"),
        ("ARIMA",                    "02_arima.py"),
        ("SARIMAX",                  "03_sarimax.py"),
        ("NB GLM",                   "04_nbglm.py"),
        ("XGBoost (Grid CV)",        "06_xgboost.py"),
        ("ANN (Random CV)",          "07_ann.py"),
        ("LSTM (Optuna CV)",         "08_lstm.py"),
        ("Hybrids (legacy)",         "09_hybrids.py"),
        ("LSTM+XGB hybrid",          "11_lstm_xgb_hybrid.py"),
        ("Master leaderboard",       "10_master_leaderboard.py"),
        ("Ablation",                 "12_ablation.py"),
        ("Task 2 specialties",       "14_task2_specialties.py"),
        ("Task 2 ML/hybrids",        "15_task2_ml_hybrids.py"),
        ("Ensembles",                "16_ensembles.py"),
        ("Final test pass",          "17_final_test.py"),
        ("HPO fairness audit",       "18_hpo_comparison.py"),
        ("RMSE-best rerun",          "19_rerun_rmse_best.py"),
        ("Hybrids RMSE rebuild",     "20_rebuild_hybrids_rmse.py"),
        ("UQ baseline",              "21_uncertainty_quantification.py"),
        ("Augmented features",       "22_augmented_features_random_search.py"),
        ("Task 2 standalone (full)", "23_task2_standalone_models.py"),
        ("OOF hybrids",              "24_oof_hybrids.py"),
        ("Drift-aware refit",        "25_drift_aware_refit.py"),
        ("ACI UQ",                   "26_aci_uq.py"),
        ("Fill MASE / per-horizon",  "28_fill_gaps.py"),
        ("Cross-validation audit",   "27_cross_validate_claims.py"),
    ]
    for i, (label, script) in enumerate(stages, 1):
        print(f"\n[{i}/{len(stages)}] {label}  ({script})")
        rc = _sh([PY, f"scripts/{script}"], allow_fail=True)
        if rc != 0:
            print(f"  WARNING: {script} returned {rc}; continuing.")


def cmd_docker_build():
    """Build the Docker image (Dockerfile -> msc-modelling:latest)."""
    _sh(["docker", "build", "-t", "msc-modelling:latest", "."])


def cmd_docker_test():
    """Run the unit-test suite INSIDE a container (no host Python needed)."""
    _sh(["docker", "compose", "run", "--rm", "pipeline", "python", "make.py", "test"])


def cmd_docker_crossval():
    """Run the cross-validation audit INSIDE a container (no host Python needed)."""
    _sh(["docker", "compose", "run", "--rm", "pipeline", "python", "make.py", "crossval"])


def cmd_dvc_dag():
    """Print the DVC pipeline dependency graph (crossval + consistency_audit).

    See dvc.yaml's header comment for why the 31 model-training scripts
    are NOT part of this DAG (they need confidential raw data that can't
    be committed) and why scripts/28_fill_gaps.py is also excluded (it
    patches an existing file in place, which conflicts with DVC's
    clear-outputs-then-regenerate stage contract).
    """
    _sh([PY, "-m", "dvc", "dag"])


def cmd_dvc_repro():
    """Re-run any DVC stage whose dependencies have changed since the last run."""
    _sh([PY, "-m", "dvc", "repro"])


def cmd_dvc_status():
    """Show which DVC stages are stale without running them."""
    _sh([PY, "-m", "dvc", "status"], allow_fail=True)


def cmd_mlflow_ui():
    """Open the MLflow experiment-tracking UI (http://localhost:5000).

    Reads from ./mlruns, populated by scripts that call
    src.forecasting.mlflow_utils.log_run() (currently: scripts/30). Runs
    logged before this wrapper existed are not in mlruns/ -- MLflow only
    tracks what was logged at run time; historical runs weren't
    retroactively backfilled since that would mean re-running expensive
    (hours-long) training scripts just to produce tracking metadata.
    """
    _sh([PY, "-m", "mlflow", "ui", "--backend-store-uri",
          f"file:{(ROOT / 'mlruns').as_posix()}"])


def cmd_clean():
    """Remove generated artefacts (cautious: keeps committed files only)."""
    print("Removing test cache, pyc files. Run-output CSVs in "
          "artefacts/metrics/ and artefacts/predictions/ are NOT removed; "
          "they are tracked by git on this branch.")
    for pat in ["**/__pycache__", "**/*.pyc", ".pytest_cache"]:
        for path in ROOT.rglob(pat):
            if path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    print("  done.")


COMMANDS = {
    "setup":            cmd_setup,
    "setup-full":       cmd_setup_full,
    "verify":           cmd_verify,
    "test":             cmd_test,
    "crossval":         cmd_crossval,
    "fill-gaps":        cmd_fill_gaps,
    "leaderboard":      cmd_leaderboard,
    "pipeline":         cmd_pipeline,
    "docker-build":     cmd_docker_build,
    "docker-test":      cmd_docker_test,
    "docker-crossval":  cmd_docker_crossval,
    "dvc-dag":          cmd_dvc_dag,
    "dvc-repro":        cmd_dvc_repro,
    "dvc-status":       cmd_dvc_status,
    "mlflow-ui":        cmd_mlflow_ui,
    "clean":            cmd_clean,
}


def cmd_help():
    print(__doc__)
    print("\nAvailable commands:")
    for name, fn in COMMANDS.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:14s}  {doc}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        cmd_help()
        return
    cmd = sys.argv[1]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"Unknown command: {cmd}")
        cmd_help()
        sys.exit(2)
    fn()


if __name__ == "__main__":
    main()
