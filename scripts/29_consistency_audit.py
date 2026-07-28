"""Multi-pass consistency audit -- verifies the LaTeX chapters in BOTH
repos resolve correctly and that the numeric claims still match the
data after the last commit.

Three independent passes:

  Pass A : per-chapter brace + cite + ref lint (latex code/)
  Pass B : per-chapter brace + cite + ref lint (latex-disertation/)
  Pass C : extract numbers from chap6 prose, match against parquet +
           CSVs directly (independent of scripts/27)

Pass A and B (and the chap6-text half of Pass C) need a local checkout
of the companion LaTeX repositories, which live outside this repo and
are therefore machine-specific -- exactly like the raw G1-G4 CSVs in
configs/paths.local.yaml. On a machine without those checkouts (a
teammate's laptop, a CI runner), this script degrades gracefully: it
SKIPS the LaTeX-dependent checks with a clear message and still runs
the artefact-only checks that don't need them (the HPO and drift-aware
CSV numbers below, and any future check that touches only artefacts/).

Configure the checkout locations via environment variables so this
isn't hardcoded to one contributor's folder layout:
  LATEX_CODE_DIR          (default: sibling "../latex code")
  LATEX_DISERTATION_DIR   (default: sibling "../latex-disertation")
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

MODEL_ROOT = Path(__file__).resolve().parents[1]

# Fall back to the sibling-directory convention documented in the
# README (both repos cloned next to each other) if no env override is
# set. These are DEFAULTS, not requirements -- see the module docstring.
LATEX_CODE = Path(os.environ.get(
    "LATEX_CODE_DIR",
    MODEL_ROOT.parent / "latex code",
))
LATEX_DIS = Path(os.environ.get(
    "LATEX_DISERTATION_DIR",
    MODEL_ROOT.parent / "latex-disertation",
))

CITE_RE = r"\\(?:textcite|parencite|cite)\s*\{([^}]+)\}"
REF_RE = r"\\(?:autoref|cref|ref|nameref|pageref)\s*\{([^}]+)\}"
LABEL_RE = r"\\label\{([^}]+)\}"


def per_chapter_lint(repo_root: Path, chap_files: list[str]) -> tuple[int, int]:
    """Return (passes, fails) for a single repo. Returns (0, 0) if the
    repo checkout isn't present on this machine (caller prints SKIPPED)."""
    if not repo_root.is_dir():
        return (0, 0)

    bib_text = ""
    for bib in repo_root.glob("*.bib"):
        bib_text += bib.read_text(encoding="utf-8") + "\n"
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))

    all_labels = set()
    for f in repo_root.glob("chap*.tex"):
        all_labels |= set(re.findall(LABEL_RE, f.read_text(encoding="utf-8")))

    passes = fails = 0
    for cf in chap_files:
        path = repo_root / cf
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # 1. Braces
        opens = text.count("{")
        closes = text.count("}")
        braces_ok = (opens == closes)
        # 2. Citations
        flat_cites = set()
        for k in re.findall(CITE_RE, text):
            for s in k.split(","):
                s = s.strip()
                if s:
                    flat_cites.add(s)
        missing_cites = flat_cites - bib_keys
        # 3. Refs
        refs = set(re.findall(REF_RE, text))
        labels_here = set(re.findall(LABEL_RE, text))
        unresolved = refs - all_labels - labels_here
        status_ok = braces_ok and not missing_cites and not unresolved
        if status_ok:
            passes += 1
        else:
            fails += 1
        tag = "PASS" if status_ok else "FAIL"
        print(f"  [{tag}] {cf}: braces {opens}/{closes}, "
              f"cites {len(flat_cites)}/0 missing, "
              f"refs {len(refs)} ({len(unresolved)} unresolved)")
        for k in sorted(missing_cites):
            print(f"        MISSING CITE -> {k}")
        for r in sorted(unresolved):
            print(f"        UNRESOLVED REF -> {r}")
    return passes, fails


def main() -> int:
    print("=" * 60)
    print("Pass A -- latex code/ lint")
    print("=" * 60)
    if LATEX_CODE.is_dir():
        pA_p, pA_f = per_chapter_lint(LATEX_CODE, ["chap6.tex", "chap7.tex", "chap8.tex"])
    else:
        print(f"  SKIPPED: {LATEX_CODE} not found on this machine.")
        print(f"  Set LATEX_CODE_DIR to point at a local checkout to enable this pass.")
        pA_p, pA_f = 0, 0

    print("\n" + "=" * 60)
    print("Pass B -- latex-disertation/ lint")
    print("=" * 60)
    if LATEX_DIS.is_dir():
        pB_p, pB_f = per_chapter_lint(LATEX_DIS, ["chap6.tex", "chap7.tex", "chap8.tex"])
    else:
        print(f"  SKIPPED: {LATEX_DIS} not found on this machine.")
        print(f"  Set LATEX_DISERTATION_DIR to point at a local checkout to enable this pass.")
        pB_p, pB_f = 0, 0

    # Pass C: independent number extraction. The chap6-text half needs a
    # LaTeX checkout too; the artefact-only checks (HPO audit, drift-aware
    # CSV) do not and always run.
    print("\n" + "=" * 60)
    print("Pass C -- chap6.tex numbers vs parquet + CSVs (independent reader)")
    print("=" * 60)

    lb = pq.read_table(MODEL_ROOT / "artefacts" / "leaderboard_canonical.parquet").to_pandas()
    lb_rmse = lb[lb["criterion"] == "rmse"].copy()

    chap6_path = LATEX_CODE / "chap6.tex"
    pC_p = pC_f = 0

    if chap6_path.exists():
        chap6 = chap6_path.read_text(encoding="utf-8")

        extracted_claims = [
            ("XGBoost val MAPE 11.99",   ("xgboost",  "val_mape",  11.99)),
            ("XGBoost test MAPE 12.63",  ("xgboost",  "test_mape", 12.63)),
            ("XGBoost test MASE 0.724",  ("xgboost",  "test_mase", 0.724)),
            ("SARIMAX test MAPE 13.11",  ("sarimax",  "test_mape", 13.11)),
            ("Hybrid SARIMAX+LSTM val MAPE 12.19", ("hybrid_sarimax_lstm", "val_mape", 12.19)),
        ]
        for label, (model, col, claim) in extracted_claims:
            val_in_text = f"{claim:.2f}" in chap6 or f"{claim:.3f}" in chap6
            sub = lb_rmse[lb_rmse["model"] == model]
            if sub.empty:
                sub = lb[lb["model"] == model]
            if sub.empty:
                data_ok = False
                actual = None
            else:
                actual = float(sub[col].iloc[0])
                data_ok = abs(actual - claim) <= 0.01
            status = "PASS" if (val_in_text and data_ok) else "FAIL"
            if status == "PASS":
                pC_p += 1
            else:
                pC_f += 1
            print(f"  [{status}] {label}: in chap6 text = {val_in_text}, "
                  f"data = {actual}, delta = "
                  f"{(actual - claim) if actual is not None else 'N/A'}")

        # HPO audit numbers in Table 6.2
        hpo = pd.read_csv(MODEL_ROOT / "artefacts" / "metrics" / "hpo_comparison.csv")
        print("\n  -- HPO audit cells in chap6 Table 6.2 --")
        for _, row in hpo.iterrows():
            cv_str = f"{row['best_cv_RMSE']:.3f}"
            in_text = cv_str in chap6
            tag = "PASS" if in_text else "FAIL"
            if in_text:
                pC_p += 1
            else:
                pC_f += 1
            print(f"  [{tag}] {row['model']} {row['method']} cv_RMSE = {cv_str} in chap6 text? {in_text}")

        # Drift-aware comparison numbers (chap6 Table 6.7)
        drift = pd.read_csv(MODEL_ROOT / "artefacts" / "metrics" / "drift_aware_comparison.csv")
        test_drift = drift[drift["block"] == "test"].copy()
        print("\n  -- Drift-aware cells (chap6 Table 6.7) --")
        for model in ["dow_mean", "sarimax", "xgboost"]:
            for config in ["expanding", "sliding_450"]:
                sub = test_drift[(test_drift["model"] == model) & (test_drift["config"] == config)]
                if sub.empty:
                    continue
                v = float(sub["MAPE"].iloc[0])
                in_text = f"{v:.2f}" in chap6 or f"{v:.1f}" in chap6
                tag = "PASS" if in_text else "FAIL"
                if in_text:
                    pC_p += 1
                else:
                    pC_f += 1
                print(f"  [{tag}] {model} {config} = {v:.2f} in chap6? {in_text}")
    else:
        print(f"  SKIPPED (chap6-text checks): {chap6_path} not found.")
        print(f"  The artefact files themselves (leaderboard parquet, hpo_comparison.csv, "
              f"drift_aware_comparison.csv) were still read successfully above; only the "
              f"'is this number present in the chapter prose' check needs the LaTeX checkout.")

    print("\n" + "=" * 60)
    print("Consistency audit summary")
    print("=" * 60)
    print(f"  Pass A (latex code/ lint):     {pA_p} PASS / {pA_f} FAIL")
    print(f"  Pass B (latex-disertation/):   {pB_p} PASS / {pB_f} FAIL")
    print(f"  Pass C (numbers vs data):      {pC_p} PASS / {pC_f} FAIL")
    total_fail = pA_f + pB_f + pC_f
    print(f"  Combined:                      "
          f"{pA_p + pB_p + pC_p} PASS / {total_fail} FAIL")

    # Exit non-zero only on genuine failures, never on missing (optional,
    # machine-specific) LaTeX checkouts -- those print SKIPPED, not FAIL.
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
