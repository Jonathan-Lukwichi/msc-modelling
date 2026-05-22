"""Multi-pass consistency audit -- verifies the LaTeX chapters in BOTH
repos resolve correctly and that the numeric claims still match the
data after the last commit.

Three independent passes:

  Pass A : per-chapter brace + cite + ref lint (latex code/)
  Pass B : per-chapter brace + cite + ref lint (latex-disertation/)
  Pass C : extract numbers from chap6 prose, match against parquet +
           CSVs directly (independent of scripts/27)
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

MODEL_ROOT = Path(__file__).resolve().parents[1]
LATEX_CODE = Path(r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\latex code")
LATEX_DIS = Path(r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\latex-disertation")

CITE_RE = r"\\(?:textcite|parencite|cite)\s*\{([^}]+)\}"
REF_RE = r"\\(?:autoref|cref|ref|nameref|pageref)\s*\{([^}]+)\}"
LABEL_RE = r"\\label\{([^}]+)\}"


def per_chapter_lint(repo_root: Path, chap_files: list[str]) -> tuple[int, int]:
    """Return (passes, fails) for a single repo."""
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


print("=" * 60)
print("Pass A -- latex code/ lint")
print("=" * 60)
pA_p, pA_f = per_chapter_lint(LATEX_CODE, ["chap6.tex", "chap7.tex", "chap8.tex"])

print("\n" + "=" * 60)
print("Pass B -- latex-disertation/ lint")
print("=" * 60)
pB_p, pB_f = per_chapter_lint(LATEX_DIS, ["chap6.tex", "chap7.tex", "chap8.tex"])

# Pass C: independent number extraction
print("\n" + "=" * 60)
print("Pass C -- chap6.tex numbers vs parquet + CSVs (independent reader)")
print("=" * 60)

lb = pq.read_table(MODEL_ROOT / "artefacts" / "leaderboard_canonical.parquet").to_pandas()
lb_rmse = lb[lb["criterion"] == "rmse"].copy()

# Extract chap6 from latex code (the user's primary working copy)
chap6 = (LATEX_CODE / "chap6.tex").read_text(encoding="utf-8")

# Look for headline cells the chapter SAYS are true and confirm against data
extracted_claims = [
    ("XGBoost val MAPE 11.99",   ("xgboost",  "val_mape",  11.99)),
    ("XGBoost test MAPE 12.63",  ("xgboost",  "test_mape", 12.63)),
    ("XGBoost test MASE 0.724",  ("xgboost",  "test_mase", 0.724)),
    ("SARIMAX test MAPE 13.11",  ("sarimax",  "test_mape", 13.11)),
    ("Hybrid SARIMAX+LSTM val MAPE 12.19", ("hybrid_sarimax_lstm", "val_mape", 12.19)),
]
pC_p = pC_f = 0
for label, (model, col, claim) in extracted_claims:
    # Verify text contains the number to 2 decimal places (within textual rounding)
    val_in_text = f"{claim:.2f}" in chap6 or f"{claim:.3f}" in chap6
    # Verify data matches
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

# HPO audit numbers in the new Table 6.2
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

print("\n" + "=" * 60)
print("Consistency audit summary")
print("=" * 60)
print(f"  Pass A (latex code/ lint):     {pA_p} PASS / {pA_f} FAIL")
print(f"  Pass B (latex-disertation/):   {pB_p} PASS / {pB_f} FAIL")
print(f"  Pass C (numbers vs data):      {pC_p} PASS / {pC_f} FAIL")
print(f"  Combined:                      "
      f"{pA_p + pB_p + pC_p} PASS / {pA_f + pB_f + pC_f} FAIL")
