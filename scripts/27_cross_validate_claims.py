"""Cross-validation pass: every number cited in chap6/7/8.tex against the
actual artefacts on disk.

Runs four independent checks and prints PASS / FAIL per claim, with the
delta between the claim and the on-disk number.

Sources cross-referenced:
  - artefacts/leaderboard_canonical.parquet         (parquet leaderboard)
  - artefacts/metrics/uq_coverage_aci.csv           (ACI sweep)
  - artefacts/metrics/task2_standalone_metrics.csv  (5 specialties x 6 models)
  - artefacts/metrics/aggregated_metrics_period.csv (weekly/monthly/yearly)
  - artefacts/metrics/augmented_random_search.csv   (Sec 6ter)
  - artefacts/metrics/test_per_quarter.csv          (OOD honesty)
  - artefacts/metrics/test_per_horizon.csv          (per-horizon)
  - artefacts/metrics/hpo_comparison.csv            (HPO audit)
  - artefacts/metrics/{model}_rmse_metrics.csv      (per-model RMSE rerun)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.05  # tolerance in percentage points for MAPE / units for RMSE

passes, fails, warnings = 0, 0, 0


def check(label, claim, actual, tol=TOL, units=""):
    """Compare a claimed value to the on-disk actual."""
    global passes, fails, warnings
    if actual is None or pd.isna(actual):
        warnings += 1
        print(f"  [WARN] {label}: claim={claim} {units}  on-disk=N/A")
        return
    delta = float(claim) - float(actual)
    ok = abs(delta) <= tol
    tag = "[PASS]" if ok else "[FAIL]"
    if ok:
        passes += 1
    else:
        fails += 1
    print(f"  {tag} {label}: claim={claim} {units}  actual={actual:.3f}  Δ={delta:+.3f}")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =====================================================================
# 1. Leaderboard cross-check (chap6 Table 6.6)
# =====================================================================
section("1. Final leaderboard claims (chap6 Tab 6.6) vs parquet")
import pyarrow.parquet as pq
lb = pq.read_table(ROOT / "artefacts" / "leaderboard_canonical.parquet").to_pandas()
lb_rmse = lb[lb["criterion"] == "rmse"].copy()
print(f"Loaded {len(lb)} leaderboard rows ({len(lb_rmse)} under RMSE criterion).")

# Claims from chap6 Table 6.6 final leaderboard
claims_lb = [
    ("XGBoost val MAPE", "xgboost", "val_mape", 11.99),
    ("XGBoost test MAPE", "xgboost", "test_mape", 12.63),
    ("XGBoost val RMSE", "xgboost", "val_rmse", 9.35),
    ("XGBoost test RMSE", "xgboost", "test_rmse", 10.30),
    ("SARIMAX val MAPE", "sarimax", "val_mape", 12.52),
    ("SARIMAX test MAPE", "sarimax", "test_mape", 13.11),
    ("ANN val MAPE (RMSE-tuned)", "ann", "val_mape", 11.90),
    ("ANN test MAPE (RMSE-tuned)", "ann", "test_mape", 13.24),
    ("LSTM val MAPE (RMSE-tuned)", "lstm", "val_mape", 12.31),
    ("LSTM test MAPE (RMSE-tuned)", "lstm", "test_mape", 13.76),
    ("ARIMA val MAPE", "arima", "val_mape", 13.33),
    ("NB-GLM val MAPE", "nbglm", "val_mape", 12.65),
    ("Hybrid SARIMAX+LSTM val MAPE", "hybrid_sarimax_lstm", "val_mape", 12.19),
    ("Hybrid SARIMAX+LSTM test MAPE", "hybrid_sarimax_lstm", "test_mape", 12.95),
]
for label, model, col, claim in claims_lb:
    # Prefer rmse criterion rows since those are the RMSE-best winners
    sub = lb_rmse[lb_rmse["model"] == model]
    if sub.empty:
        sub = lb[lb["model"] == model]
    if sub.empty:
        check(label, claim, None)
        continue
    check(label, claim, sub[col].iloc[0])

# MASE column in chap6 Table 6.6 -- post scripts/28, the parquet has values.
# Claims here are the *corrected* chap6 values, not the original guesses.
print("\nMASE column (test_mase) check -- corrected chap6 values vs computed parquet:")
for label, model, claim in [
    ("XGBoost test MASE", "xgboost", 0.724),
    ("SARIMAX test MASE", "sarimax", 0.732),
    ("Hybrid SARIMAX+LSTM test MASE", "hybrid_sarimax_lstm", 0.738),
]:
    sub = lb_rmse[lb_rmse["model"] == model]
    if sub.empty:
        sub = lb[lb["model"] == model]
    if sub.empty:
        check(label, claim, None)
        continue
    check(label, claim, sub["test_mase"].iloc[0])

# =====================================================================
# 2. ACI table (chap6 Table 6.4) vs uq_coverage_aci.csv
# =====================================================================
section("2. ACI claims (chap6 Tab 6.4) vs uq_coverage_aci.csv")
aci = pd.read_csv(ROOT / "artefacts" / "metrics" / "uq_coverage_aci.csv")
aci_95 = aci[aci["level"] == 95].copy()
print(f"Loaded {len(aci_95)} rows at the 95% level.")

aci_claims = [
    # (label, base, method, gamma, col, claim)
    ("XGBoost split / coverage",       "XGBoost", "split", 0.0,    "coverage", 0.924),
    ("XGBoost split / mean width",     "XGBoost", "split", 0.0,    "mean_width", 36.13),
    ("XGBoost split / Winkler",        "XGBoost", "split", 0.0,    "winkler", 48.31),
    ("XGBoost ACI 0.005 / coverage",   "XGBoost", "aci",   0.005,  "coverage", 0.942),
    ("XGBoost ACI 0.005 / mean width", "XGBoost", "aci",   0.005,  "mean_width", 38.79),
    ("XGBoost ACI 0.005 / Winkler",    "XGBoost", "aci",   0.005,  "winkler", 48.02),
    ("SARIMAX split / coverage",       "SARIMAX", "split", 0.0,    "coverage", 0.922),
    ("SARIMAX split / Winkler",        "SARIMAX", "split", 0.0,    "winkler", 50.70),
    ("SARIMAX ACI 0.005 / coverage",   "SARIMAX", "aci",   0.005,  "coverage", 0.942),
    ("SARIMAX ACI 0.005 / Winkler",    "SARIMAX", "aci",   0.005,  "winkler", 50.44),
    ("Hybrid split / coverage",        "Hybrid_SARIMAX_LSTM", "split", 0.0,   "coverage", 0.914),
    ("Hybrid split / Winkler",         "Hybrid_SARIMAX_LSTM", "split", 0.0,   "winkler", 50.81),
    ("Hybrid ACI 0.001 / coverage",    "Hybrid_SARIMAX_LSTM", "aci",   0.001, "coverage", 0.927),
    ("Hybrid ACI 0.001 / Winkler",     "Hybrid_SARIMAX_LSTM", "aci",   0.001, "winkler", 50.36),
]
for label, base, method, gamma, col, claim in aci_claims:
    sub = aci_95[
        (aci_95["base"] == base)
        & (aci_95["method"] == method)
        & (aci_95["gamma"].round(3) == round(gamma, 3))
    ]
    if sub.empty:
        check(label, claim, None)
        continue
    check(label, claim, sub[col].iloc[0])

# =====================================================================
# 3. Task 2 specialty numbers (chap8 NHI section + chap6 mention)
# =====================================================================
section("3. Task 2 specialty test MAPE (chap8 + RESULTS Sec 6quater)")
t2 = pd.read_csv(ROOT / "artefacts" / "metrics" / "task2_standalone_metrics.csv")
print(f"Loaded {len(t2)} rows.")

# Best-per-specialty test MAPE per memory + RESULTS.md
t2_claims = [
    ("Medicine XGBoost test MAPE",       "Medicine",     "XGBoost", 18.86),
    ("Orthopaedics ARIMA test MAPE",     "Orthopaedics", "ARIMA",   82.18),
    ("Surgery NB-GLM test MAPE",         "Surgery",      "NB-GLM",  55.28),
    ("Paediatrics NB-GLM test MAPE",     "Paediatrics",  "NB-GLM",  48.58),
    ("Gynaecology NB-GLM test MAPE",     "Gynaecology",  "NB-GLM",  45.10),
]
for label, spec, model, claim in t2_claims:
    sub = t2[(t2["specialty"] == spec) & (t2["model"] == model) & (t2["block"] == "test")]
    if sub.empty:
        check(label, claim, None)
        continue
    check(label, claim, sub["MAPE"].iloc[0])

# =====================================================================
# 4. Aggregated weekly/monthly MAPE (chap7 + chap8)
# =====================================================================
section("4. Aggregated weekly/monthly MAPE (chap7)")
agg_path = ROOT / "artefacts" / "metrics" / "aggregated_metrics_period.csv"
if agg_path.exists():
    agg = pd.read_csv(agg_path)
    print(f"Loaded {len(agg)} rows.")
    print(agg.head(15).to_string(index=False))
    # chap7 cites: weekly 5.89%, monthly 3.47% (XGBoost RMSE-best on test)
    # Find them
    sub_w = agg[(agg["period"] == "weekly") & (agg["block"] == "test") &
                  (agg["model"].str.contains("XGBoost", case=False, na=False))]
    sub_m = agg[(agg["period"] == "monthly") & (agg["block"] == "test") &
                  (agg["model"].str.contains("XGBoost", case=False, na=False))]
    if not sub_w.empty:
        check("XGBoost weekly test MAPE (chap7)", 5.89, sub_w["MAPE"].iloc[0])
    if not sub_m.empty:
        check("XGBoost monthly test MAPE (chap7)", 3.47, sub_m["MAPE"].iloc[0])
else:
    print(f"  [WARN] {agg_path} not found; chap7 weekly/monthly numbers unverifiable.")

# =====================================================================
# 5. Per-quarter SARIMAX (chap6 Table 6.7)
# =====================================================================
section("5. Per-quarter SARIMAX test MAPE (chap6 Tab 6.7)")
pq_path = ROOT / "artefacts" / "metrics" / "test_per_quarter.csv"
if pq_path.exists():
    pq_df = pd.read_csv(pq_path)
    print(pq_df.head(10).to_string(index=False))
    pq_claims = [
        ("SARIMAX 2025Q1", "2025Q1", 12.73),
        ("SARIMAX 2025Q2", "2025Q2", 12.70),
        ("SARIMAX 2025Q3", "2025Q3", 13.56),
        ("SARIMAX 2025Q4", "2025Q4", 13.90),
        ("SARIMAX 2026Q1", "2026Q1", 11.67),
    ]
    for label, q, claim in pq_claims:
        sub = pq_df[(pq_df["model"] == "sarimax") & (pq_df["quarter"] == q)]
        if sub.empty:
            check(label, claim, None)
            continue
        check(label, claim, sub["MAPE"].iloc[0])

# =====================================================================
# 6. Per-horizon recursive XGBoost (chap6 Table 6.3)
# =====================================================================
section("6. Per-horizon recursive XGBoost test MAPE (chap6 Tab 6.3)")
ph_path = ROOT / "artefacts" / "metrics" / "test_per_horizon.csv"
if ph_path.exists():
    ph = pd.read_csv(ph_path)
    print(ph.head(15).to_string(index=False))
    # Corrected per-horizon XGBoost claims (from chap6 Table 6.3 after fill_gaps).
    ph_claims = [(f"h={h}", h, claim) for h, claim in
                  zip(range(1, 8), [13.80, 12.31, 13.41, 14.29, 12.02, 11.38, 11.16])]
    for label, h, claim in ph_claims:
        sub = ph[(ph["model"].str.contains("xgboost", case=False, na=False)) &
                  (ph["horizon"] == h)]
        if sub.empty:
            check(label, claim, None)
            continue
        check(label, claim, sub["MAPE"].iloc[0])

# =====================================================================
# 7. HPO fairness audit (chap6 Table 6.2)
# =====================================================================
section("7. HPO fairness audit cv_RMSE (chap6 Tab 6.2)")
hpo_path = ROOT / "artefacts" / "metrics" / "hpo_comparison.csv"
if hpo_path.exists():
    hpo = pd.read_csv(hpo_path)
    print(hpo.head(15).to_string(index=False))

# =====================================================================
# 8. Augmented features rejection (RESULTS Sec 6ter)
# =====================================================================
section("8. Augmented features run (RESULTS Sec 6ter)")
aug = pd.read_csv(ROOT / "artefacts" / "metrics" / "augmented_random_search.csv")
print(aug[["model", "cv_MAPE", "val_MAPE", "test_MAPE", "val_RMSE", "test_RMSE"]].to_string(index=False))
aug_claims = [
    ("Augmented XGBoost val MAPE", "XGBoost", "val_MAPE", 12.22),
    ("Augmented XGBoost test MAPE", "XGBoost", "test_MAPE", 13.01),
    ("Augmented ANN val MAPE", "ANN", "val_MAPE", 12.03),
    ("Augmented ANN test MAPE", "ANN", "test_MAPE", 14.73),
    ("Augmented ANN cv_MAPE (sub-10% claim)", "ANN", "cv_MAPE", 9.76),
    ("Augmented LSTM val MAPE", "LSTM", "val_MAPE", 12.75),
    ("Augmented LSTM test MAPE", "LSTM", "test_MAPE", 14.93),
]
for label, model, col, claim in aug_claims:
    sub = aug[aug["model"] == model]
    if sub.empty:
        check(label, claim, None)
        continue
    check(label, claim, sub[col].iloc[0])

# =====================================================================
# Summary
# =====================================================================
section("Summary")
total = passes + fails + warnings
print(f"  Passes:   {passes}")
print(f"  Fails:    {fails}")
print(f"  Warnings: {warnings}")
print(f"  Total:    {total}")
print()
if fails > 0:
    print("REVIEW required: failing claims listed above.")
else:
    print("All numeric claims within tolerance.")
