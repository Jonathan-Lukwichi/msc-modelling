"""Systematic pattern analysis across the literature corpus.

Reads every PDF in the JL_DOCU/document selected folder, extracts the text,
and detects recurring structural patterns in how authors present and discuss
their results.

Output:
  artefacts/lit_review/per_paper_features.csv   one row per paper, ~30 columns
  artefacts/lit_review/pattern_frequencies.csv  aggregate counts + percentages
  artefacts/lit_review/pattern_report.md        human-readable synthesis

Approach:
  For each paper, detect presence/absence of ~30 standard result-presentation
  elements via regex over the extracted text (metrics named, models compared,
  table/figure types referenced, statistical tests run, discussion subsections
  present). Then aggregate counts across the corpus to identify the most
  recurrent patterns. This is the systematic-NLP equivalent of "reading every
  paper carefully", with the advantage that the criteria are explicit and
  reproducible.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import time

import pandas as pd
from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("C:/Users/BIBINBUSINESS/OneDrive/Desktop/JL_DOCU/document selected")
OUT_DIR = ROOT / "artefacts" / "lit_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pattern dictionary — what to detect in each paper
# ---------------------------------------------------------------------------

METRICS = {
    "metric_MAPE":       r"\bMAPE\b",
    "metric_RMSE":       r"\bRMSE\b",
    "metric_MAE":        r"\bMAE\b",
    "metric_R2":         r"\bR\^?2\b|R-squared|R\s?square",
    "metric_MASE":       r"\bMASE\b",
    "metric_sMAPE":      r"\bsMAPE\b",
    "metric_AUC":        r"\bAUC\b|area under (the )?curve",
    "metric_accuracy":   r"\baccuracy\b",
    "metric_F1":         r"\bF1\s?(score)?\b|F-?measure",
    "metric_precision":  r"\bprecision\b",
    "metric_recall":     r"\brecall\b|sensitivity",
    "metric_specificity":r"\bspecificity\b",
    "metric_MSE":        r"\bMSE\b",
    "metric_AIC_BIC":    r"\bAIC\b|\bBIC\b",
}

MODELS = {
    "model_ARIMA":     r"\bARIMA\b|ARMA",
    "model_SARIMA":    r"\bSARIMA[X]?\b",
    "model_Prophet":   r"\bProphet\b",
    "model_LSTM":      r"\bLSTM\b|long short[\- ]term memory",
    "model_CNN":       r"\bCNN\b|convolutional neural",
    "model_GRU":       r"\bGRU\b|gated recurrent",
    "model_RNN":       r"\bRNN\b|recurrent neural",
    "model_ANN_MLP":   r"\bANN\b|\bMLP\b|multilayer perceptron|artificial neural network",
    "model_XGBoost":   r"XGBoost|XGB\b",
    "model_GBM":       r"gradient boost|GBM|LightGBM|CatBoost",
    "model_RF":        r"random forest|\bRF\b",
    "model_SVM":       r"\bSVM\b|support vector",
    "model_ELM":       r"extreme learning machine|\bELM\b",
    "model_GLM":       r"\bGLM\b|generalized linear",
    "model_LinReg":    r"linear regression|\bOLS\b",
    "model_Lasso":     r"\bLASSO\b|elastic[\- ]net|ridge regression",
    "model_NaiveBayes":r"naive bayes",
    "model_LogReg":    r"logistic regression",
    "model_DT":        r"decision tree|\bCART\b",
    "model_KNN":       r"\bKNN\b|k[\- ]nearest",
    "model_Ensemble":  r"\bensemble\b|stacking|bagging|boosting",
    "model_Hybrid":    r"\bhybrid\b|hybrid model|hybrid approach",
    "model_Transformer":r"transformer|attention[\- ]based",
    "model_Naive":     r"\bna[ïi]ve\b (?:baseline|forecast|method)|seasonal na[ïi]ve",
}

PRESENTATION = {
    # Tables & figures
    "has_results_table":    r"Table\s+\d.{0,80}(performance|result|MAPE|RMSE|accuracy|metric|compar)",
    "has_heatmap":          r"heat ?map|colour[\- ]coded|color[\- ]coded",
    "has_actual_vs_pred":   r"actual[\s\w]*(vs|versus|and)[\s\w]*predicted|predicted[\s\w]*(vs|versus)[\s\w]*actual",
    "has_scatter_plot":     r"scatter[\- ]?plot",
    "has_residual_plot":    r"residual[\s\w]*plot|residual[\s\w]*analysis",
    "has_ACF_PACF":         r"\bACF\b|\bPACF\b|autocorrelation function",
    "has_arch_diagram":     r"architecture (of|diagram|figure)|model architecture|proposed (architecture|framework)",
    "has_PRISMA":           r"\bPRISMA\b",
    "has_flowchart":        r"flow ?chart|flow diagram",
    "has_variable_importance": r"variable importance|feature importance|SHAP|permutation importance",
    "has_confusion_matrix": r"confusion matrix",
    "has_box_plot":         r"box ?plot|whisker plot",
    "has_time_series_plot": r"time[\- ]series plot|line plot",
    # Statistical tests
    "test_Diebold_Mariano": r"Diebold[\- ]?Mariano|\bDM test\b",
    "test_Granger":         r"Granger causalit",
    "test_Pearson":         r"Pearson correlation",
    "test_Spearman":        r"Spearman",
    "test_Cointegration":   r"cointegration|Johansen",
    "test_ADF":             r"augmented Dickey[\- ]Fuller|\bADF test\b",
    "test_LjungBox":        r"Ljung[\- ]?Box",
    "test_Wilcoxon":        r"Wilcoxon",
    "test_t_test":          r"\bt[\- ]test\b|paired t",
    "test_KS":              r"Kolmogorov[\- ]Smirnov|\bKS test\b|\bKS distance\b",
    # Validation strategy
    "uses_cross_val":       r"cross[\- ]?valid|\bCV\b|k-?fold",
    "uses_rolling_origin":  r"rolling[\- ]origin|walk[\- ]forward|rolling window",
    "uses_train_test_split":r"train(ing)?[\s\/\-]+(?:and )?test(ing)?\s+(set|split)",
    "uses_train_val_test":  r"train(ing)?[\s\/\-,]+(?:and )?valid[\w]*[\s\/\-,]+(?:and )?test",
    # Forecast horizon language
    "horizon_daily":        r"daily (?:forecast|prediction|patient|arrival|visit)",
    "horizon_weekly":       r"weekly (?:forecast|prediction|patient|arrival|visit)",
    "horizon_monthly":      r"monthly (?:forecast|prediction|patient|arrival|visit)",
    "horizon_hourly":       r"hourly (?:forecast|prediction|patient|arrival|visit)",
    "horizon_multi":        r"multi[\- ]?step|multi[\- ]horizon|short[\- ]term[\w\s,]+long[\- ]term",
    # Discussion sections
    "sec_discussion":       r"\n\s*\d?\.?\s*Discussion\b",
    "sec_limitations":      r"\bLimitations?\b",
    "sec_future_work":      r"\b(?:Future (?:Work|Research|Directions)|further research)\b",
    "sec_conclusion":       r"\n\s*\d?\.?\s*Conclusions?\b",
    "sec_clinical_impl":    r"clinical implications?|practical implications?|managerial implications?",
    "sec_comparison_prior": r"compared with (?:prior|previous|existing) (?:work|studies|literature)|comparison with (?:prior|previous|existing)",
}


# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(chunks)
    except Exception:
        return ""


def detect_patterns(text: str) -> dict:
    flags: dict = {}
    patterns = {**METRICS, **MODELS, **PRESENTATION}
    for name, pat in patterns.items():
        flags[name] = 1 if re.search(pat, text, re.IGNORECASE) else 0
    # Also: page count + word count proxy
    return flags


def main() -> None:
    pdfs = sorted(CORPUS.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {CORPUS}")
    rows = []
    failed = []
    t0 = time.time()

    for i, p in enumerate(pdfs, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(pdfs)}] {p.name[:55]:55s} ({elapsed:.0f}s)")
        text = extract_text(p)
        if len(text) < 500:
            failed.append(p.name)
            continue
        row = {"paper": p.name, "n_chars": len(text)}
        row.update(detect_patterns(text))
        rows.append(row)

    print(f"\nExtracted {len(rows)} / {len(pdfs)} papers "
          f"({len(failed)} failed) in {time.time()-t0:.0f}s")
    if failed:
        print("Failed (likely scanned-image PDFs):")
        for f in failed[:10]:
            print(f"  - {f}")

    df = pd.DataFrame(rows)
    out_per = OUT_DIR / "per_paper_features.csv"
    df.to_csv(out_per, index=False)
    print(f"\nWrote: {out_per}")

    # Aggregate frequencies
    flag_cols = [c for c in df.columns if c not in {"paper", "n_chars"}]
    freq = pd.DataFrame({
        "feature": flag_cols,
        "n_papers": [int(df[c].sum()) for c in flag_cols],
        "pct_papers": [round(100 * df[c].mean(), 1) for c in flag_cols],
    }).sort_values("pct_papers", ascending=False)

    out_freq = OUT_DIR / "pattern_frequencies.csv"
    freq.to_csv(out_freq, index=False)
    print(f"Wrote: {out_freq}")

    # Build readable report
    def grp(prefix: str) -> pd.DataFrame:
        return freq[freq["feature"].str.startswith(prefix)].copy()

    report_lines = []
    add = report_lines.append
    add(f"# Literature corpus pattern report")
    add(f"")
    add(f"- Papers analysed: **{len(df)}** (of {len(pdfs)} PDFs)")
    add(f"- Failed to extract text: {len(failed)} (likely scanned images)")
    add(f"")
    add(f"## Most common metrics")
    add(grp("metric_").to_markdown(index=False))
    add("")
    add(f"## Most common models compared")
    add(grp("model_").to_markdown(index=False))
    add("")
    add(f"## Most common presentation devices")
    add(grp("has_").to_markdown(index=False))
    add("")
    add(f"## Statistical tests used")
    add(grp("test_").to_markdown(index=False))
    add("")
    add(f"## Validation strategies")
    add(grp("uses_").to_markdown(index=False))
    add("")
    add(f"## Forecast-horizon framing")
    add(grp("horizon_").to_markdown(index=False))
    add("")
    add(f"## Discussion-section structure")
    add(grp("sec_").to_markdown(index=False))

    report = "\n".join(report_lines)
    out_report = OUT_DIR / "pattern_report.md"
    out_report.write_text(report, encoding="utf-8")
    print(f"Wrote: {out_report}")
    print("\n" + "=" * 70)
    print("TOP 20 MOST RECURRENT PATTERNS (across full corpus)")
    print("=" * 70)
    print(freq.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
