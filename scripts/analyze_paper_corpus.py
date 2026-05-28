"""Extract a feature vector per paper across the whole corpus.

Strategy: scan every PDF, detect presence/absence of each methodological /
presentation feature (40+ features). The output is a CSV with one row per
paper. Then aggregate: what fraction of papers do X? That gives us the
"recurrent pattern" the user asked for.
"""
from __future__ import annotations

import re
import csv
from pathlib import Path
from PyPDF2 import PdfReader

FOLDER = Path(r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\JL_DOCU\machine learning and forecasting data")

# Features to detect in each paper.
# Each is a list of regex patterns; presence of ANY = the feature is True.
FEATURES = {
    # SECTION HEADERS
    "has_methods_section": [
        r"\n\s*(?:\d+\.?\s*)?(?:materials\s+and\s+methods|methodology|methods)\s*\n",
    ],
    "has_results_section": [
        r"\n\s*(?:\d+\.?\s*)?(?:results|findings)\s*\n",
    ],
    "has_discussion_section": [
        r"\n\s*(?:\d+\.?\s*)?discussion\s*\n",
    ],
    "has_limitations_section": [
        r"\n\s*(?:\d+\.?\s*)?limitations?\s*\n",
    ],
    "has_conclusion_section": [
        r"\n\s*(?:\d+\.?\s*)?conclusions?\s*\n",
    ],

    # METRICS
    "metric_MAPE": [r"\bMAPE\b", r"mean\s+absolute\s+percentage\s+error"],
    "metric_RMSE": [r"\bRMSE\b", r"root\s+mean\s+square[d]?\s+error"],
    "metric_MAE":  [r"\bMAE\b", r"mean\s+absolute\s+error"],
    "metric_MSE":  [r"\bMSE\b(?!E)", r"mean\s+square[d]?\s+error"],
    "metric_R2":   [r"R\^?2", r"R[\s\-]?squared", r"coefficient\s+of\s+determination"],
    "metric_AIC":  [r"\bAIC\b", r"akaike"],
    "metric_AUC":  [r"\bAUC\b", r"area\s+under\s+the?\s+curve"],
    "metric_ROC":  [r"\bROC\b", r"receiver\s+operating\s+characteristic"],
    "metric_accuracy": [r"\baccuracy\b"],
    "metric_precision_recall": [r"\bprecision\b.{0,30}\brecall\b|\brecall\b.{0,30}\bprecision\b"],
    "metric_F1": [r"\bF1[\s\-]?score\b", r"\bF[\s\-]?measure\b"],
    "metric_MASE": [r"\bMASE\b", r"mean\s+absolute\s+scaled\s+error"],
    "metric_Winkler": [r"winkler"],

    # MODEL FAMILIES
    "model_ARIMA": [r"\bARIMA\b"],
    "model_SARIMA": [r"\bSARIMA\b"],
    "model_SARIMAX": [r"\bSARIMAX\b"],
    "model_LSTM": [r"\bLSTM\b", r"long\s+short[\-\s]term\s+memory"],
    "model_GRU": [r"\bGRU\b"],
    "model_RNN": [r"\bRNN\b", r"recurrent\s+neural\s+network"],
    "model_CNN": [r"\bCNN\b", r"convolutional\s+neural\s+network"],
    "model_XGBoost": [r"\bXGBoost\b", r"\bXGB\b", r"extreme\s+gradient\s+boost"],
    "model_RandomForest": [r"\brandom\s+forest\b", r"\bRF\b"],
    "model_NN_MLP": [r"\bMLP\b", r"multi[\-\s]?layer\s+perceptron", r"feed[\-\s]?forward\s+neural"],
    "model_LinearReg": [r"\blinear\s+regression\b", r"\bOLS\b"],
    "model_Lasso": [r"\blasso\b"],
    "model_Ridge": [r"\bridge\b"],
    "model_ElasticNet": [r"\belastic\s+net\b"],
    "model_SVR_SVM": [r"\bSVR\b", r"\bSVM\b", r"support\s+vector"],
    "model_Prophet": [r"\bprophet\b"],
    "model_GBR_GBM": [r"\bGBR\b", r"\bGBM\b", r"\bGBRT\b", r"gradient\s+boost(?:ing|ed)\s+(?:tree|machine|regression)"],
    "model_NB_GLM": [r"negative\s+binomial", r"\bGLM\b", r"poisson\s+regression"],
    "model_ETS_HoltWinters": [r"\bETS\b", r"holt[\-\s]?winters"],
    "model_Hybrid": [r"hybrid\s+model", r"hybrid\s+approach"],
    "model_Ensemble": [r"\bensemble\b"],

    # PROTOCOL
    "split_chronological": [r"chronologic", r"in\s+chronological\s+order"],
    "split_random": [r"randomly\s+split", r"random\s+split"],
    "split_80_20_70_30": [r"80[\s/-]?20", r"70[\s/-]?30", r"60[\s/-]?40"],
    "validation_kfold": [r"\b[k\d]+\s*[\-\s]?fold", r"\bk-fold\b", r"5[\-\s]?fold|10[\-\s]?fold"],
    "validation_loocv": [r"leave[\-\s]?one[\-\s]?out", r"\bLOOCV\b"],
    "validation_holdout": [r"\bhold[\-\s]?out\b"],
    "validation_rolling_origin": [r"rolling[\-\s]?origin", r"walk[\-\s]?forward", r"time[\-\s]?series\s+cross"],
    "validation_train_test_only": [r"train(?:ing)?\s+(?:and|/|\\)\s+test", r"training\s+set.{0,80}testing\s+set"],

    # HPO
    "hpo_grid_search": [r"grid\s+search", r"GridSearchCV"],
    "hpo_random_search": [r"random\s+search", r"randomized\s+search"],
    "hpo_bayesian": [r"bayesian\s+optimi[sz]", r"hyperopt", r"BOHB"],
    "hpo_optuna": [r"\bOptuna\b", r"\bTPE\b", r"tree[\-\s]?structured\s+parzen"],
    "hpo_genetic": [r"genetic\s+algorithm"],
    "hpo_manual": [r"manual(?:ly)?\s+tun", r"trial\s+and\s+error"],

    # FEATURE ENGINEERING & SELECTION
    "feature_selection_lasso": [r"feature\s+selection.{0,40}lasso|lasso.{0,40}feature\s+selection"],
    "feature_selection_importance": [r"feature\s+importance"],
    "feature_selection_SHAP": [r"\bSHAP\b", r"shapley"],
    "feature_selection_LIME": [r"\bLIME\b"],
    "feature_selection_RFE": [r"recursive\s+feature\s+elimination", r"\bRFE\b"],
    "exogenous_weather": [r"weather|temperature|humidity|rainfall|precipitation"],
    "exogenous_calendar": [r"\bholiday\b|day\s+of\s+(?:the\s+)?week|calendar"],
    "exogenous_internet_search": [r"google\s+trends|search\s+index|search\s+queries"],
    "exogenous_social_media": [r"twitter|tweet|social\s+media"],

    # RESULTS PRESENTATION
    "results_table_with_models": [r"table\s+\d+.{0,80}(?:model|method|algorithm)"],
    "results_per_horizon": [r"\d+[\s\-]?day(?:s)?\s+(?:ahead|forecast|horizon)"],
    "results_per_period_quarter_month": [r"per[\s\-]?quarter|per[\s\-]?month|quarterly|monthly\s+performance"],
    "results_per_segment": [r"per[\s\-]?(?:specialty|clinic|department|group)|sub[\s\-]?group\s+analysis"],
    "results_residual_analysis": [r"residual.{0,30}(?:analysis|diagnosis|plot|correlogram)|correlogram|ljung[\-\s]?box"],
    "results_lineplot_actual_vs_pred": [r"line\s+plot|actual\s+(?:vs|versus)\s+predict|predicted\s+vs\s+actual"],
    "results_scatter_plot": [r"scatter\s+plot|scatterplot"],
    "results_confidence_interval": [r"confidence\s+interval|prediction\s+interval|\b95%\s+CI\b"],
    "results_significance_test": [r"p[\s\-]?value|statistic(?:ally)?\s+significan|p\s*<\s*0\.0\d"],

    # UNCERTAINTY
    "uq_conformal": [r"conformal\s+prediction|adaptive\s+conformal"],
    "uq_bootstrap": [r"bootstrap"],
    "uq_quantile": [r"quantile\s+regression|quantile\s+loss"],
    "uq_bayesian": [r"bayesian\s+inference|posterior\s+distribution|MCMC"],

    # DRIFT / OOD
    "drift_acknowledged": [r"\bdrift\b|distribution\s+shift|covariate\s+shift|concept\s+drift"],
    "drift_pandemic": [r"COVID|pandemic"],
    "drift_sliding_window": [r"sliding\s+window|rolling\s+window"],

    # DISCUSSION PATTERNS
    "compares_to_literature": [r"prior\s+(?:studies|work|literature)|compared\s+to\s+(?:other|previous)\s+studies|in\s+line\s+with"],
    "states_limitations_explicitly": [r"limitations?\s+of\s+(?:this|our|the)\s+study|study\s+(?:has\s+several|has\s+some)\s+limitations"],
    "explains_feature_importance": [r"most\s+important\s+features?|feature\s+importance\s+(?:analysis|results?)"],
    "operational_recommendations": [r"operational\s+(?:recommendation|implication)|recommendation\s+for\s+(?:hospital|clinic|practitioners?)"],

    # SOFTWARE
    "uses_python": [r"\bpython\b"],
    "uses_R": [r"\bR\s+(?:programming|statistical|version|software|package)"],
    "uses_scikit_learn": [r"scikit[\-\s]?learn|\bsklearn\b"],
    "uses_tensorflow_keras": [r"tensorflow|\bkeras\b"],
    "uses_pytorch": [r"pytorch"],
    "uses_pmdarima": [r"pmdarima|auto[_\s]?arima"],
}


def extract_full_text(reader: PdfReader, max_pages: int = 80) -> str:
    out = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        try:
            txt = reader.pages[i].extract_text() or ""
            out.append(txt)
        except Exception:
            continue
    return "\n".join(out)


def detect_features(text: str) -> dict[str, int]:
    """Return dict of {feature: 1 if present else 0} plus the total page count."""
    out = {}
    for feat, patterns in FEATURES.items():
        found = 0
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                found = 1
                break
        out[feat] = found
    return out


def main():
    pdfs = sorted(FOLDER.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs.\n", flush=True)

    rows = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            r = PdfReader(str(pdf))
            text = extract_full_text(r)
        except Exception as e:
            print(f"  [{i}/{len(pdfs)}] FAILED: {pdf.name}: {e}", flush=True)
            continue
        feats = detect_features(text)
        feats["__title"] = pdf.name
        feats["__n_pages"] = len(r.pages)
        feats["__n_chars"] = len(text)
        rows.append(feats)
        present = sum(v for k, v in feats.items() if not k.startswith("__"))
        print(f"  [{i}/{len(pdfs)}] {pdf.name[:60]:60s} ({len(r.pages)} pp, {present} features)", flush=True)

    # Write CSV
    if rows:
        out_csv = Path("artefacts/paper_corpus_features.csv")
        out_csv.parent.mkdir(exist_ok=True)
        all_keys = ["__title", "__n_pages", "__n_chars"] + sorted(FEATURES.keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\nWrote {out_csv}")

    # Pattern tally
    print(f"\n{'=' * 70}")
    print(f"PATTERN PREVALENCE (across {len(rows)} papers)")
    print(f"{'=' * 70}\n")
    n = len(rows)
    counts = {f: sum(r[f] for r in rows) for f in FEATURES.keys()}
    groups = {
        "SECTION HEADERS":   [k for k in counts if k.startswith("has_")],
        "METRICS":           [k for k in counts if k.startswith("metric_")],
        "MODEL FAMILIES":    [k for k in counts if k.startswith("model_")],
        "TRAIN/TEST":        [k for k in counts if k.startswith("split_") or k.startswith("validation_")],
        "HPO":               [k for k in counts if k.startswith("hpo_")],
        "FEATURE ENG":       [k for k in counts if k.startswith("feature_") or k.startswith("exogenous_")],
        "RESULTS PRESENT.":  [k for k in counts if k.startswith("results_")],
        "UQ":                [k for k in counts if k.startswith("uq_")],
        "DRIFT/OOD":         [k for k in counts if k.startswith("drift_")],
        "DISCUSSION":        [k for k in counts if k.startswith("compares_") or k.startswith("states_") or k.startswith("explains_") or k.startswith("operational_")],
        "SOFTWARE":          [k for k in counts if k.startswith("uses_")],
    }
    for gname, keys in groups.items():
        if not keys:
            continue
        print(f"--- {gname} ---")
        for k in sorted(keys, key=lambda x: -counts[x]):
            pct = 100.0 * counts[k] / n if n else 0
            bar = "*" * int(pct / 4)
            print(f"  {pct:5.1f}%  ({counts[k]:2d}/{n})  {bar:25s}  {k}")
        print()


if __name__ == "__main__":
    main()
