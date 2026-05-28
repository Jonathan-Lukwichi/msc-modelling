"""Extract abstract + methodology + results from PDFs more carefully.

For each paper, we want:
  - Abstract (first 1200 chars of page 1)
  - Methodology section (real one, not the abstract heading)
  - Results section (real one)

Strategy: search for section headers in mid-document, not front-matter.
"""
from __future__ import annotations

import re
from pathlib import Path
from PyPDF2 import PdfReader

FOLDER = Path(r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\JL_DOCU\machine learning and forecasting data")

PAPERS = [
    "Accurate Forecasting of Emergency Department Arrivals.pdf",
    "An explainable machine learning approach for hospital emergency.pdf",
    "Application of a Machine Learning Algorithm.pdf",
    "CAAI Trans on Intel Tech - 2023 - Susnjak - Forecasting patient demand at urgent care clinics using explainable machine.pdf",
    "Combining machine learning and optimization for the operational.pdf",
    "Comparison of linear, penalized linear and machine learning models.pdf",
    "Development and Validation of Machine Learning.pdf",
    "Development and validation of a machine learning model.pdf",
    "Development, evaluation and validation of machine learning models.pdf",
    "Error and Timeliness Analysis for Using Machine Learning.pdf",
]

METHODS_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:\d+\.?\s*)?"
    r"(materials\s+and\s+methods|methodology|methods|data\s+and\s+methods)\b"
    r"\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
RESULTS_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:\d+\.?\s*)?"
    r"(results(?:\s+and\s+discussion)?|findings|experimental\s+results|model\s+performance)\b"
    r"\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
DISCUSSION_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s*)?(discussion|conclusions?)\b\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)

KEYWORD_PHRASES = [
    r"train(?:ing)?(?:\s+and\s+test|\s+set)",
    r"cross[- ]?valid",
    r"hyper[- ]?parameter",
    r"grid\s+search",
    r"random\s+search",
    r"hyperopt",
    r"\bMAPE\b",
    r"\bRMSE\b",
    r"\bMAE\b",
    r"\bAIC\b",
    r"\bAUC\b|\bROC\b",
    r"sensitivity",
    r"specificity",
    r"holdout|hold-out|hold\s+out",
    r"backtest|rolling[- ]?origin|walk[- ]?forward",
    r"\bSARIMA\w*",
    r"\bARIMA\w*",
    r"XGBoost|XGB\b",
    r"random\s+forest",
    r"LSTM",
    r"GRU",
    r"linear\s+regression",
    r"Lasso",
    r"Ridge",
    r"Prophet",
    r"feature\s+select",
]


def extract_all_text(reader: PdfReader, max_pages: int = 60) -> list[tuple[int, str]]:
    """Return list of (page_index, text) tuples, capped at max_pages."""
    out = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        try:
            txt = reader.pages[i].extract_text() or ""
            out.append((i, txt))
        except Exception:
            out.append((i, ""))
    return out


def find_real_section(pages: list[tuple[int, str]], pattern: re.Pattern,
                       skip_first_n: int = 1) -> tuple[int, str]:
    """Find first match of pattern AFTER the first n pages (to skip the abstract)."""
    for i, txt in pages:
        if i < skip_first_n:
            continue
        m = pattern.search(txt)
        if m:
            # Return excerpt starting from the match
            start = m.start()
            return i, txt[start:start + 3000]
    return -1, ""


def keyword_hits(full_text: str) -> dict:
    """Count occurrences of methodology-relevant keywords across the paper."""
    hits = {}
    for pat in KEYWORD_PHRASES:
        n = len(re.findall(pat, full_text, re.IGNORECASE))
        if n > 0:
            hits[pat] = n
    return hits


def summarize_paper(path: Path) -> dict:
    try:
        r = PdfReader(str(path))
    except Exception as e:
        return {"title": path.name, "error": str(e)}

    pages = extract_all_text(r, max_pages=60)
    full_text = "\n".join(t for _, t in pages)

    # Page 1 only for abstract
    page1 = pages[0][1] if pages else ""
    # Strip front-matter chrome to keep abstract crisp
    abstract = page1[:1400]

    methods_page, methods_excerpt = find_real_section(pages, METHODS_RE, skip_first_n=1)
    results_page, results_excerpt = find_real_section(pages, RESULTS_RE, skip_first_n=2)
    discussion_page, discussion_excerpt = find_real_section(
        pages, DISCUSSION_RE, skip_first_n=2,
    )

    hits = keyword_hits(full_text)

    return {
        "title": path.name,
        "n_pages": len(r.pages),
        "abstract": abstract,
        "methods_page": methods_page + 1 if methods_page >= 0 else None,
        "methods_excerpt": methods_excerpt[:2000],
        "results_page": results_page + 1 if results_page >= 0 else None,
        "results_excerpt": results_excerpt[:2000],
        "discussion_page": discussion_page + 1 if discussion_page >= 0 else None,
        "discussion_excerpt": discussion_excerpt[:1200],
        "keyword_hits": hits,
    }


def main():
    for i, name in enumerate(PAPERS, 1):
        path = FOLDER / name
        if not path.exists():
            print(f"\n=== PAPER {i}: {name}  [NOT FOUND] ===")
            continue
        s = summarize_paper(path)
        print(f"\n{'#' * 78}")
        print(f"# PAPER {i}: {name}")
        print(f"# Pages: {s.get('n_pages', '?')}")
        print(f"{'#' * 78}")

        def out(label, txt):
            print(f"\n--- {label} ---")
            try:
                print(txt.encode("utf-8", errors="replace").decode("utf-8"))
            except Exception:
                print("(extraction failed)")

        out("ABSTRACT", s["abstract"])
        if s["methods_excerpt"]:
            out(f"METHODS (p{s['methods_page']})", s["methods_excerpt"])
        if s["results_excerpt"]:
            out(f"RESULTS (p{s['results_page']})", s["results_excerpt"])
        if s["discussion_excerpt"]:
            out(f"DISCUSSION/CONCLUSION (p{s['discussion_page']})", s["discussion_excerpt"])
        print("\n--- KEYWORD HITS (methodology vocabulary in paper) ---")
        for k, v in sorted(s["keyword_hits"].items(), key=lambda x: -x[1])[:15]:
            print(f"  {v:3d}x  {k}")


if __name__ == "__main__":
    main()
