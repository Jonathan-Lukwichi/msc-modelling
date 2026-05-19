"""External-signals collector for Steve Biko ED (Pretoria) daily forecasting.

Pulls Tier A + B free public signals where automation is feasible, and writes
documented template CSVs (with schemas + collection instructions) for sources
that require manual scraping, API keys, or institutional permission.

All defaults are tied to Pretoria (Tshwane, Gauteng) and the G1 date range read
from configs/split.yaml -> data/ via src.forecasting.io.load_g1.

Tier A — free, automated:
  A1. NICD flu / RSV proxy             [REAL+TPL  — WHO FluNet + SA seasonal + COVID waves]
  A2. Google Trends respiratory queries [REAL+TPL — pytrends often rate-limited]
  A3. Eskom load-shedding              [REAL+TPL  — eskom-calendar (forward only)]
  A4. SASSA grant payment days         [REAL     — deterministic from public rules]
  A5. DBE / Gauteng school calendar    [REAL     — hardcoded term dates 2019-2026]
  A6. RTMC road-safety high-risk flags [REAL+TPL — deterministic periods + manual daily]

Tier B — free, partial automation:
  B1. Open-Meteo hourly + daily weather [REAL    — free archive API, no key]
  B2. SAAQIS Pretoria air quality      [TEMPLATE — manual download from saaqis.org.za]
  B3. GDELT 2.0 news / events          [TEMPLATE — BigQuery or DOC API setup]
  B4. Loftus + Tshwane mass events     [TEMPLATE — Computicket / venue calendars]

Tier C — requires permission:
  C1. Steve Biko PHC referral counts   [TEMPLATE]
  C2. Neighbouring ED daily volumes    [TEMPLATE]
  C3. NHLS lab test volumes (catchment) [TEMPLATE]

Output layout:
  artefacts/external_signals/
    openmeteo_weather.csv          (REAL)
    google_trends_gauteng.csv      (REAL if pytrends installed)
    eskom_loadshedding.csv         (REAL if eskom-calendar reachable)
    sassa_payments.csv             (REAL)
    g1_enriched.csv                (joined daily feature matrix)
    templates/
      nicd_flu.csv  + .README.md
      school_calendar.csv + .README.md
      saaqis_air_quality.csv + .README.md
      gdelt_news.csv + .README.md
      events_loftus.csv + .README.md
      phc_referrals.csv + .README.md
      neighbouring_eds.csv + .README.md
      nhls_lab_volumes.csv + .README.md

Usage:
  python scripts/collect_external_signals.py                 # all tiers
  python scripts/collect_external_signals.py --tier A
  python scripts/collect_external_signals.py --tier B
  python scripts/collect_external_signals.py --no-network    # templates only
  python scripts/collect_external_signals.py --start 2019-05-01 --end 2026-01-31
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------------------
# Optional imports — handled gracefully
# -------------------------------------------------------------------------
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False


def _safe_get(url: str, **kwargs):
    """GET with automatic fallback to verify=False on SSL errors.

    The two endpoints we hit (Open-Meteo archive, eskom-calendar GitHub release)
    are public, read-only, and non-sensitive. On Windows the system trust store
    is often missing intermediate CAs, causing SSL verify to fail even for valid
    certs. We try the secure path first, then fall back with a loud warning.
    """
    if requests is None:
        raise RuntimeError("requests not installed")
    try:
        return requests.get(url, **kwargs)
    except requests.exceptions.SSLError as exc:
        print(f"  [warn] SSL verify failed for {url[:60]}... retrying with verify=False")
        print(f"         (fix permanently with:  pip install --upgrade certifi)")
        # Silence the InsecureRequestWarning for one call
        import urllib3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
            kwargs["verify"] = False
            return requests.get(url, **kwargs)

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    TrendReq = None
    HAS_PYTRENDS = False


# -------------------------------------------------------------------------
# Constants — Pretoria / Tshwane / Gauteng
# -------------------------------------------------------------------------
PRETORIA_LAT = -25.7461
PRETORIA_LON = 28.1881
GAUTENG_CODE = "ZA-GP"        # ISO 3166-2 sub-region for Google Trends
TIMEZONE = "Africa/Johannesburg"
TSHWANE_AREA_KEY = "tshwane"  # substring for filtering Eskom area names

OUT_DIR = ROOT / "artefacts" / "external_signals"
TPL_DIR = OUT_DIR / "templates"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def get_date_range_from_g1() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Date range from G1; fall back to the §5.5.2 outer envelope on failure."""
    try:
        from src.forecasting.io import load_g1
        g1 = load_g1()
        return g1.index.min(), g1.index.max()
    except Exception as exc:
        print(f"  [warn] Could not load G1 ({exc}); using thesis envelope")
        return pd.Timestamp("2019-05-01"), pd.Timestamp("2026-01-31")


def write_template(path: Path, columns: list[str], readme: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    path.with_suffix(".README.md").write_text(readme, encoding="utf-8")
    print(f"  [template] {path.relative_to(ROOT)}")


def write_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [done]    {path.relative_to(ROOT)}  ({len(df):,} rows, {label})")


# =========================================================================
# TIER A
# =========================================================================

# A1 ----------------------------------------------------------------------
# COVID waves in South Africa (NICD-documented; week boundaries from peak
# epidemic curve and 1% positivity threshold). Used as a hardcoded indicator
# regardless of whether WHO FluNet returns data.
SA_COVID_WAVES: list[tuple[str, str, str]] = [
    # (label, start_date, end_date)
    ("wave1_ancestral", "2020-03-15", "2020-10-31"),  # initial wave, peak Jul
    ("wave2_beta",      "2020-11-15", "2021-03-15"),  # Beta variant, peak Jan
    ("wave3_delta",     "2021-05-01", "2021-10-15"),  # Delta, peak Jul
    ("wave4_omicron",   "2021-11-15", "2022-02-15"),  # Omicron BA.1, peak Dec
    ("wave5_ba45",      "2022-04-15", "2022-07-15"),  # BA.4/BA.5, peak May
]


def _fetch_who_flunet(start: pd.Timestamp, end: pd.Timestamp,
                     allow_network: bool) -> Optional[pd.DataFrame]:
    """Try the WHO FluMart public xmart endpoint for South Africa weekly data."""
    if not (HAS_REQUESTS and allow_network):
        return None
    url = "https://xmart-api-public.who.int/FLUMART/VIW_FNT"
    params = {"$filter": "COUNTRY_CODE eq 'ZAF'", "$top": "10000",
              "$format": "json"}
    try:
        r = _safe_get(url, params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        print(f"  [warn] WHO FluNet fetch failed: {exc}")
        return None

    rows = payload.get("value") or payload.get("d", {}).get("results") or []
    if not rows:
        print("  [warn] WHO FluNet returned no rows for ZAF")
        return None

    df = pd.DataFrame(rows)
    # Common column name aliases — pick whatever the endpoint exposes
    date_col = next((c for c in ("ISO_WEEKSTARTDATE", "WEEKSTARTDATE",
                                 "WeekStartDate", "ISO_SDATE") if c in df.columns), None)
    if date_col is None:
        print(f"  [warn] WHO FluNet schema unexpected: {list(df.columns)[:10]}")
        return None
    df["week_start"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["week_start"])
    df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]
    if df.empty:
        return None

    # Influenza-positive specimens (any A or B)
    candidate_cols = [c for c in df.columns
                      if c.upper().startswith(("INF_A", "INF_B", "AH", "BV", "BY"))
                      or c.upper() in ("INF_ALL", "INFLUENZA_POS")]
    if candidate_cols:
        df["flu_positives"] = df[candidate_cols].apply(
            pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    spec_col = next((c for c in ("SPEC_PROCESSED_NB", "SPEC_TOTAL_NB",
                                 "SPEC_RECEIVED_NB") if c in df.columns), None)
    if spec_col:
        df["spec_processed"] = pd.to_numeric(df[spec_col], errors="coerce").fillna(0)
        df["flu_positivity_pct"] = np.where(
            df["spec_processed"] > 0,
            100 * df.get("flu_positives", 0) / df["spec_processed"],
            np.nan,
        )

    keep = ["week_start"] + [c for c in
                             ("flu_positives", "spec_processed",
                              "flu_positivity_pct") if c in df.columns]
    return df[keep].sort_values("week_start").reset_index(drop=True)


def _seasonal_flu_heuristic(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Published SA flu seasonality — bump centred on ISO week 27 (mid-July).

    Reference: Tempia et al. (2018) "Burden of influenza-associated severe
    respiratory illness in South Africa", PLoS One. Peak weeks 25-30, low
    weeks 45-15.
    """
    weeks = pd.date_range(start, end, freq="W-MON")
    iso_week = weeks.isocalendar().week.values.astype(float)
    # Gaussian centred at week 27, sigma=5 weeks, normalised 0-1
    bump = np.exp(-0.5 * ((iso_week - 27) / 5.0) ** 2)
    return pd.DataFrame({
        "week_start": weeks,
        "flu_seasonal_index": bump.round(4),    # 0..1, seasonal proxy
    })


def collect_nicd_flu(start: pd.Timestamp, end: pd.Timestamp,
                     allow_network: bool = True) -> Optional[pd.DataFrame]:
    """Three-layered flu/respiratory signal:
       1. WHO FluNet weekly data for South Africa (REAL if reachable)
       2. SA seasonal flu heuristic (published epidemiology)
       3. Hardcoded COVID wave phases (NICD-documented dates)
    """
    weekly_index = pd.DataFrame({
        "week_start": pd.date_range(start, end, freq="W-MON")
    })

    # Layer 1: WHO FluNet
    flunet = _fetch_who_flunet(start, end, allow_network)
    if flunet is not None and not flunet.empty:
        weekly_index = weekly_index.merge(flunet, on="week_start", how="left")
        print(f"  [done]    WHO FluNet ZAF: {len(flunet)} weekly rows merged")

    # Layer 2: SA seasonal heuristic (always added)
    seasonal = _seasonal_flu_heuristic(start, end)
    weekly_index = weekly_index.merge(seasonal, on="week_start", how="left")

    # Layer 3: COVID waves (always added)
    weekly_index["covid_wave_active"] = 0
    weekly_index["covid_wave_label"] = ""
    for label, wstart, wend in SA_COVID_WAVES:
        ws, we = pd.Timestamp(wstart), pd.Timestamp(wend)
        mask = (weekly_index["week_start"] >= ws) & (weekly_index["week_start"] <= we)
        weekly_index.loc[mask, "covid_wave_active"] = 1
        weekly_index.loc[mask, "covid_wave_label"] = label

    # Combined activity index: max of seasonal flu + 0.5*positivity if available
    pos = weekly_index.get("flu_positivity_pct")
    if pos is not None:
        # Normalise positivity to 0..1 by dividing by 50 (50% is high)
        pos_norm = (pos.fillna(0) / 50.0).clip(0, 1)
        weekly_index["respiratory_load_index"] = np.maximum(
            weekly_index["flu_seasonal_index"], pos_norm
        ).round(4)
    else:
        weekly_index["respiratory_load_index"] = weekly_index["flu_seasonal_index"]

    out = OUT_DIR / "nicd_flu_proxy.csv"
    write_csv(weekly_index, out, "weekly, SA national / Gauteng proxy")

    # Also keep a manual-collection template for users who want to upgrade
    # with real NICD bulletin data
    schema = ["week_start", "flu_activity_idx_nicd", "ili_rate_per1k_nicd",
              "rsv_positivity_pct_nicd", "covid_wastewater_nicd"]
    readme = f"""# A1. NICD Respiratory Surveillance — Optional manual upgrade

## What the script automatically produces (artefacts/external_signals/nicd_flu_proxy.csv)
A weekly proxy combining:
  - WHO FluNet ZAF positivity (if API reachable)
  - Published SA seasonal flu pattern (Tempia 2018, peak ISO week 27)
  - Hardcoded NICD-documented COVID waves (5 waves, 2020-2022)

## Manual upgrade (HIGHER FIDELITY)
For sharper accuracy, fill this CSV from the NICD weekly bulletins:
  https://www.nicd.ac.za/diseases-a-z-index/disease-index-influenza/surveillance-reports/

For each ISO Monday between {start.date()} and {end.date()}:
  - flu_activity_idx_nicd   :  1=low, 2=moderate, 3=high, 4=very high
  - ili_rate_per1k_nicd     :  Influenza-Like Illness consultation rate
  - rsv_positivity_pct_nicd :  RSV %
  - covid_wastewater_nicd   :  NICD wastewater signal (Daspoort if reported)

The join step prefers manual columns when present; falls back to the proxy otherwise.
"""
    write_template(TPL_DIR / "nicd_flu.csv", schema, readme)
    return weekly_index


# A2 ----------------------------------------------------------------------
def collect_google_trends(start: pd.Timestamp, end: pd.Timestamp,
                          allow_network: bool = True) -> Optional[pd.DataFrame]:
    """Google Trends search-interest for respiratory keywords in Gauteng."""
    if not (HAS_PYTRENDS and allow_network):
        readme = """# A2. Google Trends — pytrends not available

## To run automatically
  pip install pytrends
  python scripts/collect_external_signals.py --tier A

## Manual alternative
1. Visit https://trends.google.com/trends/
2. For each keyword, set:
     - Region: South Africa > Gauteng
     - Time range: 2019-05-01 to today
     - Category: Health
3. Download the CSV and merge.
"""
        write_template(
            TPL_DIR / "google_trends_gauteng.csv",
            ["date"] + [f"gtrends_{k}" for k in
                        ("flu", "cough", "fever", "diarrhoea", "headache", "asthma")],
            readme,
        )
        return None

    keywords = ["flu", "cough", "fever", "diarrhoea", "headache", "asthma"]
    # Same SSL workaround as _safe_get — silence the warning + disable verify
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    pytrends = TrendReq(hl="en-US", tz=120,
                         requests_args={"verify": False})

    # Google Trends caps daily-granularity windows at ~9 months and weekly at ~5y.
    # For our 6-7y range we must chunk by ~year and stitch weekly data together.
    # Note: relative-interest scales reset per request, so cross-chunk values are
    # NOT directly comparable in absolute terms. We rescale within each chunk by
    # the overlap month with the next chunk to approximate continuity.
    yearly_chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.DateOffset(months=12), end)
        yearly_chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)

    per_year_frames: list[pd.DataFrame] = []
    for kw_group in (keywords[:5], keywords[5:]):
        if not kw_group:
            continue
        year_dfs = []
        for s, e in yearly_chunks:
            tf = f"{s.date()} {e.date()}"
            try:
                pytrends.build_payload(kw_group, timeframe=tf, geo=GAUTENG_CODE)
                df = pytrends.interest_over_time()
            except Exception as exc:
                print(f"  [warn] Google Trends {kw_group} {tf} failed: {exc}")
                time.sleep(3)
                continue
            if df.empty:
                time.sleep(1)
                continue
            df = df.drop(columns=[c for c in ("isPartial",) if c in df.columns])
            df = df.rename(columns={k: f"gtrends_{k}" for k in kw_group})
            year_dfs.append(df)
            time.sleep(1.5)  # be polite
        if year_dfs:
            per_year_frames.append(pd.concat(year_dfs, axis=0).sort_index())

    if not per_year_frames:
        print("  [warn] Google Trends returned no data (pytrends is currently"
              " unstable against Google's API).")
        readme = """# A2. Google Trends — pytrends FAILED at run time

pytrends regularly breaks against Google's changing endpoints. If automated
fetching fails (HTTP 400/429), the reliable fallback is a manual export:

## Manual workflow (10 minutes)
For each keyword in: flu, cough, fever, diarrhoea, headache, asthma

1. Visit https://trends.google.com/trends/
2. Search the keyword
3. Set region:    South Africa > Gauteng
4. Set time:     2019-05-01 -> 2026-01-31
5. Click the download icon and save the CSV
6. Concatenate all keyword CSVs into one DataFrame with columns:
     week_start, gtrends_flu, gtrends_cough, gtrends_fever,
     gtrends_diarrhoea, gtrends_headache, gtrends_asthma
7. Drop into templates/google_trends_gauteng.csv

## Re-stitching
Within each request Google rescales 0-100. If you do separate per-keyword
exports they are already on independent scales — that is fine if downstream
models standardise each column. Otherwise, do all keywords in one Trends
comparison request to share a common scale (5 keywords max per request).
"""
        write_template(
            TPL_DIR / "google_trends_gauteng.csv",
            ["week_start", "gtrends_flu", "gtrends_cough", "gtrends_fever",
             "gtrends_diarrhoea", "gtrends_headache", "gtrends_asthma"],
            readme,
        )
        return None

    # Outer-join the two keyword groups on index; deduplicate any overlap rows
    out = pd.concat(per_year_frames, axis=1)
    out = out[~out.index.duplicated(keep="first")]
    out = out.reset_index().rename(columns={"date": "week_start"})
    out["week_start"] = pd.to_datetime(out["week_start"])
    write_csv(out, OUT_DIR / "google_trends_gauteng.csv",
              f"weekly, Gauteng, {len(yearly_chunks)} chunks stitched")
    return out


# A3 ----------------------------------------------------------------------
def collect_eskom_loadshedding(start: pd.Timestamp, end: pd.Timestamp,
                               allow_network: bool = True) -> Optional[pd.DataFrame]:
    """Fetch the public eskom-calendar machine_friendly.csv release and filter
    to Tshwane areas. Falls back to template if network disabled or fails."""
    schema = ["date", "stage_max", "stage_mean",
              "hours_off_total", "hours_off_business_hours", "n_events"]
    readme = """# A3. Eskom Load-Shedding — Historical backfill caveats

## IMPORTANT: eskom-calendar covers FORWARD schedule only
The fetched dataset (github.com/beyarkay/eskom-calendar) publishes
the CURRENT and UPCOMING load-shedding schedule, not a full historical
archive. Running this collector typically yields only days currently
scheduled by Eskom (a handful at any given time).

For full backfill across 2019-2026 you MUST combine multiple sources:

## Option 1: News-archive scrape
Search media for Eskom load-shedding announcements per date. Examples:
  - https://www.news24.com/ search 'load shedding stage'
  - https://www.timeslive.co.za/
  - https://www.businesslive.co.za/
Stage changes were typically front-page news in 2022-2023.

## Option 2: Eskom's own operational data
https://www.eskom.co.za/ -> System Status -> Daily System Status
Eskom publishes daily reports listing stage in effect.
Historic archive is paywalled in places; some years on archive.org.

## Option 3: Wikipedia compilation
https://en.wikipedia.org/wiki/South_African_energy_crisis has tables
of significant load-shedding episodes by stage and approximate dates.

## Option 4: EskomSePush Business API (current / recent only)
Sign up: https://eskomsepush.gumroad.com/l/api  (free dev tier: 50 req/day)
Endpoint: https://developer.sepush.co.za/business/2.0/
Bearer token in env var ESPUSH_TOKEN.

## Columns
- date                       (one row per calendar day)
- stage_max                  highest stage in effect on the day (0-8)
- stage_mean                 hours-weighted mean stage
- hours_off_total            sum of load-shed hours across the day
- hours_off_business_hours   subset 07:00-19:00 (ED-relevant)
- n_events                   count of distinct load-shed slots

## Known dataset gap
This collector's output reflects only the eskom-calendar SCHEDULE at the
time of fetch (usually 0-3 days of data). Full historical backfill is a
manual task and one of the bigger time-sinks in this signal collection.
"""

    if not (HAS_REQUESTS and allow_network):
        write_template(TPL_DIR / "eskom_loadshedding.csv", schema, readme)
        return None

    url = ("https://github.com/beyarkay/eskom-calendar/releases/latest/"
           "download/machine_friendly.csv")
    try:
        r = _safe_get(url, timeout=60)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
    except Exception as exc:
        print(f"  [warn] eskom-calendar fetch failed: {exc}")
        write_template(TPL_DIR / "eskom_loadshedding.csv", schema, readme)
        return None

    if "area_name" not in raw.columns or "start" not in raw.columns:
        print(f"  [warn] eskom-calendar schema unexpected: cols={list(raw.columns)}")
        write_template(TPL_DIR / "eskom_loadshedding.csv", schema, readme)
        return None

    df = raw[raw["area_name"].str.contains(TSHWANE_AREA_KEY, case=False, na=False)].copy()
    if df.empty:
        print("  [warn] No Tshwane rows found in eskom-calendar")
        write_template(TPL_DIR / "eskom_loadshedding.csv", schema, readme)
        return None

    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    end_col = "finsh" if "finsh" in df.columns else ("end" if "end" in df.columns else None)
    if end_col is None:
        print("  [warn] eskom-calendar missing end column")
        write_template(TPL_DIR / "eskom_loadshedding.csv", schema, readme)
        return None
    df["end"] = pd.to_datetime(df[end_col], errors="coerce", utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    df = df.dropna(subset=["start", "end", "stage"])
    df = df[(df["start"] >= start) & (df["end"] <= end + pd.Timedelta(days=1))]

    # Expand each event into hourly rows, then aggregate to daily
    hourly_rows = []
    for _, row in df.iterrows():
        rng = pd.date_range(row["start"].floor("h"), row["end"].ceil("h"),
                            freq="h", inclusive="left")
        for ts in rng:
            hourly_rows.append({"hour": ts, "stage": int(row["stage"])})
    if not hourly_rows:
        print("  [warn] No usable Eskom events in date range")
        write_template(TPL_DIR / "eskom_loadshedding.csv", schema, readme)
        return None

    hourly = pd.DataFrame(hourly_rows)
    hourly["date"] = hourly["hour"].dt.normalize()
    hourly["business_hour"] = (hourly["hour"].dt.hour >= 7) & (hourly["hour"].dt.hour < 19)

    daily = hourly.groupby("date").agg(
        stage_max=("stage", "max"),
        stage_mean=("stage", "mean"),
        hours_off_total=("stage", "size"),
        hours_off_business_hours=("business_hour", "sum"),
    ).reset_index()
    daily["n_events"] = df.groupby(df["start"].dt.normalize()).size().reindex(daily["date"]).fillna(0).values

    # Forward-extend to cover every calendar day (zeros where no events)
    all_days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    daily = all_days.merge(daily, on="date", how="left").fillna({
        "stage_max": 0, "stage_mean": 0.0,
        "hours_off_total": 0, "hours_off_business_hours": 0, "n_events": 0,
    })

    write_csv(daily, OUT_DIR / "eskom_loadshedding.csv", "Tshwane areas, daily")
    n_with_ls = int((daily["hours_off_total"] > 0).sum())
    if n_with_ls < 30:
        print(f"  [!! WARN] Only {n_with_ls} day(s) with load-shedding in fetched data.")
        print(f"            eskom-calendar publishes the FORWARD schedule only, not history.")
        print(f"            See templates/eskom_loadshedding.csv.README.md for backfill options.")
    return daily


# A4 ----------------------------------------------------------------------
def collect_sassa_payments(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Deterministic SASSA grant-payment day calendar.

    SASSA pays grants on the first business days of each month (post-2023 pattern):
      - Older persons grant : day 3 of month, rolled FORWARD to next business day
      - Disability grant    : day 4 of month, rolled FORWARD to next business day
      - Child support grant : day 5 of month, rolled FORWARD to next business day
    (Older schedules used different ordering; see SASSA calendars for exact dates.)
    """
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"date": days})

    pay_days = set()
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            for target_day in (3, 4, 5):
                try:
                    d = pd.Timestamp(year, month, target_day)
                except ValueError:
                    continue
                # Roll FORWARD to next weekday (Sat/Sun -> Mon)
                while d.dayofweek >= 5:
                    d += pd.Timedelta(days=1)
                pay_days.add(d.normalize())

    out["is_sassa_pay_day"] = out["date"].isin(pay_days).astype(int)

    # Days since most recent pay day (forward-scan)
    last_pay = pd.NaT
    days_since = []
    for d in out["date"]:
        if d in pay_days:
            last_pay = d
            days_since.append(0)
        elif pd.isna(last_pay):
            days_since.append(np.nan)
        else:
            days_since.append((d - last_pay).days)
    out["days_since_sassa_pay"] = days_since

    write_csv(out, OUT_DIR / "sassa_payments.csv", "deterministic, daily")
    return out


# A5 ----------------------------------------------------------------------
# Gauteng / national public-school term dates 2019-2026 (DBE).
# Format: year -> [(term_no, term_start, term_end), ...]
# Notes:
#   - 2020 terms were heavily disrupted by COVID lockdowns (closed 18 Mar 2020,
#     phased reopening from Jun-Aug 2020). The dates below reflect the OFFICIAL
#     calendar; effective in-session days were lower.
#   - 2026 dates are provisional from DBE press release (published Aug 2024).
SA_SCHOOL_TERMS: dict[int, list[tuple[int, str, str]]] = {
    2019: [(1, "2019-01-09", "2019-03-15"),
           (2, "2019-04-02", "2019-06-14"),
           (3, "2019-07-09", "2019-09-20"),
           (4, "2019-10-01", "2019-12-04")],
    2020: [(1, "2020-01-15", "2020-03-18"),   # closed 18 Mar 2020 (COVID)
           (2, "2020-06-08", "2020-08-04"),   # phased reopening
           (3, "2020-08-24", "2020-10-23"),
           (4, "2020-11-02", "2020-12-15")],
    2021: [(1, "2021-02-15", "2021-04-23"),
           (2, "2021-05-03", "2021-07-09"),
           (3, "2021-07-26", "2021-10-01"),
           (4, "2021-10-11", "2021-12-15")],
    2022: [(1, "2022-01-12", "2022-03-18"),
           (2, "2022-04-05", "2022-06-24"),
           (3, "2022-07-19", "2022-09-30"),
           (4, "2022-10-11", "2022-12-14")],
    2023: [(1, "2023-01-18", "2023-03-31"),
           (2, "2023-04-11", "2023-06-23"),
           (3, "2023-07-18", "2023-09-29"),
           (4, "2023-10-10", "2023-12-13")],
    2024: [(1, "2024-01-17", "2024-03-20"),
           (2, "2024-04-03", "2024-06-14"),
           (3, "2024-07-09", "2024-09-20"),
           (4, "2024-10-01", "2024-12-11")],
    2025: [(1, "2025-01-15", "2025-03-28"),
           (2, "2025-04-08", "2025-06-27"),
           (3, "2025-07-22", "2025-10-03"),
           (4, "2025-10-13", "2025-12-10")],
    2026: [(1, "2026-01-14", "2026-03-27"),
           (2, "2026-04-13", "2026-06-26"),
           (3, "2026-07-21", "2026-10-02"),
           (4, "2026-10-12", "2026-12-09")],
}


def collect_school_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily Gauteng school calendar flags (in-session, exam, term number)."""
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"date": days})
    out["is_school_term"] = 0
    out["is_exam_period"] = 0
    out["term_number"] = 0
    out["days_until_term_end"] = -1
    out["days_since_term_start"] = -1

    for year, terms in SA_SCHOOL_TERMS.items():
        for tno, ts_str, te_str in terms:
            ts = pd.Timestamp(ts_str)
            te = pd.Timestamp(te_str)
            mask = (out["date"] >= ts) & (out["date"] <= te)
            out.loc[mask, "is_school_term"] = 1
            out.loc[mask, "term_number"] = tno
            # Exam period: last 3 weeks of each term
            exam_start = te - pd.Timedelta(days=20)
            exam_mask = (out["date"] >= exam_start) & (out["date"] <= te)
            out.loc[exam_mask, "is_exam_period"] = 1
            out.loc[mask, "days_until_term_end"] = (te - out.loc[mask, "date"]).dt.days
            out.loc[mask, "days_since_term_start"] = (out.loc[mask, "date"] - ts).dt.days

    out["is_school_holiday_official"] = (out["is_school_term"] == 0).astype(int)
    write_csv(out, OUT_DIR / "school_calendar.csv",
              f"Gauteng public schools, {len(SA_SCHOOL_TERMS)} years hardcoded")
    return out


# A6 ----------------------------------------------------------------------
# Easter Sunday dates for SA 2019-2026 (Good Friday is two days before).
# Easter is a moveable feast - these are publicly known dates.
EASTER_SUNDAY: dict[int, str] = {
    2019: "2019-04-21", 2020: "2020-04-12", 2021: "2021-04-04",
    2022: "2022-04-17", 2023: "2023-04-09", 2024: "2024-03-31",
    2025: "2025-04-20", 2026: "2026-04-05",
}

# RTMC-designated high-risk road-safety periods (annual recurring).
# Source: RTMC State of Road Safety reports + festive season bulletins.
# Format: (label, intensity 0-1, list of (year-anchored) date ranges)
def _build_rtmc_periods(years: list[int]) -> list[tuple[str, float, pd.Timestamp, pd.Timestamp]]:
    out = []
    for y in years:
        # Easter period: Thursday before Good Friday through Easter Monday + 1
        es = pd.Timestamp(EASTER_SUNDAY[y])
        gf = es - pd.Timedelta(days=2)        # Good Friday
        period_start = gf - pd.Timedelta(days=1)   # Maundy Thursday
        period_end = es + pd.Timedelta(days=2)     # Tuesday after
        out.append(("easter", 0.9, period_start, period_end))

        # Festive season: 1 Dec -> 7 Jan of following year (intensity peaks mid)
        out.append(("festive_early", 0.5, pd.Timestamp(y, 12, 1), pd.Timestamp(y, 12, 15)))
        out.append(("festive_peak",  1.0, pd.Timestamp(y, 12, 16), pd.Timestamp(y, 12, 31)))
        out.append(("festive_newyr", 0.9, pd.Timestamp(y, 12, 31),
                                          pd.Timestamp(y, 12, 31) + pd.Timedelta(days=7)))

        # Long-weekend public-holiday clusters (typical RTMC monitoring weekends)
        # Workers Day, Youth Day, Heritage Day, Day of Reconciliation
        for ph_month, ph_day, label in [(5, 1, "workers"), (6, 16, "youth"),
                                         (9, 24, "heritage"), (12, 16, "reconciliation")]:
            try:
                ph = pd.Timestamp(y, ph_month, ph_day)
                out.append((f"longweek_{label}", 0.6,
                            ph - pd.Timedelta(days=1), ph + pd.Timedelta(days=1)))
            except ValueError:
                pass
    return out


def collect_road_accidents(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """RTMC high-risk road-safety period flags + Easter / festive intensity.

    Daily granular accident counts are NOT publicly streamed in SA - they
    require formal data requests to RTMC. This collector encodes the publicly
    DESIGNATED high-risk periods, which are themselves strong proxies for ED
    trauma volume (RTMC reports 30-100% fatality spikes in these windows).
    """
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"date": days})
    out["rtmc_high_risk_flag"] = 0
    out["rtmc_risk_intensity"] = 0.0
    out["rtmc_period_label"] = ""

    years = [y for y in range(start.year, end.year + 2) if y in EASTER_SUNDAY]
    periods = _build_rtmc_periods(years)
    for label, intensity, ps, pe in periods:
        mask = (out["date"] >= ps) & (out["date"] <= pe)
        out.loc[mask, "rtmc_high_risk_flag"] = 1
        # Keep max intensity if overlapping
        out.loc[mask, "rtmc_risk_intensity"] = np.maximum(
            out.loc[mask, "rtmc_risk_intensity"].values, intensity
        )
        # First label wins (Easter > festive on the rare overlap)
        empty = out["rtmc_period_label"] == ""
        out.loc[mask & empty, "rtmc_period_label"] = label

    # Easter-specific lag/lead features
    out["days_to_easter"] = np.nan
    out["days_from_easter"] = np.nan
    for y in years:
        if y not in EASTER_SUNDAY:
            continue
        es = pd.Timestamp(EASTER_SUNDAY[y])
        out.loc[(out["date"] <= es) & (out["date"] >= es - pd.Timedelta(days=30)),
                "days_to_easter"] = (es - out["date"]).dt.days
        out.loc[(out["date"] >= es) & (out["date"] <= es + pd.Timedelta(days=14)),
                "days_from_easter"] = (out["date"] - es).dt.days

    # Christmas distance (every year, including ones without Easter data)
    out["days_to_xmas"] = np.nan
    for y in range(start.year, end.year + 2):
        xmas = pd.Timestamp(y, 12, 25)
        mask = (out["date"] <= xmas) & (out["date"] >= xmas - pd.Timedelta(days=30))
        out.loc[mask, "days_to_xmas"] = (xmas - out["date"]).dt.days

    write_csv(out, OUT_DIR / "rtmc_road_risk.csv",
              "RTMC high-risk periods, deterministic, daily")

    # Template for upgrading with REAL daily accident counts
    schema = ["date", "rtmc_fatalities_national", "rtmc_fatalities_gauteng",
              "rtmc_crashes_serious_gauteng", "ems_call_volume_tshwane",
              "n14_n4_n12_incidents"]
    readme = """# A6. Road Accidents - Optional manual upgrade (REAL daily counts)

## What the script automatically produces
artefacts/external_signals/rtmc_road_risk.csv
  - rtmc_high_risk_flag       (1 during RTMC-designated high-risk periods)
  - rtmc_risk_intensity       (0-1, peaks during festive season)
  - rtmc_period_label         (categorical: easter / festive_* / longweek_*)
  - days_to_easter / days_from_easter  (Easter is a moveable feast)
  - days_to_xmas              (Christmas approach effect)

## To upgrade with REAL daily counts (manual)
Sources for daily / weekly accident data in South Africa:

### RTMC (Road Traffic Management Corporation)
- Quarterly State of Road Safety: https://www.rtmc.co.za/index.php/publications
- Annual report includes provincial breakdown (Gauteng-specific)
- Festive season + Easter bulletins published in April + December
- For daily granularity: formal data request to data@rtmc.co.za

### Arrive Alive
- https://www.arrivealive.mobi/
- Daily incident reports on major routes (N1, N4, N14 - all serving Pretoria)
- Scrapeable

### City of Tshwane / Gauteng Traffic Police
- Press releases on serious accidents on Tshwane roads
- Tshwane EMS reports (sometimes shared with Steve Biko)

### News-based proxy
- GDELT 2.0 events (see templates/gdelt_news.csv)
  Filter EventRootCode = '07' (assault / accidents) AND ActionGeo_ADM1Code = 'SF06'

## Why
Road trauma is a major Steve Biko ED workload driver. RTMC reports show
30-100% fatality spikes in festive + Easter periods. Even without daily
counts, the period flags (already in rtmc_road_risk.csv) capture the
predictable variance.

## Schema (for manual fill)
- rtmc_fatalities_national       daily national road deaths
- rtmc_fatalities_gauteng        daily Gauteng road deaths
- rtmc_crashes_serious_gauteng   daily serious crashes
- ems_call_volume_tshwane        daily 10177 EMS calls in Tshwane
- n14_n4_n12_incidents           incidents on Pretoria-serving routes
"""
    write_template(TPL_DIR / "rtmc_daily_accidents.csv", schema, readme)
    return out


# =========================================================================
# TIER B
# =========================================================================

# B1 ----------------------------------------------------------------------
def collect_openmeteo_weather(start: pd.Timestamp, end: pd.Timestamp,
                              allow_network: bool = True) -> Optional[pd.DataFrame]:
    """Hourly + daily weather from Open-Meteo Archive (free, no key)."""
    if not (HAS_REQUESTS and allow_network):
        schema = ["date", "temp_max_C", "temp_min_C", "temp_mean_C",
                  "precip_sum_mm", "wind_max_kmh", "wind_gusts_kmh",
                  "humidity_mean_pct", "humidity_min_pct",
                  "pressure_mean_hPa", "heat_wave_flag", "storm_flag"]
        readme = """# B1. Open-Meteo weather — manual fallback

## Auto run
  pip install requests
  python scripts/collect_external_signals.py --tier B

## Manual alternative
1. Visit https://open-meteo.com/en/docs/historical-weather-api
2. Set latitude=-25.7461, longitude=28.1881
3. Set start/end dates and the daily + hourly variables listed in code.
4. Download CSV, drop in here.
"""
        write_template(TPL_DIR / "openmeteo_weather.csv", schema, readme)
        return None

    url = "https://archive-api.open-meteo.com/v1/archive"
    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "rain_sum",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "weather_code", "sunshine_duration", "shortwave_radiation_sum",
    ]
    hourly_vars = ["relative_humidity_2m", "surface_pressure", "apparent_temperature"]
    params = {
        "latitude": PRETORIA_LAT,
        "longitude": PRETORIA_LON,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "daily": ",".join(daily_vars),
        "hourly": ",".join(hourly_vars),
        "timezone": TIMEZONE,
        "wind_speed_unit": "kmh",
    }

    try:
        r = _safe_get(url, params=params, timeout=180)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"  [warn] Open-Meteo fetch failed: {exc}")
        return None

    if "daily" not in data:
        print(f"  [warn] Open-Meteo returned no daily block. Reason: {data.get('reason')}")
        return None

    daily = pd.DataFrame(data["daily"])
    daily["date"] = pd.to_datetime(daily["time"])
    daily = daily.drop(columns=["time"]).rename(columns={
        "temperature_2m_max": "temp_max_C",
        "temperature_2m_min": "temp_min_C",
        "temperature_2m_mean": "temp_mean_C_openmeteo",
        "precipitation_sum": "precip_sum_mm",
        "rain_sum": "rain_sum_mm",
        "wind_speed_10m_max": "wind_max_kmh_openmeteo",
        "wind_gusts_10m_max": "wind_gusts_kmh",
        "weather_code": "wmo_weather_code",
        "sunshine_duration": "sunshine_seconds",
        "shortwave_radiation_sum": "solar_MJ_per_m2",
    })

    # Hourly → daily aggregates
    if "hourly" in data:
        hourly = pd.DataFrame(data["hourly"])
        hourly["datetime"] = pd.to_datetime(hourly["time"])
        hourly["date"] = hourly["datetime"].dt.normalize()
        agg = hourly.groupby("date").agg(
            humidity_mean_pct=("relative_humidity_2m", "mean"),
            humidity_min_pct=("relative_humidity_2m", "min"),
            humidity_max_pct=("relative_humidity_2m", "max"),
            pressure_mean_hPa=("surface_pressure", "mean"),
            apparent_temp_max_C=("apparent_temperature", "max"),
            apparent_temp_min_C=("apparent_temperature", "min"),
        ).reset_index()
        daily = daily.merge(agg, on="date", how="left")

    # Engineered flags
    daily["heat_wave_flag"] = (
        (daily["temp_max_C"] > 30).rolling(3, min_periods=3).sum() == 3
    ).astype(int)
    # WMO codes 95, 96, 99 = thunderstorm variants
    daily["storm_flag"] = daily["wmo_weather_code"].isin([95, 96, 99]).astype(int)
    daily["heavy_rain_flag"] = (daily["precip_sum_mm"] >= 20).astype(int)
    daily["cold_snap_flag"] = (daily["temp_min_C"] < 5).astype(int)

    # Move date to first column
    cols = ["date"] + [c for c in daily.columns if c != "date"]
    daily = daily[cols]

    write_csv(daily, OUT_DIR / "openmeteo_weather.csv", "Pretoria, daily + agg")
    return daily


# B2 ----------------------------------------------------------------------
def collect_saaqis_air_quality(start: pd.Timestamp, end: pd.Timestamp) -> None:
    schema = ["date", "pm25_mean_ug_m3", "pm10_mean_ug_m3",
              "no2_mean_ppb", "so2_mean_ppb", "o3_max_ppb", "aqi_max"]
    readme = """# B2. SAAQIS Air Quality — Manual Collection

## Source
SAAQIS portal: https://saaqis.environment.gov.za/
Pretoria monitoring stations (Gauteng province):
  - Pretoria CBD
  - Diepsloot
  - Booysens (closest to Steve Biko, depending on availability)
  - Rosslyn

## Why
PM2.5 / NO2 peaks correlate with respiratory and cardiac ED presentations.
COPD and asthma flare on high-pollution days.

## Workflow
1. Log in at saaqis.environment.gov.za (free registration)
2. Select 'Data > Download'
3. Pick station(s), pollutants (PM2.5, PM10, NO2, SO2, O3), and date range
4. Export hourly CSV; aggregate to daily means/maxes outside this script
   or extend this collector when permission is granted.

## Schema
- pm25_mean_ug_m3   daily mean PM2.5
- pm10_mean_ug_m3   daily mean PM10
- no2_mean_ppb      daily mean NO2
- so2_mean_ppb      daily mean SO2
- o3_max_ppb        daily 1-hour max O3
- aqi_max           daily max composite AQI (if computed)
"""
    write_template(TPL_DIR / "saaqis_air_quality.csv", schema, readme)


# B3 ----------------------------------------------------------------------
def collect_gdelt_news(start: pd.Timestamp, end: pd.Timestamp) -> None:
    schema = ["date", "n_protests_tshwane", "n_accidents_tshwane",
              "n_health_alerts_gauteng", "n_violent_incidents",
              "avg_tone_health_news"]
    readme = """# B3. GDELT 2.0 News / Events — BigQuery Setup

## Source
GDELT Project: https://www.gdeltproject.org/
Two access modes:

### Mode A: Article Search API (limited, free)
  https://api.gdeltproject.org/api/v2/doc/doc
Example:
  /doc?query=protest%20pretoria&mode=ArtList&format=JSON&maxrecords=250
  &startdatetime=20240101000000&enddatetime=20241231235959

### Mode B: BigQuery public dataset (RECOMMENDED for backfill)
  gdelt-bq.gdeltv2.events
  gdelt-bq.gdeltv2.gkg
Free Google Cloud BigQuery tier: 1 TB/month query.

Example query for daily protest counts near Pretoria:
  SELECT DATE(PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS d,
         COUNT(*) AS n
  FROM `gdelt-bq.gdeltv2.events`
  WHERE EventRootCode = '14'              -- protests
    AND ActionGeo_CountryCode = 'SF'      -- South Africa (GDELT code)
    AND ActionGeo_ADM1Code IN ('SF06')    -- Gauteng
    AND SQLDATE BETWEEN 20190501 AND 20260131
  GROUP BY d ORDER BY d

## Why
Local protests, mass-casualty events, transport strikes drive sharp
short-window ED spikes. GDELT captures these from news-text mining
(English + Afrikaans + Zulu sources).
"""
    write_template(TPL_DIR / "gdelt_news.csv", schema, readme)


# B4 ----------------------------------------------------------------------
def collect_events_loftus(start: pd.Timestamp, end: pd.Timestamp) -> None:
    schema = ["date", "loftus_event", "loftus_attendance_est",
              "concert_in_tshwane", "concert_capacity_est",
              "rugby_match", "soccer_match"]
    readme = """# B4. Mass Events in Tshwane — Manual

## Sources
- Loftus Versfeld stadium: https://www.bluebulls.co.za/fixtures/
  (Currie Cup, URC, rugby tests; capacity ~52,000)
- Lucas Moripe / Tshwane stadiums
- Computicket: https://www.computicket.com (concert / event listings)
- Quicket: https://www.quicket.co.za
- SuperSport TV schedule (back-dates rugby/soccer fixtures)

## Why
Rugby matches at Loftus drive predictable trauma / intoxication surges
2-6 hours post-final-whistle. Concerts produce similar but smaller spikes.

## Workflow
1. For each year, build a list of dates with:
   - Loftus event (yes/no + estimated attendance)
   - Concert in Pretoria/Centurion (yes/no + capacity)
2. Most fixtures are publicised retrospectively in news articles.
3. Save one row per event date.

## Schema
- loftus_event             0/1
- loftus_attendance_est    estimated, integer
- concert_in_tshwane       0/1
- concert_capacity_est     integer
- rugby_match              0/1 (any rugby in Tshwane)
- soccer_match             0/1
"""
    write_template(TPL_DIR / "events_loftus.csv", schema, readme)


# =========================================================================
# TIER C — internal data (templates only)
# =========================================================================

def collect_phc_referrals_template(start, end) -> None:
    schema = ["date", "referrals_in_count", "referrals_in_acuity_mean",
              "self_referrals_count", "referral_top_clinic"]
    readme = """# C1. Primary Health Care (PHC) referral counts — INTERNAL

## Source
Steve Biko Hospital administration / Tshwane Health District.
Requires data sharing agreement.

## Why
ED is the overflow buffer for surrounding clinics. Higher upstream PHC
referral volume on day D directly increases ED arrivals D+0 and D+1.

## Schema
- referrals_in_count            count of formal referrals to ED that day
- referrals_in_acuity_mean      mean acuity score of those referrals
- self_referrals_count          count of walk-ins (non-referred)
- referral_top_clinic           categorical, top contributor clinic name
"""
    write_template(TPL_DIR / "phc_referrals.csv", schema, readme)


def collect_neighbouring_eds_template(start, end) -> None:
    schema = ["date", "tembisa_arrivals", "kalafong_arrivals",
              "mamelodi_arrivals", "george_mukhari_arrivals",
              "spillover_index"]
    readme = """# C2. Neighbouring ED Volumes — INTERNAL

## Source
Gauteng Department of Health data office (formal request).
Hospitals: Tembisa, Kalafong, Mamelodi, George Mukhari, Pretoria West.

## Why
If a neighbouring ED is overwhelmed or temporarily diverting, spillover
goes to Steve Biko (and vice versa). Lag-1 / lag-2 neighbouring counts
are a leading indicator.

## Schema
- tembisa_arrivals             daily count at Tembisa ED
- kalafong_arrivals            daily count at Kalafong ED
- mamelodi_arrivals            daily count at Mamelodi ED
- george_mukhari_arrivals      daily count at George Mukhari ED
- spillover_index              engineered: deviation from each ED's own dow-mean
"""
    write_template(TPL_DIR / "neighbouring_eds.csv", schema, readme)


def collect_nhls_lab_template(start, end) -> None:
    schema = ["date", "nhls_blood_culture_count", "nhls_resp_panel_count",
              "nhls_hba1c_count", "nhls_tb_smear_count"]
    readme = """# C3. NHLS Lab Test Volumes (Catchment) — INTERNAL

## Source
National Health Laboratory Service (NHLS) — data sharing agreement required.
Tshwane catchment: Steve Biko + connected PHC labs.

## Why
Test volumes are a real-time proxy for clinical demand. Surges in
respiratory panels precede flu wave ED peaks by ~1 week; blood culture
surges align with sepsis presentations.

## Schema
- nhls_blood_culture_count     daily across catchment
- nhls_resp_panel_count        respiratory PCR / culture tests
- nhls_hba1c_count             diabetes monitoring tests (chronic-care proxy)
- nhls_tb_smear_count          TB tests
"""
    write_template(TPL_DIR / "nhls_lab_volumes.csv", schema, readme)


# =========================================================================
# JOIN STEP
# =========================================================================

def join_signals(start: pd.Timestamp, end: pd.Timestamp,
                 collected: dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Merge all collected real-data frames onto a daily skeleton + G1."""
    skeleton = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

    # Attach G1 target if available
    try:
        from src.forecasting.io import load_g1
        g1 = load_g1().reset_index().rename(columns={"index": "date"})
        if "date" not in g1.columns:
            g1 = g1.rename(columns={g1.columns[0]: "date"})
        g1["date"] = pd.to_datetime(g1["date"])
        skeleton = skeleton.merge(g1, on="date", how="left")
    except Exception as exc:
        print(f"  [warn] Skipping G1 join: {exc}")

    # Weather (daily)
    if "weather" in collected:
        w = collected["weather"].copy()
        w["date"] = pd.to_datetime(w["date"])
        skeleton = skeleton.merge(w, on="date", how="left", suffixes=("", "_openmeteo"))

    # Eskom (daily)
    if "eskom" in collected:
        e = collected["eskom"].copy()
        e["date"] = pd.to_datetime(e["date"])
        skeleton = skeleton.merge(e, on="date", how="left")

    # SASSA (daily)
    if "sassa" in collected:
        s = collected["sassa"].copy()
        s["date"] = pd.to_datetime(s["date"])
        skeleton = skeleton.merge(s, on="date", how="left")

    # Google Trends (weekly → forward-fill to daily)
    if "gtrends" in collected:
        gt = collected["gtrends"].copy()
        gt["week_start"] = pd.to_datetime(gt["week_start"])
        gt = gt.set_index("week_start").resample("D").ffill().reset_index()
        gt = gt.rename(columns={"week_start": "date"})
        skeleton = skeleton.merge(gt, on="date", how="left")

    # NICD / flu proxy (weekly → forward-fill to daily)
    if "flu" in collected:
        fl = collected["flu"].copy()
        fl["week_start"] = pd.to_datetime(fl["week_start"])
        fl = fl.set_index("week_start").resample("D").ffill().reset_index()
        fl = fl.rename(columns={"week_start": "date"})
        skeleton = skeleton.merge(fl, on="date", how="left")

    # School calendar (daily)
    if "school" in collected:
        sc = collected["school"].copy()
        sc["date"] = pd.to_datetime(sc["date"])
        skeleton = skeleton.merge(sc, on="date", how="left", suffixes=("", "_school"))

    # Road accidents / RTMC risk periods (daily)
    if "roads" in collected:
        rd = collected["roads"].copy()
        rd["date"] = pd.to_datetime(rd["date"])
        skeleton = skeleton.merge(rd, on="date", how="left")

    out_path = OUT_DIR / "g1_enriched.csv"
    skeleton.to_csv(out_path, index=False)
    print(f"\n  [join] {out_path.relative_to(ROOT)}  "
          f"({len(skeleton):,} rows, {skeleton.shape[1]} cols)")
    return skeleton


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tier", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--start", default=None,
                        help="Override start date (YYYY-MM-DD). Default: G1.min()")
    parser.add_argument("--end", default=None,
                        help="Override end date (YYYY-MM-DD). Default: G1.max()")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip all network calls; only emit templates")
    args = parser.parse_args()

    if args.start and args.end:
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
    else:
        start, end = get_date_range_from_g1()

    allow_network = not args.no_network

    print("=" * 72)
    print(f"External-signals collector — Pretoria / Tshwane")
    print(f"  date range : {start.date()} -> {end.date()}  ({(end-start).days+1} days)")
    print(f"  coordinates: lat={PRETORIA_LAT}, lon={PRETORIA_LON}")
    print(f"  tier       : {args.tier}")
    print(f"  network    : {'on' if allow_network else 'off (templates only)'}")
    print(f"  requests   : {'available' if HAS_REQUESTS else 'MISSING (pip install requests)'}")
    print(f"  pytrends   : {'available' if HAS_PYTRENDS else 'MISSING (pip install pytrends)'}")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TPL_DIR.mkdir(parents=True, exist_ok=True)

    collected: dict[str, pd.DataFrame] = {}

    if args.tier in ("A", "all"):
        print("\n--- Tier A: free, automated where possible ---")
        flu = collect_nicd_flu(start, end, allow_network)
        if flu is not None:
            collected["flu"] = flu
        gt = collect_google_trends(start, end, allow_network)
        if gt is not None:
            collected["gtrends"] = gt
        es = collect_eskom_loadshedding(start, end, allow_network)
        if es is not None:
            collected["eskom"] = es
        collected["sassa"] = collect_sassa_payments(start, end)
        collected["school"] = collect_school_calendar(start, end)
        collected["roads"] = collect_road_accidents(start, end)

    if args.tier in ("B", "all"):
        print("\n--- Tier B: free, partial automation ---")
        wx = collect_openmeteo_weather(start, end, allow_network)
        if wx is not None:
            collected["weather"] = wx
        collect_saaqis_air_quality(start, end)
        collect_gdelt_news(start, end)
        collect_events_loftus(start, end)

    if args.tier in ("C", "all"):
        print("\n--- Tier C: requires permission (templates only) ---")
        collect_phc_referrals_template(start, end)
        collect_neighbouring_eds_template(start, end)
        collect_nhls_lab_template(start, end)

    if collected:
        print("\n--- Join step ---")
        join_signals(start, end, collected)

    print("\nDone.")
    print(f"Output dir: {OUT_DIR.relative_to(ROOT)}")
    print(f"Templates : {TPL_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
