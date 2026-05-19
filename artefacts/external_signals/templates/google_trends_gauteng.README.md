# A2. Google Trends — pytrends FAILED at run time

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
