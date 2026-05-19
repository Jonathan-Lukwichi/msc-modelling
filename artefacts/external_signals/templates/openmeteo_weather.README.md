# B1. Open-Meteo weather — manual fallback

## Auto run
  pip install requests
  python scripts/collect_external_signals.py --tier B

## Manual alternative
1. Visit https://open-meteo.com/en/docs/historical-weather-api
2. Set latitude=-25.7461, longitude=28.1881
3. Set start/end dates and the daily + hourly variables listed in code.
4. Download CSV, drop in here.
