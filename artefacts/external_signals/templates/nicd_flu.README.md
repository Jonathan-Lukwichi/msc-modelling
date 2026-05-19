# A1. NICD Respiratory Surveillance — Optional manual upgrade

## What the script automatically produces (artefacts/external_signals/nicd_flu_proxy.csv)
A weekly proxy combining:
  - WHO FluNet ZAF positivity (if API reachable)
  - Published SA seasonal flu pattern (Tempia 2018, peak ISO week 27)
  - Hardcoded NICD-documented COVID waves (5 waves, 2020-2022)

## Manual upgrade (HIGHER FIDELITY)
For sharper accuracy, fill this CSV from the NICD weekly bulletins:
  https://www.nicd.ac.za/diseases-a-z-index/disease-index-influenza/surveillance-reports/

For each ISO Monday between 2019-05-01 and 2026-01-31:
  - flu_activity_idx_nicd   :  1=low, 2=moderate, 3=high, 4=very high
  - ili_rate_per1k_nicd     :  Influenza-Like Illness consultation rate
  - rsv_positivity_pct_nicd :  RSV %
  - covid_wastewater_nicd   :  NICD wastewater signal (Daspoort if reported)

The join step prefers manual columns when present; falls back to the proxy otherwise.
