# A3. Eskom Load-Shedding — Two options

## Option 1 (RECOMMENDED for backfill): eskom-calendar dataset
GitHub: https://github.com/beyarkay/eskom-calendar
Releases: https://github.com/beyarkay/eskom-calendar/releases
File: machine_friendly.csv (download latest release)

This script attempts to fetch the latest release automatically.

## Option 2: EskomSePush Business API (current / recent only)
Sign up: https://eskomsepush.gumroad.com/l/api  (free dev tier: 50 req/day)
Bearer token in env var ESPUSH_TOKEN.
Endpoint: https://developer.sepush.co.za/business/2.0/

## Columns
- date                       (one row per calendar day)
- stage_max                  highest stage in effect on the day (0-8)
- stage_mean                 hours-weighted mean stage
- hours_off_total            sum of load-shed hours across the day
- hours_off_business_hours   subset 07:00-19:00 (ED-relevant)
- n_events                   count of distinct load-shed slots
