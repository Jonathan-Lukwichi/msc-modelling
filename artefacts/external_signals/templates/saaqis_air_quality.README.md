# B2. SAAQIS Air Quality — Manual Collection

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
