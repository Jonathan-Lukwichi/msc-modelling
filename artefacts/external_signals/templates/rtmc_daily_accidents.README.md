# A6. Road Accidents - Optional manual upgrade (REAL daily counts)

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
