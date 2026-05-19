# B3. GDELT 2.0 News / Events — BigQuery Setup

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
