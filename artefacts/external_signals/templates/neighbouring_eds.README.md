# C2. Neighbouring ED Volumes — INTERNAL

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
