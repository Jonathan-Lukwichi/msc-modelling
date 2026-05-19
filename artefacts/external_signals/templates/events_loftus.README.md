# B4. Mass Events in Tshwane — Manual

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
