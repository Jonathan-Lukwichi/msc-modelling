# C3. NHLS Lab Test Volumes (Catchment) — INTERNAL

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
