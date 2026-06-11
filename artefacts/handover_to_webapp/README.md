# Steve Biko ED Forecasting — Web-App Handover Package

**Source thesis**: *Modelling and Evaluation of Emergency Department Patient Arrivals at Steve Biko Academic Hospital, Pretoria, South Africa* — University of Pretoria, 2026.

**Hospital**: Steve Biko Academic Hospital, Pretoria, Gauteng, South Africa.

**Generated**: 2026-06-11

This package contains every artefact needed to deploy **two distinct forecasting products** as separate sections of one web application. The integrator should treat Task 1 and Task 2 as logically independent products that share a UI shell.

---

## TWO TASKS, TWO APP SECTIONS

### Task 1 — Daily Total ED Arrivals
- **Target**: Single time-series, the daily total number of patients walking into the ED.
- **Resolution**: Daily (1 prediction per day).
- **Horizons supported**: 1-day, 7-day, monthly, yearly (aggregated from daily).
- **Use case**: hospital-wide capacity planning, nurse rostering by date, monthly budget allocation.
- **Models**: 6 deployed (Stat 1, Stat 2, ML 1, ML 2, Hybrid 1, Hybrid 2). See `task1_daily_arrivals/`.

### Task 2 — Per-Specialty Arrivals
- **Targets**: 7 separate sub-products, one per clinical specialty.
- **Resolution**: Daily for 5 specialties (Medicine, Orthopaedics, Surgery, Gynaecology, Paediatrics) — Weekly only for 2 specialties (Maternity, Psychiatry — ≤2 patients/day average means daily forecasts are not meaningful).
- **Horizons supported**: 1-day / 7-day / monthly / yearly for daily specialties; 1-week / 4-week / yearly for weekly specialties.
- **Use case**: specialty-level resource allocation, on-call rota, ward bed planning.
- **Models**: 13 deployed total (varies per specialty — only models that produce useful forecasts are shipped). See `task2_specialties/`.

> **CRITICAL UX RULE**: Tasks 1 and 2 are TWO SEPARATE PAGES / SECTIONS in the app. Do not mix them in a single dropdown. A user choosing "Forecast" should first pick Task 1 or Task 2, then proceed.

---

## CONVENTIONAL MODEL NAMING

The app **always** refers to models by their conventional alias, **never** by their scientific name. Scientific names appear only in `_internal_only/alias_scientific_mapping.json` for engineering audit.

| Alias | Type | Use in app |
|---|---|---|
| **Stat 1** | Statistical | dropdown label, badge, URL slug |
| **Stat 2** | Statistical | same |
| **ML 1** | Machine Learning | same |
| **ML 2** | Machine Learning | same |
| **Hybrid 1** | Hybrid (Stat + ML) | same |
| **Hybrid 2** | Hybrid (Stat + ML) | same |

See `ALIASES.md` for the full alias → file mapping.

---

## ACCURACY BADGE SYSTEM (mandatory in UI)

Every model card / forecast result must show a colour badge based on its val MAPE:

| Badge | Range | Label | Meaning for user |
|---|---|---|---|
| 🟢 | val MAPE < 15% | **Operational** | Safe for next-day staffing decisions |
| 🟡 | val MAPE 15–30% | **Planning** | Suitable for week-ahead / monthly planning, NOT for daily staffing |
| 🔴 | val MAPE > 30% | **Research** | Trend visualisation only — do NOT base operational decisions on this model |

See `DASHBOARD_SPEC.md` for visual specification.

---

## FOLDER MAP

```
handover_to_webapp/
├── README.md                          ← you are here
├── ALIASES.md                          alias mapping
├── DASHBOARD_SPEC.md                   UI spec + badge rules
├── api_spec.yaml                       REST API contract (OpenAPI 3.0)
├── requirements.txt                    pip dependencies
│
├── task1_daily_arrivals/               ← Task 1 (6 pickles)
│   ├── README.md
│   ├── models/{stat1,stat2,ml1,ml2,hybrid1,hybrid2}.pkl
│   ├── metrics/{headline,per_horizon,cv_folds}.json
│   ├── cards/{stat1,...}.json
│   └── inference/{load,forecast,example_demo}.py
│
├── task2_specialties/                  ← Task 2 (13 pickles)
│   ├── README.md
│   ├── medicine/                       (3 models: stat1, ml1, ml2)
│   ├── orthopaedics/                   (1 model: stat1)
│   ├── surgery/                        (2 models: ml1, ml2)
│   ├── gynaecology/                    (3 models: stat1, ml1, ml2)
│   ├── paediatrics/                    (2 models: stat1, ml2)
│   ├── maternity_weekly/               (1 model: stat2 — weekly only)
│   ├── psychiatry_weekly/              (1 model: stat2 — weekly only)
│   └── inference/{load,forecast,example_demo}.py
│
└── _internal_only/                     ← engineering audit (hidden from UI)
    ├── alias_scientific_mapping.json
    └── training_provenance.json
```

---

## QUICK START FOR THE INTEGRATOR

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Smoke test Task 1
python task1_daily_arrivals/inference/example_demo.py

# 3. Smoke test Task 2
python task2_specialties/inference/example_demo.py

# 4. Open api_spec.yaml in Swagger Editor / Postman to inspect REST contract
```

---

## DEPLOYMENT MUST-HAVES (read before shipping)

1. **Re-training cadence**: monthly, on a rolling 24-month window of new data.
2. **Drift monitoring**: track KS distance between recent month and training distribution; soft alarm at 0.20, hard retrain at 0.35 (Chapter 5 §5.5.2 thresholds).
3. **Prediction intervals**: not bundled in this package (point forecasts only). Use conformal intervals from `artefacts/uq/` if needed (out of scope of this handover).
4. **POPIA compliance**: any patient data fed into the system must be de-identified per POPIA (South African POPIA Act).
5. **Liability**: these are research-grade artefacts produced for academic evaluation. Clinical deployment requires ethical approval, multi-site validation, and a manual override pathway.

---

## CONTACT / AUDIT TRAIL

- Thesis source: University of Pretoria, 2026
- Training data window: 2022-03-01 → 2024-12-31 (post-COVID block)
- Validation block: 2024-07-01 → 2024-12-31 (184 days)
- Test block: 2025-01-01 → 2026-01-31 (396 days, out-of-distribution)
- HPO traces: see `_internal_only/training_provenance.json`
