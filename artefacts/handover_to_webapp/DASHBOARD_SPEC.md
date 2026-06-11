# Dashboard / UI Specification

This document tells the integrator (the other Claude CLI session) exactly what the web app must show.

---

## 1. Top-Level Navigation

The app has **two main sections** at the top level:

```
┌─────────────────────────────────────────────────────────────┐
│  Steve Biko ED Forecast                          [User Menu] │
├─────────────────────────────────────────────────────────────┤
│  ▶ Task 1: Daily Total ED Arrivals                          │
│  ▶ Task 2: Per-Specialty Arrivals                           │
└─────────────────────────────────────────────────────────────┘
```

Clicking each takes the user to a separate page. **They are not mixed in a single dropdown.**

---

## 2. Common Components (both tasks)

### 2.1 Accuracy Badge (mandatory)

Every model card, dropdown option, and forecast result must show a colour badge based on the model's val MAPE.

| Badge | Range | Label | Meaning |
|---|---|---|---|
| 🟢 | val MAPE < 15% | **Operational** | Safe for next-day staffing decisions |
| 🟡 | val MAPE 15–30% | **Planning** | Suitable for week-ahead / monthly planning, NOT for daily staffing |
| 🔴 | val MAPE > 30% | **Research preview** | Trend visualisation only — do NOT base operational decisions on this model |

#### Visual mockup

```
┌──────────────────────────────────────────────────────────────┐
│  Pick a model                                                 │
├──────────────────────────────────────────────────────────────┤
│  ⚪ Hybrid 1     val MAPE 12.04%       🟢 Operational         │
│  ⚪ Stat 2       val MAPE 12.34%       🟢 Operational         │
│  ⚪ ML 1         val MAPE 11.96%       🟢 Operational         │
│  ⚪ ML 2         val MAPE 12.32%       🟢 Operational         │
│  ⚪ Stat 1       val MAPE 13.33%       🟢 Operational         │
│  ⚪ Hybrid 2     val MAPE 12.19%       🟢 Operational         │
└──────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│  Orthopaedics — Pick a model                                  │
├──────────────────────────────────────────────────────────────┤
│  ⚪ Stat 1       val MAPE 84.60%       🔴 Research preview    │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Horizon Selector

The user always picks one of these horizons:

| Option | Days predicted | Aggregation |
|---|---|---|
| **1 day ahead** | 1 | none |
| **7 days ahead** | 7 | none (show day-by-day) |
| **Monthly** | 30 | sum daily forecasts → 1 monthly total |
| **Yearly** | 365 | sum daily forecasts → 1 yearly total |

Weekly specialties (Maternity, Psychiatry) get a different selector:

| Option | Weeks predicted | Aggregation |
|---|---|---|
| **1 week ahead** | 1 | none |
| **4 weeks ahead** | 4 | none |
| **Yearly** | 52 | sum weekly forecasts |

### 2.3 Output Display

After the user picks model + horizon + start date and clicks "Forecast":

```
┌──────────────────────────────────────────────────────────────┐
│  Forecast — Task 1, Model Hybrid 1 🟢, Horizon: 7 days        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [line chart: predicted blue line, optional history grey]   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Date         │ Predicted │ ± uncertainty (if available)     │
│  2026-06-12   │     65    │                                  │
│  2026-06-13   │     67    │                                  │
│  ...                                                         │
├──────────────────────────────────────────────────────────────┤
│  [ Download CSV ]   [ Try another model ]                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Task 1 Page (Daily Total ED Arrivals)

### 3.1 Model picker

Show all 6 aliases (Stat 1, Stat 2, ML 1, ML 2, Hybrid 1, Hybrid 2) with their accuracy badges. Sort by val_RMSE ascending (best first).

### 3.2 Headline metrics panel

```
┌──────────────────────────────────────────────────────────────┐
│  Model performance — Hybrid 1                                 │
├──────────────────────────────────────────────────────────────┤
│  Daily forecast   :  12.04% MAPE   🟢 Operational             │
│  Weekly forecast  :   6.40% error  🟢 Operational             │
│  Monthly forecast :   2.47% error  🟢 Operational             │
│  Yearly forecast  :   0.38% error  🟢 Operational             │
└──────────────────────────────────────────────────────────────┘
```

Data source: `task1_daily_arrivals/metrics/per_horizon.json`.

### 3.3 "Compare models" tab

Show a table with all 6 models × all 4 horizons. Each cell coloured by its badge tier.

---

## 4. Task 2 Page (Per-Specialty Arrivals)

### 4.1 Specialty picker

A single dropdown with all 7 specialties:

```
Specialty: [ Medicine ▼ ]
           ┌───────────────────┐
           │ Medicine          │
           │ Orthopaedics      │
           │ Surgery           │
           │ Gynaecology       │
           │ Paediatrics       │
           │ Maternity (weekly)│
           │ Psychiatry (weekly)│
           └───────────────────┘
```

> When the user picks **Maternity** or **Psychiatry**, the horizon selector auto-switches from daily mode to weekly mode (see section 2.2).

### 4.2 Filtered model picker

After the specialty is picked, the model dropdown is filtered to only show aliases that have a trained pickle for that specialty.

```
Specialty: Medicine
Models available: Stat 1 🟡, ML 1 🟡, ML 2 🟡

Specialty: Orthopaedics
Models available: Stat 1 🔴  ← only this one is shown
```

### 4.3 Per-specialty headline metrics

Same panel as Task 1 but per specialty:

```
┌──────────────────────────────────────────────────────────────┐
│  Medicine — Stat 1                                            │
├──────────────────────────────────────────────────────────────┤
│  Daily forecast MAPE  : 21.4%       🟡 Planning use only      │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Dashboard / "About this model" Page

For each model alias the user can click "Details" to open a dedicated info panel:

- Alias (large heading)
- Accuracy badge
- val MAPE / RMSE / MAE / R² (numeric table)
- Per-horizon error table
- A small "About" paragraph (from `cards/{alias}.json` → `description`)
- Training data window
- Last retrain date
- **NO scientific name shown** — but a small "i" tooltip can reveal it if the user is logged in as engineer/admin role.

---

## 6. Mobile-Friendly Considerations

- The forecast chart should reflow on narrow screens.
- The model picker becomes a single dropdown on mobile (not a 6-card grid).
- Badge colour must remain visible (do not collapse to text).

---

## 7. Accessibility

- Badge colour ALONE must not encode meaning. Always show the badge text label too: 🟢 "Operational" / 🟡 "Planning" / 🔴 "Research preview".
- All badge colours must satisfy WCAG AA contrast.
- Charts must include alt-text describing the trend in plain English (auto-generated from the forecast values is acceptable).

---

## 8. Error / Empty States

- If a forecast endpoint returns an error: show "Forecast temporarily unavailable" + retry button. Do NOT show the Python traceback to the user.
- If a model isn't trained for a specialty: do NOT show that alias in the dropdown at all (filter it out).
- If the user picks a 365-day horizon: show a banner "Yearly forecasts are aggregates; daily/weekly accuracy applies to underlying predictions."
