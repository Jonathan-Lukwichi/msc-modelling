# §7.10 (App-aligned framing) Joint Forecast-Driven Replenishment and Rostering

This subsection complements the academic Chapter 3 formalisation reported in §7.10.A by re-running the integrated experiment under the parameter framing the production web app uses. The framing has three changes from §7.10.A. First, the nurse pool is the 23 filled posts (9 Professional Nurses, 8 Enrolled Nurses, 6 Enrolled Nursing Auxiliaries) that reflect the 23 percent vacancy rate of \parencite{rispel2014} applied to the 30 nominal posts. Second, the integer programme runs over three shifts per day (Day, Evening, Night) under the NDoH nurse-to-patient ratios 1:4, 1:5, and 1:6 \parencite{NDoH2012Norms}. Third, the baseline against which savings are measured is the **naive supply policy with busy-day staffing**, meaning no $(s, S)$ thresholds in inventory (orders only when stock reaches zero) and a roster sized to the peak day every day. The baseline is therefore deliberately uncalibrated, reflecting what a no-operations-research-support hospital actually does.

The chapter reports the academic framing of §7.10.A as its main empirical finding and reports this app-aligned framing as a complementary commercial framing.

## 7.10.5 Configurations compared

Seven configurations are evaluated under common random numbers across 5 independent seeds and the same 396-day Steve Biko ED arrivals window.

1. **Naive supply, Busy-day staffing** (the baseline, what a no-OR-support hospital does).
2. Naive supply, Average staffing.
3. Textbook $(s, S)$ supply (Silver 2017), Busy-day staffing.
4. Textbook $(s, S)$ supply, Average staffing.
5. Forecast-driven $(s, S)$ supply (Algorithm 8), Average staffing.
6. Textbook $(s, S)$ supply, Forecast-driven staffing (Algorithm 9 with XGBoost weekly forecast).
7. **Forecast-driven $(s, S)$ supply plus Forecast-driven staffing** (the full app chain).

The Monte Carlo budget for Algorithm 8 is raised to $R = 200$ replications with an 8-point grid over the order-up-to level $S$, against the $R = 30$, 6-point setting used in §7.10.A.

## 7.10.6 Results

The headline numbers, annualised from the 396-day window, are in Table 7.10.5. Saving columns are measured against the naive baseline.

| Rank | Method | Annual Inv. | Annual Sched. | **Annual Total** | Saving vs Naive Busy-day |
|---|---|---|---|---|---|
| 1 | **Forecast (s,S) + Forecast staffing** (the full app chain) | R 1 348 282 | R 25 433 366 | **R 26 781 647** | **R 14 814 362 (36 percent)** |
| 2 | Textbook (s,S) + Forecast staffing | R 1 588 676 | R 25 437 377 | R 27 026 053 | R 14 569 956 (35 percent) |
| 3 | Forecast (s,S) + Average staffing | R 1 348 282 | R 26 118 317 | R 27 466 599 | R 14 129 410 (34 percent) |
| 4 | Textbook (s,S) + Average staffing | R 1 588 676 | R 26 118 317 | R 27 706 993 | R 13 889 016 (33 percent) |
| 5 | Naive supply + Average staffing | R 3 684 092 | R 26 118 317 | R 29 802 409 | R 11 793 600 (28 percent) |
| 6 | Textbook (s,S) + Busy-day staffing | R 1 588 676 | R 37 911 917 | R 39 500 593 | R 2 095 416 (5 percent) |
| 7 | **Naive supply + Busy-day staffing** (the baseline) | R 3 684 092 | R 37 911 917 | **R 41 596 009** | reference |

The headline saving of the **full forecast-driven chain over the no-operations-research-support baseline is R 14.8 million per year, or 36 percent of total annual operational cost**.

### *(Figure 7.14 image, `artefacts/chapter7/app_aligned/figures/figure_7_14_annual_total_cost.png`)*

Figure 7.14 reports the annual total cost per method, stacked to expose the inventory-versus-scheduling split. The naive baseline sits at R 41.6 million; the forecast-driven full chain sits at R 26.8 million. The visual gap between the two end bars is the R 14.8 million annual saving.

### *(Figure 7.15 image, `artefacts/chapter7/app_aligned/figures/figure_7_15_savings_vs_naive.png`)*

Figure 7.15 reports the annual saving for each configuration against the naive baseline as a horizontal bar chart, sorted by magnitude. The forecast-driven full chain delivers the largest saving; the textbook $(s, S)$ supply with busy-day staffing delivers the smallest non-zero saving, because the inventory improvement alone cannot offset the staff over-staffing under the busy-day policy.

### *(Figure 7.16 image, `artefacts/chapter7/app_aligned/figures/figure_7_16_weekly_cost.png`)*

Figure 7.16 reports the same numbers as weekly totals, matching the format the production web app shows on its KPI cards. The naive baseline costs R 800 thousand per week; the forecast-driven full chain costs R 515 thousand per week, a R 285 thousand per week saving.

### *(Figure 7.17 image, `artefacts/chapter7/app_aligned/figures/figure_7_17_supply_vs_staff_saving.png`)*

Figure 7.17 decomposes the saving into its supply (inventory) and staff (scheduling) components for each non-baseline configuration. The decomposition for the full chain is:

- Supply saving: R 2.34 million per year (R 3.68 million naive inventory cost reduced to R 1.35 million through forecast-driven $(s, S)$).
- Staff saving: R 12.48 million per year (R 37.91 million busy-day scheduling cost reduced to R 25.43 million through forecast-driven rostering).
- Joint saving: R 14.81 million per year.

The staff component dominates the saving because the busy-day policy carries a 43 percent over-staffing burden on average days (mean arrivals 69 patients, peak arrivals 99 patients), and the forecast-driven roster eliminates this over-staffing.

### *(Figure 7.18 image, `artefacts/chapter7/app_aligned/figures/figure_7_18_cost_decomposition.png`)*

Figure 7.18 decomposes the total cost of each method into its six components (holding, ordering, stockout penalty, expiry, payroll, locum). Three observations.

- The locum component dominates the scheduling cost under every configuration. The 23-active-nurse pool cannot deliver the required nurse-hours within the BCEA cap, so the residual demand is filled by locum staff at R 450 per hour.
- The stockout penalty dominates the inventory cost under the naive baseline (R 3.6 million of R 3.7 million); the forecast-driven $(s, S)$ policy collapses this to R 1.3 million.
- Holding cost rises from R 28 thousand under the naive baseline to R 44 thousand under the textbook and forecast policies, the small additional buffer the safety stock buys.

## 7.10.7 What the app-aligned framing hands forward

The app-aligned framing delivers the headline number the production web application shows users: **a forecast-driven joint inventory-and-rostering pipeline saves Steve Biko Academic Hospital approximately R 14.8 million per year against the no-operations-research-support baseline**. The decomposition is R 12.5 million from the staffing layer and R 2.3 million from the inventory layer.

Two interpretive notes for §7.11 (Discussion).

First, this saving is measured against a deliberately weak baseline (no $(s, S)$ at all, peak staffing every day). It is the right comparison for showing the value of an integrated operations-research-supported pipeline to a non-technical hospital manager. It is **not** the same number reported in §7.10.A, which measures the marginal value of the forecast against the textbook $(s, S)$ plus average-mean roster baseline that an operations-research-supported hospital is already using. The §7.10.A number (R 196 400 per year, 1.8 percent) is the marginal forecast contribution; this number (R 14.8 million per year, 36 percent) is the total operations-research-plus-forecast contribution.

Second, the structural finding from §7.10.A holds under this framing as well. Coverage at the BCEA 45-hour cap rises from 47 percent (busy-day) to 61 percent (forecast-driven), but the residual 39 percent gap is the structural shortage that no forecasting intervention can close. The R 14.8 million per year is what the hospital recovers through better operations; the structural shortage that \parencite{malakoane2020public} identify across the South African public-health system is what only hiring can recover.

---

## Source files

| Artefact | Path |
|---|---|
| Per-method summary, app-aligned | `artefacts/chapter7/app_aligned/app_aligned_summary.csv` |
| Run log | `artefacts/chapter7/app_aligned/run.log` |
| Figure 7.14, annual total cost stacked | `artefacts/chapter7/app_aligned/figures/figure_7_14_annual_total_cost.png` |
| Figure 7.15, savings vs naive baseline | `artefacts/chapter7/app_aligned/figures/figure_7_15_savings_vs_naive.png` |
| Figure 7.16, weekly cost (app KPI cards) | `artefacts/chapter7/app_aligned/figures/figure_7_16_weekly_cost.png` |
| Figure 7.17, supply vs staff saving decomposition | `artefacts/chapter7/app_aligned/figures/figure_7_17_supply_vs_staff_saving.png` |
| Figure 7.18, 6-component cost decomposition | `artefacts/chapter7/app_aligned/figures/figure_7_18_cost_decomposition.png` |
| Reproducer | `scripts/chapter7_app_aligned_experiment.py` |
| Renderer | `scripts/chapter7_app_aligned_visualization.py` |
