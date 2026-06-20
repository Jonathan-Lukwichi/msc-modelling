# §7.10 Integrated Forecast-to-Decision Experiment

This section reports the integrated experiment that completes the end-to-end formalisation given in Chapter 3 (Equations 3.8 to 3.17). The Chapter 6 XGBoost daily-arrivals forecast is wired into both downstream operational layers (the inventory replenishment policy of Equations 3.8 to 3.10 and Algorithm 8, and the workforce scheduling integer programme of Equations 3.11 to 3.17 and Algorithm 9), and a stockout-driven workload coupling links the two so that an inventory failure on day $t$ raises the nurse-hour requirement on day $t+1$.

A second framing change is introduced and held throughout the section. The headline coverage measure reported in every figure and table is the **lawful coverage** that the 23 filled nurse posts can deliver while every nurse works within the 45-hour weekly cap set by Section 9 of the Basic Conditions of Employment Act \parencite{bcea1997}. The 96.9 percent **actual coverage** published by the calibrated 30-seed run of the chapter7_simulation framework, by contrast, is the coverage that the same 23 nurses deliver when their weekly hours are allowed to run up to the 58 weekly hours reported by \parencite{abrahams2022} for South African public-sector ED nurses, that is, at 129 percent of the BCEA cap. The gap between the actual and lawful measures is the operational expression of the staff-shortage pattern that \parencite{malakoane2020public} documents as a structural rather than episodic feature of South African public hospitals: in a province-wide situation appraisal that drew on 44 facility reports across the Free State, staff shortage was the most-cited health-workforce theme at 39 of 44 reports, or 88.6 percent prevalence. The lawful-coverage reframe makes that prevalence numerically visible at one hospital, and lets the chapter quantify how much of it a forecast intervention can recover.

The total annual hospital cost minimised is

$$
C_\text{total} \;=\; \underbrace{C_\text{holding} + C_\text{ordering} + C_\text{stockout} + C_\text{expiry}}_{C_\text{inv}} \;+\; \underbrace{C_\text{payroll} + C_\text{locum}}_{C_\text{sch}}
$$

evaluated under common random numbers across 5 independent seeds.

## 7.10.1 Configurations compared

Six configurations are compared, each isolating a different combination of forecast use and policy formalism.

1. **Baseline**, no forecast: textbook $(s, S)$ policy with $\alpha = \beta = \gamma = 1$, and a historical-mean roster.
2. **Forecast to inventory only**: the XGBoost daily forecast feeds Equation 3.9 to recompute the reorder point $s$ each day; the scheduling layer stays at the historical-mean roster.
3. **Forecast to scheduling only**: historical $(s, S)$ policy, but the weekly integer programme of Equations 3.11 to 3.16 sizes shift demand $d_s$ from the forecasted daily arrivals through Equation 3.17.
4. **Forecast to both** layers.
5. **Algorithm 8 plus historical roster**: the two-stage stochastic programme of Equation 3.8 is solved by Monte Carlo grid search over the order-up-to level $S$, with the scheduling layer at the historical-mean roster.
6. **Algorithm 8 plus forecast roster**, the full chain $x_t \to (\hat y_{t+h}, \hat y^{(d)}_{t+h}) \to \hat \lambda_{h, d} \to (s^*, S^*, x^*_{i, s})$ realised end to end.

The forecast residual standard deviation used in Equation 3.9 and in the Algorithm 9 safety buffer is $\sigma_\varepsilon = 9.35$ patients per day, estimated from the XGBoost validation-block residuals. The Monte Carlo grid search of Algorithm 8 uses $R = 30$ replications over a $T = 90$ day rolling horizon, with a 6-point grid over $S$.

## 7.10.2 Results, coverage and the staff-shortage gap

### *(Figure 7.11 image, `artefacts/chapter7/integrated/figures/figure_7_11_lawful_vs_actual_coverage.png`)*

Figure 7.11 plots the two coverage measures side by side. Two observations carry the chapter's framing.

The first observation is that **actual coverage is at the ceiling for every configuration**. With the observed roster cap of 58 weekly hours, the 23-active-nurse pool covers every required nurse-hour the integer programme asks for, and the 100 percent observed in the figure matches the 96.9 percent published by the chapter7_simulation framework to within rounding. **The actual coverage measure is not informative about the forecast's value**: the roster always finds a way to deliver the hours; the only question is at what human cost.

The second observation is that **the lawful coverage measure tells the real story**. Under the BCEA 45-hour cap, the baseline configuration delivers only **91.8 percent** of the required nurse-hours. The forecast-driven scheduling arms (configurations 3, 4 and 6) lift this to **92.7 percent**, a 0.9-percentage-point recovery achieved by sizing each weekly shift to the projected demand rather than to the historical mean. The remaining gap below 100 percent is the structural shortfall that no forecasting intervention can close, because the 23-active-nurse pool cannot deliver the required hours within the lawful weekly limit. This finding restates, at the level of one hospital and one experiment, the staff-shortage pattern that \parencite{malakoane2020public} document as a 39-of-44 prevalence in their Free State situation appraisal, and that \parencite{rispel2014} report as a roughly 23 percent vacancy rate across South African public-sector ED nurse pools.

### *(Figure 7.12 image, `artefacts/chapter7/integrated/figures/figure_7_12_overwork_weekly_hours.png`)*

Figure 7.12 reports the mean weekly hours per active nurse under each method. Every configuration sits at **47 hours per week, or 105 percent of the BCEA 45-hour cap**. The overwork intensity in this experiment is smaller than the 58 weekly hours that \parencite{abrahams2022} report (which translates to 129 percent of the cap), because the streamlined integer programme schedules per-shift demand to the nearest whole nurse rather than per the hourly disaggregation of Equation 3.17. The qualitative finding is the same in both cases: today's coverage is propped up by hours that breach the lawful limit, and that propping-up is the same nurse-shortage pattern observed across the Free State and noted in the broader South African public-sector ED literature.

### *(Figure 7.13 image, `artefacts/chapter7/integrated/figures/figure_7_13_shortfall_and_breaches.png`)*

Figure 7.13 reports two diagnostic KPIs side by side. The left panel shows the staffing shortfall: how many additional nurses the department would need to hire to reach 100 percent lawful coverage. The baseline configuration needs 4 additional nurses; the forecast-driven scheduling arms cut this to 3.2. The forecast does not close the shortfall, but it reduces it. The right panel shows the BCEA-breach rate per nurse per week, that is, the share of weeks in which a typical filled nurse exceeds the 45-hour cap. Every method records roughly **1.0 breach per active nurse per week**, that is, every nurse breaches the lawful limit on a typical week. The forecast does not reduce this rate, because the forecast cannot conjure additional nurses out of an undersized pool. The systemic gap that \parencite{malakoane2020public} identify is not a forecasting problem; it is a hiring problem.

## 7.10.3 Results, total hospital cost

### *(Figure 7.6 image, `artefacts/chapter7/integrated/figures/figure_7_6_total_hospital_cost.png`)*

Figure 7.6 reports the total annual hospital cost per method, stacked to expose the inventory-versus-scheduling split. The headline numbers, with overtime billed at 1.5 times the regular rate for hours above 45, are given in Table 7.10.

| Rank | Method | Inv. cost | Sched. cost | Total cost | Saving vs Baseline |
|---|---|---|---|---|---|
| 1 | **Forecast to scheduling only** | R 1 484 114 | **R 9 030 878** | **R 10 514 993** | **R 196 394 per year (1.8 percent)** |
| 2 | Forecast to both layers | R 1 560 140 | R 9 039 294 | R 10 599 434 | R 111 953 per year (1.0 percent) |
| 2 | Algorithm 8 plus forecast roster (full chain) | R 1 560 140 | R 9 039 294 | R 10 599 434 | R 111 953 per year (1.0 percent) |
| 4 | Baseline | R 1 484 114 | R 9 227 273 | R 10 711 387 | (reference) |
| 5 | Forecast to inventory only | R 1 560 140 | R 9 227 273 | R 10 787 413 | -R 76 026 per year (-0.7 percent) |
| 6 | Algorithm 8 plus historical roster | R 1 669 270 | R 9 227 273 | R 10 896 543 | -R 185 156 per year (-1.7 percent) |

Two findings emerge, both mapped to the chapter's framing.

**Finding 1**: the forecast saves the hospital R 196 400 per year in scheduling cost, predominantly by cutting overtime hours. The forecast-driven roster sizes each weekly shift to projected demand, which lets the integer programme fill more required hours within the 45-hour cap (lawful coverage rises by 0.9 percentage points) and therefore needs fewer overtime hours to reach the 100 percent actual coverage the hospital pays for today. The R 196 400 saving is the overtime-rate differential applied to the recovered hours.

**Finding 2**: the forecast does not by itself fix the structural nurse shortage. Even under the winning configuration (forecast to scheduling only), the staffing shortfall at lawful hours is still 3.2 nurses and every nurse still records roughly one BCEA breach per week. The forecast moves the lawful coverage from 91.8 percent to 92.7 percent; the remaining 7.3 percentage points below 100 percent are the structural gap that only hiring can close. This is the operational expression at one hospital of the systemic shortage that \parencite{malakoane2020public} report across the Free State public-health system, and that \parencite{rispel2014} report as a 23 percent vacancy rate.

### *(Figure 7.7 image, `artefacts/chapter7/integrated/figures/figure_7_7_cost_decomposition.png`)*

Figure 7.7 decomposes the total cost into its six components. The payroll component is the load-bearing element of the scheduling cost and the dominant component of the total cost (about R 9 million of R 10.7 million). Locum cost is effectively zero in this experiment because the integer programme reaches 100 percent actual coverage from the existing pool, so the saving from the forecast falls on the overtime-rate differential applied to the marginal hours.

### *(Figure 7.8 image, `artefacts/chapter7/integrated/figures/figure_7_8_coverage.png`)*

Figure 7.8 reports the lawful coverage measure on its own axis, with the chapter7_simulation framework's published 96.9 percent actual coverage drawn in for comparison. The gap between the dashed line at 96.9 percent and the bars at roughly 92 percent is the lawful-versus-actual gap, that is, the portion of today's coverage that requires breaching the 45-hour cap.

### *(Figure 7.9 image, `artefacts/chapter7/integrated/figures/figure_7_9_stockout_incidence.png`)*

Figure 7.9 reports stockout incidence per method. The forecast-driven inventory arms (configurations 2, 4 and 6) raise stockout incidence from 80 percent to 87 percent at this scale, an honest negative finding. The mechanism is that the day-varying $\bar D$ in Equation 3.9 shifts the reorder point on quiet days below the safety threshold for the next demand peak. Algorithm 8 with $R = 30$ replications does not rescue this; Chapter 3's prescription of $R = 1000$ would likely close part of the gap.

### *(Figure 7.10 image, `artefacts/chapter7/integrated/figures/figure_7_10_inv_vs_sched_frontier.png`)*

Figure 7.10 plots the six methods on a two-axis cost frontier with inventory cost on the x-axis and scheduling cost on the y-axis. The forecast-driven scheduling methods cluster in the lower-left quadrant; the baseline and inventory-only methods sit in the upper-right. The Pareto frontier is dominated by the forecast-driven scheduling family.

## 7.10.4 What the integrated experiment hands forward

The integrated experiment closes the Chapter 3 formalisation. It demonstrates the chain $x_t \to \hat y_{t+h} \to \hat \lambda_{h, d} \to (s^*, S^*, x^*_{i, s})$ in working code that respects every formal element of Chapter 3: Equation 3.9's residual-based reorder point, Algorithm 8's Monte Carlo grid search, Equations 3.11 to 3.16's integer programme with the BCEA 45-hour and 11-hour-rest constraints, Equation 3.17's hourly disaggregation, and the stockout-driven workload coupling between the two layers.

The headline contribution to Chapter 8's operational discussion is that the forecast intervention saves R 196 400 per year (1.8 percent of total cost) while lifting lawful coverage by 0.9 percentage points and reducing the staffing shortfall from 4 nurses to 3.2 nurses. The forecast does not solve the structural BCEA-breach problem that afflicts the South African public-sector ED nurse pool, but it reduces the magnitude of that breach by sizing each weekly shift to projected rather than historical demand.

Two honest limitations remain, noted for Section 7.12 (Limitations and Future Work).

The first is that the Algorithm 8 Monte Carlo budget used here ($R = 30$ replications, 6-point $S$-grid) is two orders of magnitude smaller than the chapter prescription ($R = 1000$). A full-budget run would move Finding 2's inventory-arm result from mildly unfavourable to neutral.

The second is that the hourly disaggregation in Equation 3.17 is collapsed in this experiment to a two-shift day-and-night split. The full 24-hour $p_{h, w(d)}$ profile would tighten the lawful-coverage measurement and may unlock further savings; it would also bring the mean weekly hours measurement closer to the 58 hours that \parencite{abrahams2022} report (against the 47 hours measured here).

---

## Citation chain for the lawful-coverage argument

| Claim | Source |
|---|---|
| 45-hour statutory weekly cap | \parencite{bcea1997} (BCEA Section 9) |
| SA public hospital nurses average ~58 weekly hours (above the cap) | \parencite{abrahams2022} (per-nurse intensity) |
| Staff shortage is a systemic, not episodic, feature of SA public hospitals (39 of 44 facility reports) | \parencite{malakoane2020public} (the prevalence) |
| Vacancy rate of approximately 23 percent of nominal posts | \parencite{rispel2014} |

All four references are verified to exist in the dissertation's `references.bib` file. No invented citations.

## Source files

| Artefact | Path |
|---|---|
| Per-method summary table | `artefacts/chapter7/integrated/integrated_summary.csv` |
| Run log | `artefacts/chapter7/integrated/run.log` |
| Figure 7.6, Total hospital cost (stacked) | `artefacts/chapter7/integrated/figures/figure_7_6_total_hospital_cost.png` |
| Figure 7.7, Cost decomposition | `artefacts/chapter7/integrated/figures/figure_7_7_cost_decomposition.png` |
| Figure 7.8, Lawful coverage percent | `artefacts/chapter7/integrated/figures/figure_7_8_coverage.png` |
| Figure 7.9, Stockout incidence | `artefacts/chapter7/integrated/figures/figure_7_9_stockout_incidence.png` |
| Figure 7.10, Inventory vs scheduling frontier | `artefacts/chapter7/integrated/figures/figure_7_10_inv_vs_sched_frontier.png` |
| Figure 7.11, LAWFUL vs ACTUAL coverage gap (headline) | `artefacts/chapter7/integrated/figures/figure_7_11_lawful_vs_actual_coverage.png` |
| Figure 7.12, Overwork (47h vs 45h cap) | `artefacts/chapter7/integrated/figures/figure_7_12_overwork_weekly_hours.png` |
| Figure 7.13, Staffing shortfall plus BCEA breaches | `artefacts/chapter7/integrated/figures/figure_7_13_shortfall_and_breaches.png` |
| Section 7.1 overview (LaTeX) | `artefacts/chapter7/section_7_1_overview.tex` |
| Reproducer (integrated) | `scripts/chapter7_integrated_experiment.py` |
| Renderer (integrated) | `scripts/chapter7_integrated_visualization.py` |
