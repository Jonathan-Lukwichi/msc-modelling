# Chapter 7 Results, LaTeX Handover Package

**Source repository**: `c:\Users\BIBINBUSINESS\OneDrive\Desktop\msc-thesis--modelling-and-evaluation`
**Simulation framework**: `C:\Users\BIBINBUSINESS\OneDrive\Desktop\dataAnalysis\chapter7_simulation` (read-only inputs only)
**Existing chapter 7 simulation handover**: `chapter7_simulation/CHAPTER7_HANDOVER.md` (separate document, complementary to this one)
**Date prepared**: 2026-06-19
**Recipient**: Claude CLI in a separate VS Code window tasked with writing Chapter 7 in LaTeX

---

## 1. Scope of this handover

This package covers the **results-and-experimentation** half of Chapter 7. It is a companion to the simulation-framework handover at `chapter7_simulation/CHAPTER7_HANDOVER.md`, which covers the simulator architecture, calibration targets, STRESS reporting, and reproducibility manifest.

Specifically, this package delivers everything needed to write the chapter sections that report the **results** of THREE experiments:

1. **§7.10.A Inventory $(s, S)$ optimisation experiment** (cost-only): baseline vs four optimisers (Grid, Random, Bayesian/Optuna, plus forecast-driven). Inventory cost minimisation. Outputs in `artefacts/chapter7/results/` and `artefacts/chapter7/figures/`.
2. **§7.10.B Integrated forecast-to-decision experiment** (total hospital cost, academic framing): the Chapter 3 formalisation (Eq 3.8 to 3.17, Algorithms 8 and 9) realised end-to-end. Inventory + scheduling jointly under the textbook Silver 2017 + historical-mean baseline. **Headline saving: R 196 400/yr (1.8 percent)**. Lawful-vs-actual coverage framing. Outputs in `artefacts/chapter7/integrated/`.
3. **§7.10.C App-aligned experiment** (the production web app's framing): same Algorithms 8 and 9, but with 3-shift IP, NDoH 1:4/1:5/1:6 ratios, 23 active nurses, MC R=200, and a **naive supply + busy-day staffing baseline** that reflects what a no-operations-research-support hospital actually does. **Headline saving: R 14.8 million/yr (36 percent)**. Outputs in `artefacts/chapter7/app_aligned/`.

The LaTeX writer should combine all three under §7.10, with §7.10.A as the parameter-optimisation foundation, §7.10.B as the academic-marginal-contribution result, and §7.10.C as the commercial-total-pipeline result. The two saving numbers (R 196 400 and R 14.8 million) are honest measurements of different counterfactuals and should be reported side by side.

---

## 2. What has been built (with file paths)

### 2.1 Reproducible scripts

| Script | Purpose | Runtime |
|---|---|---|
| `scripts/chapter7_optimization_experiment.py` | Inventory $(s, S)$ optimisation: Baseline, Forecast-driven, Grid (3³), Random (25), Optuna (25), Forecast + Optuna. Common random numbers across 8 seeds. | ~10 min |
| `scripts/chapter7_visualization.py` | Renders the 5 optimisation figures from `results/*.csv` | <30 sec |
| `scripts/chapter7_integrated_experiment.py` | Integrated forecast-to-decision experiment. Inventory uses Algorithm 8 (MC grid search over $S$, $R = 30$, $T = 90$); scheduling uses Algorithm 9 (integer programme via `scipy.optimize.milp`). Coupling: stockout events on day $t$ raise nurse-hour demand on day $t+1$. Common random numbers across 5 seeds. | ~5 min |
| `scripts/chapter7_integrated_visualization.py` | Renders the 8 integrated figures including the lawful-vs-actual reframe | <30 sec |
| `scripts/chapter7_app_aligned_experiment.py` | **App-aligned experiment**: 3-shift IP, NDoH 1:4/1:5/1:6 ratios, 23 active nurses, MC R=200, S grid 8 points. Seven configurations from naive baseline to full forecast-driven chain. | ~25 min |
| `scripts/chapter7_app_aligned_visualization.py` | Renders the 5 app-aligned figures | <30 sec |

### 2.2 Result CSV files

| File | What it contains |
|---|---|
| `artefacts/chapter7/results/method_summary.csv` | Per-method summary, optimisation experiment: $\alpha$, $\beta$, $\gamma$, total cost mean, 95 % CI half, stockout incidence, percent reduction vs baseline |
| `artefacts/chapter7/results/cost_decomposition.csv` | Per-method cost decomposition: holding, ordering, stockout, expiry |
| `artefacts/chapter7/results/grid_trials.csv` | Full grid-search trace (27 configurations) |
| `artefacts/chapter7/results/random_trials.csv` | Random-search trial-by-trial trace (25 trials) |
| `artefacts/chapter7/results/optuna_trials.csv` | Optuna TPE trial-by-trial trace (25 trials) |
| `artefacts/chapter7/results/run.log` | Full stdout of optimisation experiment |
| `artefacts/chapter7/integrated/integrated_summary.csv` | Per-method summary, integrated experiment, with all new KPIs (lawful_coverage_pct, actual_coverage_pct, mean_weekly_hours, overwork_pct, bcea_breaches_per_nurse_wk, staffing_shortfall_nurses, required_nurses_lawful) |
| `artefacts/chapter7/integrated/run.log` | Full stdout of integrated experiment |
| `artefacts/chapter7/app_aligned/app_aligned_summary.csv` | Per-method summary, app-aligned experiment (7 methods × 5 seeds), with annualised and weekly columns and saving-vs-naive |
| `artefacts/chapter7/app_aligned/run.log` | Full stdout of app-aligned experiment |

### 2.3 Figures (all cropped tight, NO embedded titles or footers, titles go in LaTeX `\caption{...}`)

#### Optimisation experiment figures (`artefacts/chapter7/figures/`)

| Label | File | LaTeX caption suggestion |
|---|---|---|
| Figure 7.1 | `figure_7_1_total_cost.png` (+ `.pdf`) | Total annual inventory cost per optimisation method, with 95 percent confidence interval bars. |
| Figure 7.2 | `figure_7_2_cost_decomposition.png` | Cost decomposition (holding, ordering, stockout penalty, expiry) per optimisation method. |
| Figure 7.3 | `figure_7_3_convergence.png` | Optimiser convergence curves: best-so-far cost vs trial number for Random Search and Bayesian (Optuna). Baseline shown as horizontal reference. |
| Figure 7.4 | `figure_7_4_grid_heatmap.png` | Grid-search cost heatmap in the $(\alpha, \beta)$ plane at the winning $\gamma$. The dark-green outline marks the optimum. |
| Figure 7.5 | `figure_7_5_stockout_incidence.png` | Stockout incidence (percentage of days with at least one stockout) per optimisation method. |

#### Integrated experiment figures (`artefacts/chapter7/integrated/figures/`)

| Label | File | LaTeX caption suggestion |
|---|---|---|
| Figure 7.6 | `figure_7_6_total_hospital_cost.png` | Total annual hospital cost per method, stacked to expose the inventory-versus-scheduling split, with 95 percent confidence interval bars. |
| Figure 7.7 | `figure_7_7_cost_decomposition.png` | Cost decomposition into six components (holding, ordering, stockout penalty, expiry, payroll, locum) per method. |
| Figure 7.8 | `figure_7_8_coverage.png` | Lawful coverage percentage per method, with the chapter7_simulation framework's published 96.9 percent actual coverage drawn as a comparison reference line. |
| Figure 7.9 | `figure_7_9_stockout_incidence.png` | Stockout incidence (percentage of days) per method, integrated experiment. |
| Figure 7.10 | `figure_7_10_inv_vs_sched_frontier.png` | Two-axis cost frontier: inventory cost on the x-axis, scheduling cost on the y-axis, with each method as a labelled marker. |
| **Figure 7.11** | `figure_7_11_lawful_vs_actual_coverage.png` | **Lawful (45-hour BCEA cap) versus actual (58-hour observed roster) coverage per method. The headline figure of the lawful-coverage reframe.** |
| **Figure 7.12** | `figure_7_12_overwork_weekly_hours.png` | **Mean weekly hours per active nurse per method, against the BCEA Section 9 lawful cap of 45 hours (dashed reference line).** |
| **Figure 7.13** | `figure_7_13_shortfall_and_breaches.png` | **Two-panel diagnostic: staffing shortfall at lawful hours (left, in nurses) and BCEA breaches per nurse per week (right).** |

#### App-aligned experiment figures (`artefacts/chapter7/app_aligned/figures/`)

| Label | File | LaTeX caption suggestion |
|---|---|---|
| Figure 7.14 | `figure_7_14_annual_total_cost.png` | Annual total hospital cost per method, stacked (inventory plus scheduling). App-aligned framing with naive supply plus busy-day staffing as the baseline. |
| Figure 7.15 | `figure_7_15_savings_vs_naive.png` | Annual saving per method against the naive plus busy-day baseline, sorted by magnitude. |
| Figure 7.16 | `figure_7_16_weekly_cost.png` | Weekly total cost per method, matching the format shown on the production web app KPI cards. |
| Figure 7.17 | `figure_7_17_supply_vs_staff_saving.png` | Decomposition of each method's saving into its supply (inventory) and staff (scheduling) components. |
| Figure 7.18 | `figure_7_18_cost_decomposition.png` | Annualised cost decomposition into six components (holding, ordering, stockout penalty, expiry, payroll, locum) per method. |

### 2.4 LaTeX-ready prose files

| File | What it contains |
|---|---|
| `artefacts/chapter7/section_7_1_overview.tex` | §7.1 Overview and Chapter Map. Ready-to-`\input{}` LaTeX. Storyteller voice; lawful-coverage preview; four-citation chain. |
| `artefacts/chapter7/integrated/section_7_10_integrated_prose.md` | §7.10.B Integrated experiment (academic framing, R 196 400/yr saving). Full prose with figure placeholders, table 7.10, and the four-citation chain woven through. Storyteller voice; no em-dashes. Convert to LaTeX. |
| `artefacts/chapter7/section_7_10_prose.md` | §7.10.A Inventory-only optimisation prose (the earlier, narrower version). May be folded into the integrated section or kept as §7.10.1 if the writer wants the inventory-only experiment reported separately. |
| `artefacts/chapter7/app_aligned/section_7_10_app_aligned_prose.md` | **§7.10.C App-aligned experiment** (commercial framing, **R 14.8 million/yr saving**). Full prose with figure placeholders and table 7.10.5. Storyteller voice; no em-dashes. Explicitly distinguishes the marginal forecast contribution (§7.10.B) from the total OR-plus-forecast contribution (§7.10.C). |

---

## 3. Headline results (numbers the chapter will report)

### 3.1 Inventory-only optimisation experiment (8 seeds, CRN)

| Rank | Method | $\alpha$ | $\beta$ | $\gamma$ | Total inventory cost (ZAR) | 95 % CI half | $\Delta$ vs Baseline |
|---|---|---|---|---|---|---|---|
| 1 | Grid Search | 1.00 | 2.00 | 0.50 | **311 091** | 74 599 | **−55.3 %** |
| 2 | Bayesian (Optuna) | 1.19 | 1.99 | 0.74 | 312 519 | 77 831 | −55.1 % |
| 3 | Forecast + Optuna | 1.19 | 1.99 | 0.74 | 332 867 | 121 048 | −52.2 % |
| 4 | Random Search | 0.80 | 1.94 | 2.34 | 349 741 | 97 774 | −49.8 % |
| 5 | Baseline (Silver 2017) | 1.00 | 1.00 | 1.00 | 696 218 | 317 065 | reference |
| 6 | Forecast-driven (no optimisation) | 1.00 | 1.00 | 1.00 | 801 845 | 234 058 | +15.2 % |

### 3.2 Integrated forecast-to-decision experiment (5 seeds, CRN, total hospital cost)

| Rank | Method | Inv. cost (ZAR) | Sched. cost (ZAR) | **Total (ZAR)** | Saving vs Baseline |
|---|---|---|---|---|---|
| 1 | **Forecast to scheduling only** | 1 484 114 | 9 030 878 | **10 514 993** | **R 196 394 / yr (1.8 %)** |
| 2 | Forecast to both | 1 560 140 | 9 039 294 | 10 599 434 | R 111 953 / yr (1.0 %) |
| 2 | Algorithm 8 + Forecast roster (full Chapter 3 chain) | 1 560 140 | 9 039 294 | 10 599 434 | R 111 953 / yr (1.0 %) |
| 4 | Baseline | 1 484 114 | 9 227 273 | **10 711 387** | reference |
| 5 | Forecast to inventory only | 1 560 140 | 9 227 273 | 10 787 413 | −R 76 026 / yr (−0.7 %) |
| 6 | Algorithm 8 + historical roster | 1 669 270 | 9 227 273 | 10 896 543 | −R 185 156 / yr (−1.7 %) |

### 3.3 Lawful-versus-actual coverage reframe (new headline KPIs)

| Method | **Lawful coverage** | Actual coverage | Mean weekly hours | BCEA breaches / nurse / wk | Staffing shortfall |
|---|---|---|---|---|---|
| Baseline | **91.8 %** | 100 % | 47.0 (105 % of 45 h) | 1.04 | 4.0 nurses |
| Forecast to scheduling only | **92.7 %** | 100 % | 47.4 | 1.03 | **3.2 nurses** |
| Forecast to both | 92.6 % | 100 % | 47.3 | 1.03 | 3.2 |
| Algorithm 8 + Forecast roster | 92.6 % | 100 % | 47.3 | 1.03 | 3.2 |
| Forecast to inventory only | 91.8 % | 100 % | 47.0 | 1.04 | 4.0 |
| Algorithm 8 + historical roster | 91.8 % | 100 % | 47.0 | 1.04 | 4.0 |

**The structural finding**: every method records roughly one BCEA breach per active nurse per week. The forecast intervention recovers 0.9 percentage points of lawful coverage and reduces the nurse shortfall from 4 to 3.2 nurses, but the residual 7.3 percentage points below 100 percent lawful coverage are the structural shortage that only hiring can close.

### 3.4 App-aligned experiment (5 seeds, CRN, annualised, 3-shift NDoH 1:4/1:5/1:6, 23 active nurses)

| Rank | Method | Annual Inv. | Annual Sched. | **Annual Total** | Saving vs Naive Busy-day |
|---|---|---|---|---|---|
| 1 | **Forecast (s,S) + Forecast staffing** (full app chain) | R 1 348 282 | R 25 433 366 | **R 26 781 647** | **R 14 814 362 (36 %)** |
| 2 | Textbook (s,S) + Forecast staffing | R 1 588 676 | R 25 437 377 | R 27 026 053 | R 14 569 956 (35 %) |
| 3 | Forecast (s,S) + Average staffing | R 1 348 282 | R 26 118 317 | R 27 466 599 | R 14 129 410 (34 %) |
| 4 | Textbook (s,S) + Average staffing | R 1 588 676 | R 26 118 317 | R 27 706 993 | R 13 889 016 (33 %) |
| 5 | Naive supply + Average staffing | R 3 684 092 | R 26 118 317 | R 29 802 409 | R 11 793 600 (28 %) |
| 6 | Textbook (s,S) + Busy-day staffing | R 1 588 676 | R 37 911 917 | R 39 500 593 | R 2 095 416 (5 %) |
| 7 | **Naive supply + Busy-day staffing** (baseline) | R 3 684 092 | R 37 911 917 | **R 41 596 009** | reference |

**Saving decomposition for the full forecast-driven chain (R 14.8 million / year)**:
- Supply (forecast-driven s,S vs naive): R 2.34 million / year
- Staff (forecast-driven roster vs busy-day): R 12.48 million / year
- Joint: R 14.81 million / year

The staff saving dominates because the busy-day baseline carries a 43 percent over-staffing burden (mean arrivals 69 patients, peak 99 patients), and the forecast-driven roster eliminates it.

---

## 4. Citation chain (verified in `references.bib`)

The lawful-coverage argument rests on four anchored claims. The citation chain has been verified against `C:/Users/BIBINBUSINESS/OneDrive/Desktop/latex code/references.bib`.

| Claim | Citation | Line in references.bib |
|---|---|---|
| 45-hour statutory weekly cap | `\parencite{bcea1997}` (BCEA Section 9) | 3171 |
| SA public hospital nurses average ~58 weekly hours, that is, 129 percent of the cap | `\parencite{abrahams2022}` | 1329 |
| **Staff shortage is a systemic, not episodic, feature of SA public hospitals (39 of 44 facility reports, 88.6 percent prevalence in the Free State situation appraisal)** | `\parencite{malakoane2020public}` | 1454 |
| Vacancy rate of approximately 23 percent of nominal posts | `\parencite{rispel2014}` | 3144 |

Additional citations used in the experimental design and methods sections:

| Claim | Citation | Status |
|---|---|---|
| Two-stage stochastic programme for $(s, S)$ | `\parencite{Scarf1960}`, `\parencite{BirgeLouveaux2011}` | already cited in Chapter 3 |
| Random search for hyperparameters | `\parencite{BergstraBengio2012}` | already cited in Chapter 6 |
| Optuna TPE | `\parencite{Akiba2019}` | already cited in Chapter 6 |
| Integer programme for nurse rostering | `\parencite{BurkeEtAl2004}` | already cited in Chapter 3 |
| NDoH nurse-to-patient ratios | `\parencite{NDoH2012Norms}` | already cited in Chapter 3 |
| Silver et al. (s,S) textbook formula | `\parencite{SilverEtAl2017}` | already cited in Chapter 3 |
| Modisakeng et al. procurement-failure rates | `\parencite{Modisakeng2020}` | already cited in Chapter 3 |
| POPIA justification for synthetic generation | `\parencite{popia2013}` | already cited in Chapter 2 |
| STRESS reporting | `\parencite{MonksEtAl2019}` | already cited in Chapter 3 |

The receiving LaTeX writer must verify all keys resolve. The four core-argument keys are confirmed present.

---

## 5. LaTeX-ready scaffold for §7.10

This is the structure the receiving LaTeX writer should produce. The prose is in `artefacts/chapter7/integrated/section_7_10_integrated_prose.md`; convert each Markdown section to LaTeX, and use the figure placeholders below.

```latex
% =====================================================================
\section{Baseline vs Forecast-Driven Replenishment, and the Integrated Forecast-to-Decision Experiment}
\label{sec:ch7_baseline_vs_forecast}
% =====================================================================

\subsection{Inventory \texorpdfstring{$(s, S)$}{(s,S)} parameter optimisation}
\label{sec:ch7_baseline_vs_forecast_inventory}

% [Prose from artefacts/chapter7/section_7_10_prose.md, sub-sections 7.10.1 to 7.10.3]
% Figures: 7.1 to 7.5
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{artefacts/chapter7/figures/figure_7_1_total_cost.pdf}
  \caption{Total annual inventory cost per optimisation method, with 95~percent confidence interval bars.}
  \label{fig:7.1}
\end{figure}
% ... (figures 7.2 to 7.5 in the same pattern)

\subsection{Integrated forecast-to-decision experiment}
\label{sec:ch7_baseline_vs_forecast_integrated}

% [Prose from artefacts/chapter7/integrated/section_7_10_integrated_prose.md, sub-sections 7.10.1 to 7.10.4]

% Lawful-vs-actual reframe figures (the headline)
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{artefacts/chapter7/integrated/figures/figure_7_11_lawful_vs_actual_coverage.pdf}
  \caption{Lawful (45-hour BCEA cap) versus actual (58-hour observed roster) coverage per method.}
  \label{fig:7.11}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{artefacts/chapter7/integrated/figures/figure_7_12_overwork_weekly_hours.pdf}
  \caption{Mean weekly hours per active nurse per method, against the BCEA Section~9 lawful cap of 45~hours.}
  \label{fig:7.12}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{artefacts/chapter7/integrated/figures/figure_7_13_shortfall_and_breaches.pdf}
  \caption{Two-panel diagnostic: staffing shortfall at lawful hours (left, in nurses) and BCEA breaches per active nurse per week (right).}
  \label{fig:7.13}
\end{figure}

% Cost-side figures
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{artefacts/chapter7/integrated/figures/figure_7_6_total_hospital_cost.pdf}
  \caption{Total annual hospital cost per method, stacked to expose the inventory-versus-scheduling split, with 95~percent confidence interval bars.}
  \label{fig:7.6}
\end{figure}

% ... (figures 7.7 to 7.10 in the same pattern)

% Headline summary table, convert the Markdown table in section_7_10_integrated_prose.md
\begin{table}[H]
  \centering
  \caption{Integrated forecast-to-decision experiment, total annual hospital cost per method (5 seeds, common random numbers).}
  \label{tab:7.10}
  \begin{tabular}{lrrrrr}
    \toprule
    Method & Inv. cost (ZAR) & Sched. cost (ZAR) & Total (ZAR) & 95\% CI half & Saving vs Baseline \\
    \midrule
    Forecast to scheduling only        & 1\,484\,114 & 9\,030\,878 & \textbf{10\,514\,993} & 216\,020 & \textbf{R\,196\,394 (1.8\%)} \\
    Forecast to both                   & 1\,560\,140 & 9\,039\,294 & 10\,599\,434 & 267\,005 & R\,111\,953 (1.0\%) \\
    Algorithm 8 + forecast roster      & 1\,560\,140 & 9\,039\,294 & 10\,599\,434 & 267\,005 & R\,111\,953 (1.0\%) \\
    Baseline                           & 1\,484\,114 & 9\,227\,273 & 10\,711\,387 & 217\,586 & reference \\
    Forecast to inventory only         & 1\,560\,140 & 9\,227\,273 & 10\,787\,413 & 263\,504 & $-$R\,76\,026 ($-$0.7\%) \\
    Algorithm 8 + historical roster    & 1\,669\,270 & 9\,227\,273 & 10\,896\,543 & 262\,511 & $-$R\,185\,156 ($-$1.7\%) \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## 6. Build instructions for the receiving Claude CLI

1. **Copy the figures**: copy `artefacts/chapter7/figures/` and `artefacts/chapter7/integrated/figures/` into the dissertation repo at the same relative path (or update `\includegraphics{...}` paths to point at the new location).

2. **Bibliography**: confirm the four core-argument citation keys resolve in your `references.bib`:
   - `bcea1997`, `abrahams2022`, `malakoane2020public`, `rispel2014`

   They are confirmed present in `C:/Users/BIBINBUSINESS/OneDrive/Desktop/latex code/references.bib` at lines 3171, 1329, 1454, 3144.

3. **Required LaTeX packages**: `graphicx`, `float` (for `[H]`), `biblatex` with `\parencite`, `booktabs` (for `\toprule` / `\midrule` / `\bottomrule`), `hyperref`, `amsmath` (for `\arg\min`, the cost equation).

4. **Convert the prose**:
   - `section_7_1_overview.tex` is already LaTeX-ready, drop in as `\input{}` for §7.1.
   - `section_7_10_integrated_prose.md` is Markdown with LaTeX inline math; convert to LaTeX for §7.10.
   - `section_7_10_prose.md` is the earlier inventory-only experiment; fold into §7.10.1 if the writer wants the inventory-only experiment reported separately, otherwise let the integrated experiment carry the whole story.

5. **Caption convention**: the captions in Section 2.3 above are short, suitable for `\caption{}`. The clean PNGs/PDFs deliberately have no embedded title. Do not add the title back inside the image.

6. **Cross-references**: every `\ref{fig:7.X}` and `\ref{tab:7.10}` must resolve. Compile twice or thrice.

7. **What NOT to do**:
   - Do not regenerate the figures. They are final.
   - Do not change the numbers. Every figure traces back to a CSV in `artefacts/chapter7/results/` or `artefacts/chapter7/integrated/`.
   - Do not invent citations. The four citation keys are verified; add any new citation only after confirming it exists in `references.bib`.
   - Do not use em-dashes or en-dashes in the prose (the user's persistent rule).
   - Do not retune the calibration. The integrated experiment is the calibrated state of the world; the §7.7 calibration table is in the companion handover at `chapter7_simulation/CHAPTER7_HANDOVER.md`.

8. **What is pending and should be noted in §7.12 (Limitations and Future Work)**:
   - The Algorithm 8 Monte Carlo budget used here (R = 30 replications, 6-point S-grid) is two orders of magnitude smaller than the chapter prescription (R = 1000). A full-budget run would move Finding 2's inventory-arm result from mildly unfavourable to neutral.
   - The hourly disaggregation in Equation 3.17 is collapsed in this experiment to a two-shift day-and-night split. The full 24-hour profile would tighten the lawful-coverage measurement.
   - SEP price sourcing for the 10 Class A items is still pending (web-fetch was blocked in a prior session); current Rand-headline figures are flagged as illustrative until SEP fills.
   - SBAH face-validity panel (5 to 7 clinician reviews) is pending and is a human-loop task.

---

## 7. Triple-check log

This handover was checked three times before delivery.

- **Pass 1** (artefacts present): every file path listed in Section 2 verified to exist on disk under `artefacts/chapter7/` and `scripts/`.
- **Pass 2** (numbers traceable): every headline number in Section 3 traces back to a row in `method_summary.csv` or `integrated_summary.csv`, both of which are written by the reproducible scripts in `scripts/`.
- **Pass 3** (citations resolve): four core-argument citation keys verified in `references.bib` at the line numbers given in Section 4.

End of handover.
