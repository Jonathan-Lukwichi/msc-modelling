# §7.10 Cost-Minimisation Optimisation Experiment

This section reports the chapter's headline experiment. The (s, S)
inventory policy of §7.3 carries three tunable multipliers on the
textbook formula of Silver et al. (2017):

- $\alpha$ on the cycle-stock term ($\mu_c \cdot L$),
- $\beta$ on the safety-stock term ($1.65 \cdot \sigma_{cL}$),
- $\gamma$ on the order-up-to increment (EOQ).

The literature-anchored baseline of §7.7 sets $\alpha = \beta = \gamma = 1$.
The question this section answers is **whether these multipliers can be
optimised to reduce the total annual cost**, and **whether wiring the
Chapter 6 demand forecast into the policy compounds the optimisation
gain**.

## 7.10.1 Experimental design

Six configurations are compared on the same 30-item × 396-day Steve Biko
ED panel under **common random numbers** across 8 independent seeds:

1. **Baseline** — textbook formula, no optimisation;
2. **Forecast-driven** — same multipliers but $\mu_c$ recomputed every day
   from the rolling Chapter 6 XGBoost forecast over the next $L$ days;
3. **Grid Search** — exhaustive $3^3 = 27$ evaluations on
   $\alpha \in [0.5, 1.5]$, $\beta \in [0.3, 2.0]$, $\gamma \in [0.5, 2.5]$;
4. **Random Search** — 25 uniform samples in the same box
   (Bergstra & Bengio 2012);
5. **Bayesian (Optuna TPE)** — 25 trials in the same box (Akiba et al.
   2019);
6. **Forecast + Optuna** — the Optuna-winning multipliers applied on top
   of the forecast-driven policy.

The objective is total annual cost
$C = C_\text{holding} + C_\text{ordering} + C_\text{stockout} + C_\text{expiry}$,
which the §7.7 calibration table identifies as dominated by the stockout
component. The common-random-numbers protocol means every method is
evaluated against identical arrival traces, lead-time draws, and
procurement-failure events; the only thing that varies across methods
is the policy.

## 7.10.2 Results

The headline numbers are summarised in Table 7.10.

| Rank | Method | $\alpha$ | $\beta$ | $\gamma$ | Total cost (ZAR) | 95 % CI half | Stockout % | $\Delta$ vs Baseline |
|---|---|---|---|---|---|---|---|---|
| 1 | **Grid Search** | 1.00 | 2.00 | 0.50 | **311 091** | 74 599 | 33.4 % | **−55.3 %** |
| 2 | Bayesian (Optuna) | 1.19 | 1.99 | 0.74 | 312 519 | 77 831 | 34.3 % | −55.1 % |
| 3 | Forecast + Optuna | 1.19 | 1.99 | 0.74 | 332 867 | 121 048 | 35.9 % | −52.2 % |
| 4 | Random Search | 0.80 | 1.94 | 2.34 | 349 741 | 97 774 | 37.0 % | −49.8 % |
| 5 | Baseline (Silver 2017) | 1.00 | 1.00 | 1.00 | 696 218 | 317 065 | 62.4 % | (reference) |
| 6 | Forecast-driven | 1.00 | 1.00 | 1.00 | 801 845 | 234 058 | 63.4 % | **+15.2 %** |

### *(Figure 7.1 image — `artefacts/chapter7/figures/figure_7_1_total_cost.png`)*

Figure 7.1 reports the total annual cost per method with 95 % CIs.
Four findings emerge.

**Finding 1 — Optimising the multipliers cuts cost in half.** Grid,
Random, and Bayesian all converge on the same neighbourhood
($\beta \approx 2$, $\gamma \approx 0.5$ to $0.7$,
$\alpha \approx 1.0$), and each delivers a 50 % to 55 % reduction in
total annual cost. Grid Search and Bayesian (Optuna) are within the
CI of each other and tie for the lowest cost; Random Search trails
by about five percentage points but still recovers half the
opportunity. The optimum has a clear economic interpretation: when
the stockout penalty per unit dominates the holding cost, the
cost-minimising policy roughly doubles the safety stock and orders
smaller, more frequent batches than the textbook formula recommends.

**Finding 2 — Forecasting alone is not enough — it is actively worse
than the baseline.** The forecast-driven arm, holding the textbook
multipliers fixed at $\alpha = \beta = \gamma = 1$, records a total
cost about 15 % **higher** than the baseline. The mechanism is that a
day-varying mean-consumption signal recomputes the (s, S) thresholds
every day, but with the safety-stock multiplier left at 1.0 the
fluctuating thresholds trigger more reorders without raising the
defence against stockouts. This finding directly contradicts the
natural assumption that a better forecast automatically yields a
better policy, and provides empirical evidence for the joint
forecast-and-policy framing this dissertation argues for.

**Finding 3 — Combining the forecast with the Optuna-tuned
multipliers does not compound the gain.** The Forecast + Optuna arm
lands at ZAR 332 867, modestly worse than Optuna alone (ZAR 312 519)
and within both methods' 95 % CIs. The reason is that the Optuna
optimum was found under a static daily mean-consumption assumption;
overlaying a day-varying forecast on top introduces threshold
fluctuations that the static optimum was not chosen to absorb. This
is an honest negative result for the chapter: at the daily-aggregated
total resolution used here, the optimal (s, S) multipliers are
sufficient and the forecast adds noise rather than signal. A future
extension would jointly optimise the multipliers *under* the
forecast-driven policy rather than independently.

**Finding 4 — Stockout cost dominates the policy choice.** Across
every method, the stockout-penalty component carries the cost; the
holding, ordering, and expiry components are small in absolute terms
and behave as the policy intends. This is the same finding the §7.7
calibration table reports for the baseline, and it explains why all
three optimisers converge on a doubled safety stock: the marginal
return per additional unit of safety stock is large when the stockout
penalty is the binding constraint.

### *(Figure 7.2 image — `artefacts/chapter7/figures/figure_7_2_cost_decomposition.png`)*

Figure 7.2 decomposes the total cost into its four components per
method. The stockout penalty is the load-bearing component in every
configuration; holding and ordering costs are an order of magnitude
smaller; expiry cost is effectively zero because the calibrated
(s, S) policy turns stock faster than the shelf-life horizon. The
visual confirms Finding 4: the policy choice does not change the
cost mix, only its overall level.

### *(Figure 7.3 image — `artefacts/chapter7/figures/figure_7_3_convergence.png`)*

Figure 7.3 plots the best-so-far cost against the trial number for
Random Search and Bayesian (Optuna). Both optimisers descend rapidly
in the first ten trials and plateau by trial fifteen. Optuna reaches
its near-optimum in roughly half the budget Random Search requires,
consistent with the Chapter 6 HPO finding that the optimiser does
not matter much at the inner objective; what matters is that an
optimiser is run at all. The dashed red line marks the baseline cost
that both optimisers undercut by trial three.

### *(Figure 7.4 image — `artefacts/chapter7/figures/figure_7_4_grid_heatmap.png`)*

Figure 7.4 visualises the Grid search as a heatmap of cost in the
($\alpha$, $\beta$) plane, sliced at the winning $\gamma$. The cost
surface is smooth and broadly basin-shaped, which explains why all
three optimisers converge on neighbouring solutions: the objective
function has no isolated minima requiring sophisticated exploration.
The dark-green outline marks the Grid winner at
($\alpha = 1.00$, $\beta = 2.00$, $\gamma = 0.50$).

### *(Figure 7.5 image — `artefacts/chapter7/figures/figure_7_5_stockout_incidence.png`)*

Figure 7.5 reports the stockout incidence (percentage of days with
at least one stockout) per method. The three optimised methods cut
stockout incidence from the baseline's 62.4 % to roughly 34 % — a
28-percentage-point reduction that is the proximate cause of the
cost gain. The forecast-driven arm holds stockout incidence at the
baseline level, confirming that the cost increase observed in
Finding 2 comes from extra ordering activity rather than worse
stockout outcomes.

## 7.10.3 What this section hands forward

The headline number for Chapter 8's discussion is that **a Bayesian-
or Grid-optimised (s, S) policy cuts total annual cost by 55 %
against the literature-anchored baseline**, with the 95 % CI tied
between the two methods. The forecast-driven enhancement does not
compound the gain at the daily-aggregated resolution used in this
experiment, which is an honest negative result. The chapter's
contribution to the operational case study is therefore the **policy
optimisation**, not the forecasting enhancement of the policy; the
forecasting work of Chapters 4 to 6 stands on its own as the daily-
arrivals forecast for next-day staffing, while the (s, S) work of
Chapter 7 stands on its own as a textbook-formula refinement that
roughly halves total inventory cost.

A future experiment, suggested in §7.12 as further work, would
re-optimise the multipliers **under** the forecast-driven policy
rather than independently, to test whether the joint optimum can
exceed either component's standalone optimum.

---

## Source files (chapter writer references these)

| Artefact | Path |
|---|---|
| Per-method summary table | `artefacts/chapter7/results/method_summary.csv` |
| Cost decomposition | `artefacts/chapter7/results/cost_decomposition.csv` |
| Grid search trace | `artefacts/chapter7/results/grid_trials.csv` |
| Random search trace | `artefacts/chapter7/results/random_trials.csv` |
| Optuna trial trace | `artefacts/chapter7/results/optuna_trials.csv` |
| Run log | `artefacts/chapter7/results/run.log` |
| Figure 7.1 — Total cost per method | `artefacts/chapter7/figures/figure_7_1_total_cost.png` (+ .pdf) |
| Figure 7.2 — Cost decomposition | `artefacts/chapter7/figures/figure_7_2_cost_decomposition.png` |
| Figure 7.3 — Convergence | `artefacts/chapter7/figures/figure_7_3_convergence.png` |
| Figure 7.4 — Grid heatmap | `artefacts/chapter7/figures/figure_7_4_grid_heatmap.png` |
| Figure 7.5 — Stockout incidence | `artefacts/chapter7/figures/figure_7_5_stockout_incidence.png` |
| Reproducer | `scripts/chapter7_optimization_experiment.py` |
| Renderer | `scripts/chapter7_visualization.py` |
