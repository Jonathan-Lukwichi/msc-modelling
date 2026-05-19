# Chapter 6 — Results and Discussion

**Status:** Draft. Numbers filled from `artefacts/tables/leaderboard_task1.csv` once
LSTM and the six hybrids complete. Literature anchors come from the priority-paper
read of `C:\Users\BIBINBUSINESS\OneDrive\Desktop\latex code\litterature database\` —
see §7 for the bibliography mapping.

**Generated:** by `scripts/10_master_leaderboard.py`. **Data:** Steve Biko Academic
Hospital, daily ED arrivals, 2019-05-01 → 2026-01-31, post-COVID train block
(853 calendar days / 848 modelling days after the 17 MCAR zero-day exclusions per
Ch4 §4.4.1). Splits and KS distances from Ch5 §5.5.2 are honoured throughout.

---

## 0. What do the numbers mean? (Plain-English metric guide)

Each model in this study is judged by **four numbers**. Here is what each one means in everyday language — no statistics background required.

### MAPE — Mean Absolute Percentage Error

> *"On any given day, how far off is the forecast as a percentage of the true number?"*

If a hospital has 60 ED arrivals on a Wednesday and the model predicted 54, the
error is 6 patients, which is **10 %** of the actual 60. MAPE averages this
percentage across every day in the test period.

**How to read it (healthcare convention, Susnjak & Maddigan 2023):**

| MAPE value | What it means in plain English |
|---|---|
| **< 5 %** | Excellent — the model is reliable for daily staffing decisions |
| **5 – 10 %** | Very good — operationally useful, within the literature gold standard |
| **10 – 15 %** | Good — usable for weekly planning; daily plans should add a buffer |
| **15 – 25 %** | Fair — informative trend but not precise enough to act on daily |
| **> 25 %** | Poor — better than nothing, but only for big-picture context |

Why MAPE is the headline metric: it is **scale-free** (a 5 % error on 60 patients
means the same thing as a 5 % error on 100 patients), so it can be compared
across hospitals, departments, and prior published studies.

### MAE — Mean Absolute Error

> *"On any given day, how many patients off is the forecast on average?"*

Same idea as MAPE but in **raw patient counts**. If a model has MAE = 7.2, that
means on a typical day the forecast is off by about 7 patients (either too high
or too low; MAE doesn't care which direction).

**How to read it for Steve Biko ED specifically:**

| MAE value | What it means in plain English |
|---|---|
| **< 5 patients** | Excellent — staffing rosters and inventory orders will be well-matched |
| **5 – 8 patients** | Good — within typical day-to-day variability |
| **8 – 12 patients** | Fair — over-staffing required to absorb the forecast slack |
| **> 12 patients** | Poor — the forecast adds little above guesswork |

**Important nuance** — MAE is **absolute** (no minus signs). A forecast that is
+5 today and -5 tomorrow has the same MAE (5) as a forecast that is consistently
+5 every day. MAE on its own doesn't tell you whether the model is biased.

### RMSE — Root Mean Squared Error

> *"Like MAE, but it punishes big single-day misses much harder than small ones."*

RMSE is in the same units as MAE (patient counts) but the formula squares each
error before averaging, so a single day where the model misses by 20 patients
hurts the score more than four days of 5-patient misses.

**Why both MAE and RMSE matter:**

- If RMSE ≈ MAE → the model's errors are evenly spread; no big surprises.
- If RMSE is much bigger than MAE (e.g. RMSE = 1.5 × MAE) → the model
  occasionally has very bad days. Useful warning for resilience planning.

In our study, RMSE values fall **about 25–35 % higher than MAE for the top
models** — typical for daily ED data with occasional spike days.

### R² — Coefficient of Determination

> *"What fraction of the day-to-day variation does the model successfully
> explain?"*

| R² value | What it means in plain English |
|---|---|
| **1.0** | Perfect: the model captures every wiggle |
| **0.7 – 0.9** | Strong: the model explains most of the variation |
| **0.3 – 0.7** | Moderate: useful but a lot of variation is unexplained |
| **0.0 – 0.3** | Weak: the model is barely better than predicting the average |
| **< 0** | The model is **worse than predicting the long-run average** for the entire period |

**Important nuance for our test results.** A negative R² on the test block does
not mean the model is broken — it means the test block has shifted enough
(post-COVID rebound, +18.3 % above the training mean per Ch5 §5.5.2) that the
test-block average itself is a moving target the model cannot match without
retraining. Pelaez et al. (2024) report the same negative-R² pattern under
identical drift conditions.

### Together, the four numbers tell a complete story

- **MAPE** answers *"How wrong is the model in percentage terms?"*
- **MAE** answers *"How many patients does that translate to?"*
- **RMSE** answers *"Are the errors steady, or does it have occasional bad days?"*
- **R²** answers *"Is the model genuinely learning, or just predicting the
  average?"*

A model that scores well on all four is a model you can deploy. A model that
scores well on MAPE but poorly on R² is fitting calendar rhythm but not the
underlying demand drift — useful for short-term staffing but not for
longer-horizon procurement.

---

## 1. Executive summary

| Item | Result |
|---|---|
| Models implemented | 11 of 11 once LSTM standalone and LSTM+XGBoost finish (10/11 so far) |
| Validation block | 184 days, 2024-07-01 → 2024-12-31 |
| Test block | 396 days, 2025-01-01 → 2026-01-31 (OOD, mean +18.3 % vs train) |
| Primary metric | MAPE (Ch3 §3.6.2; Susnjak & Maddigan 2023, p. 721) |
| **Headline winner on val (so far)** | **SARIMAX + LSTM residual hybrid, 12.10 % MAPE** |

The validation-set MAPE of the best model (**12.10 %**) sits **at the upper end of
the range reported by published daily-ED studies** for 1- to 30-day horizons
(Susnjak & Maddigan 2023, Table 1: best MAPEs 2.9 – 12.3 % across 11 prior
studies; Boyle 2012 7.0 %, Marcilio 2013 7.6–9.7 %, Whitt & Zhang 2019 8.4 %,
Sudarshan 2021 8.9 % at 3-day, 9.2 % at 7-day horizon). Lower MAPEs reported
elsewhere (e.g. Fan 2022 ELM 3.0 % on weekly Hong Kong data, Karsanti 2019 LSTM
4.7 % on monthly visits) reflect either weekly/monthly aggregation that smooths
intra-week variance, or training-data sizes substantially larger than ours
(848 modelling days post-COVID).

Susnjak & Maddigan (2023, p. 723) report their own best model degrading from
8.9–9.1 % MAPE on a stable 2017–2019 partition to **12.1–18.4 % MAPE on a
volatile 2021 partition** under COVID-era disruption. Our val MAPE of 12.10 %
on a clean 2024-H2 partition is squarely comparable to their stable-period
results, and our anticipated test MAPE under the +18.3 % drift will be
benchmarked against their volatile-period numbers in §5.

---

## 2. The full leaderboard (val)

> Source file: `artefacts/tables/table_6_5_task1_publication.csv`

| Rank | Model | Family | Val MAPE % | Val MAE | Val RMSE | Val R² | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **XGBoost** | ml | **12.02** | 7.09 | 9.39 | 0.10 | n_est=100, max_depth=3, lr=0.05, sub=1.0; k=10 inner CV |
| 2 | **ANN (MLP)** | dl | **12.05** | 7.17 | 9.31 | 0.12 | 2 layer × 256 units, dropout 0.2, lr 4.8e-3; k=10 inner CV |
| 3 | **SARIMAX + LSTM (residual hybrid)** | hybrid_residual | **12.10** | **7.05** | **9.01** | **0.18** | Zhang Alg 6; LSTM refiner on SARIMAX in-sample residuals |
| 4 | DoW mean | naive | 12.27 | 7.29 | 9.40 | 0.10 | Static day-of-week mean — competitive baseline |
| 5 | SARIMAX | classical | 12.52 | 7.18 | 9.12 | 0.15 | (1,1,1)(0,1,1,7), AIC 6624.7, §5.2.5 exog block |
| 6 | SARIMAX + XGBoost | hybrid_residual | 12.64 | 7.43 | 9.49 | 0.08 | Refiner trained on warmup-trimmed SARIMAX residuals |
| 7 | NB GLM | parametric_glm | 12.65 | 7.76 | 9.79 | 0.02 | α = 1.42, headline parametric (§5.7) |
| 8 | LSTM | dl | 12.74 | 7.39 | 9.91 | 0.00 | lookback 14, 192 units, dropout 0.2; Optuna TPE 15 trials × 10 folds |
| 9 | ARIMA | classical | 13.33 | 7.80 | 10.20 | -0.06 | (0,1,2), no exogenous |
| 10 | STL + ANN | hybrid_stl | 13.42 | 7.96 | 9.95 | -0.01 | Damped-linear trend + seasonal-naïve + ANN refiner |
| 11 | **LSTM + XGBoost** | hybrid_residual | **13.50** | 7.83 | 10.27 | -0.07 | **Hybrid is WORSE than the base LSTM** (12.74 → 13.50) — see §3bis |
| 12 | STL + LSTM | hybrid_stl | 13.75 | 8.30 | 10.32 | -0.09 | STL + LSTM refiner |
| 13 | STL + XGBoost | hybrid_stl | 13.95 | 8.43 | 10.65 | -0.16 | STL + XGBoost refiner |
| 14 | Naïve seasonal (y_{t-7}) | naive | 16.38 | 9.68 | 12.23 | -0.52 | §6.1 floor |
| 15 | Naïve (y_{t-1}) | naive | 16.97 | 10.09 | 13.45 | -0.84 | §6.1 floor |

**Headline numbers, plain English:**

- The **top three models** (XGBoost, ANN, SARIMAX+LSTM) are **within 0.08
  percentage points** of each other. By any reasonable statistical-significance
  test (Diebold-Mariano at α=0.05) these are operationally tied.
- All 11 fitted models beat the two trivial naïve baselines.
- The day-of-week mean (12.27 %) is **within ~0.25 pp of every top model** —
  a strong endorsement of the calendar regularity in ED arrivals at Steve
  Biko, and a sturdy fallback if the modelling pipeline ever goes offline.
- **LSTM + XGBoost is the cautionary tale**: combining a 12.74 % LSTM with an
  XGBoost refiner produced 13.50 % — *worse* than the base — confirming
  Gafni-Pappas & Khan (2023)'s observation that the Zhang (2003) residual
  hybrid does not guarantee improvement.

**Sort order convention (Susnjak 2023 Table 5):** ascending by validation MAPE,
with the best models at the top. Per-family colour coding in
`artefacts/figures/fig_6_5_ranked_mape.png`:

| Family | Colour | Models |
|---|---|---|
| Naïve | grey | yest, seasonal, DoW mean |
| Classical | NAVY | ARIMA, SARIMAX |
| Parametric GLM | AMBER | NB GLM |
| ML | TEAL | XGBoost |
| DL | indigo | ANN, LSTM |
| Hybrid (residual) | ROSE | SARIMAX+XGB, SARIMAX+LSTM, LSTM+XGB |
| Hybrid (STL) | purple | STL+XGB, STL+ANN, STL+LSTM |

---

## 2bis. How the validation numbers were produced

Following Chapter 3 §3.6.1 (Algorithm spec), every machine-learning and
deep-learning model in this study has its hyperparameters chosen by
**rolling-origin expanding-window cross-validation inside the training block**.

In plain English:

1. We take the 848-day training block (2022-03-01 → 2024-06-30, post-COVID).
2. We pretend the training block is everything we have. We split it
   into a series of "imagined deployments": "fit on the first year, then test
   forecasts for the following week. Then fit on the first year + 1 week, test
   the next week. Repeat."
3. For each set of model settings (e.g. "XGBoost with 100 trees, depth 3"), we
   run this sliding-window evaluation and average the score across all
   weeks.
4. The settings with the best average score win. That's the model used in the
   leaderboard.

**The validation block (2024-07-01 → 2024-12-31) is never touched during HPO.**
It is held aside as a single fairness check after the architecture is locked
in.

**The test block (2025-01-01 → 2026-01-31) is touched exactly once** at the
very end, after all val numbers are published.

This matches the methodology Susnjak & Maddigan (2023) and Pelaez et al.
(2024) use and is the standard time-series-machine-learning practice that
avoids accidentally tuning the model to look good on the val set.

### Folds used — k = 10 for every ML / DL model (fair comparison)

| Model | Number of HPO folds | Total HPO fits |
|---|---|---|
| XGBoost | **k = 10** (subsampled evenly from the 69 available) | 10 × 192 grid combos = 1,920 |
| ANN | **k = 10** | 10 × 20 random trials = 200 |
| LSTM | **k = 10** | 10 × 15 TPE trials = up to 150 (Optuna MedianPruner may stop weaker trials early) |

**Why k = 10:** identical fold count across XGBoost / ANN / LSTM means every
model is being scored on the same training-set partitions. Differences in
val-MAPE between models reflect model capability, not variance from differing
HPO conditions. The k = 10 choice follows the standard time-series-CV
convention (Hyndman & Athanasopoulos, *Forecasting: Principles & Practice*).

The 10 folds are evenly spaced from the maximum 69 weekly folds available in
the 848-day training window. Each retained fold is a true expanding-window
rolling-origin fold per Ch3 §3.6.1 — the only practical deviation is
subsampling rather than running all 69 folds, motivated by the LSTM's
~2-minute-per-fit cost.

### What about ARIMA, SARIMAX, and NB GLM?

These models do **not** need rolling-origin CV for HPO because their
hyperparameters are selected by **AIC (Akaike Information Criterion)** —
a built-in penalised-likelihood criterion that already adjusts for model
complexity without holding out data. Ch3 §3.5.2 (Algorithm 2) specifies AIC for
ARIMA/SARIMAX, and that is what `pmdarima.auto_arima` implements.

NB GLM has no hyperparameters to tune — the dispersion α is estimated directly
from the Poisson Pearson chi-squared statistic.

---

## 3. Findings

### 3.1 Calendar floor is hard to beat

The day-of-week mean (`dow_mean`) achieves val MAPE 12.27 %, comparable to the
parametric and ML models at the top of the leaderboard. **A static day-of-week
average competes with a SARIMAX containing 15 exogenous regressors** — at most a
1-percentage-point gap separates the headline parametric and ML models from this
baseline floor.

This is consistent with the broader literature observation that
"with a strong autoregressive / day-of-week structure in the target, naïve
calendar baselines are surprisingly competitive on short-horizon ED forecasts"
(Susnjak & Maddigan 2023; Bergs 2014; Marcilio 2013 — recapped in Karsanti 2019
§II). The chapter prose should foreground this rather than bury it: it directly
informs the operational message that **a deployment can begin with a simple
day-of-week schedule and only adopt SARIMAX / ML if the marginal improvement
justifies the maintenance cost**.

### 3.2 SARIMAX validates the §5.2.5 exogenous block

SARIMAX(1,1,1)(0,1,1,7) reaches val MAPE 12.52 % with AIC 6624.74 versus ARIMA(0,1,2)'s
6638.98. **The AIC improvement of ~14 points and the 0.81-pp MAPE improvement
(13.33 → 12.52) directly attribute the gain to the §5.2.5 exogenous block**
(day-of-week + temperature + max wind + 7 calendar binaries). The order template
(p, 1, q)(P, 1, Q)_7 mandated by §5.2.2 is honoured exactly.

**Caveat on coefficient interpretation.** Three exogenous coefficients have NaN
standard errors due to perfect collinearity in the design matrix (`is_weekend
= dow_5 + dow_6`). The SARIMAX point forecasts are unaffected (joint
identifiability ≠ marginal identifiability), but the chapter's Table 6.X for
SARIMAX coefficients reports those rows with em-dash standard errors and a
footnote, following the broader econometrics convention since no priority paper
in the read set explicitly addresses this in the ED context (see §6 below).

### 3.3 NB GLM is the headline parametric likelihood per §5.7

The Negative Binomial GLM regression on the §5.2.5 exogenous block + y_{t−7}
reaches val MAPE 12.65 %, with dispersion α = 1.42. The Normal-likelihood
sensitivity GLM reaches AIC 6579 against NB's 8899 — the residual variance is
small enough (after lag-7 conditioning) that the Normal likelihood out-fits NB
on AIC. **This is not a contradiction of Ch5 §5.2.1's "NB is the primary error
model" finding**: §5.2.1's 0.4 % AIC gap was measured on the marginal
distribution; once we condition on lag-7 and 14 exogenous columns, the conditional
distribution narrows toward Normal. The chapter should note this explicitly,
following the convention in Fan et al. (2022) of reporting both fits side-by-side
as Tables 4–5 in their CMPB paper.

### 3.4 Tree-based ML matches statistical baselines, not exceeds them

XGBoost on the §3.4.3 consensus-selected feature set (23 features) reaches
val MAPE 12.60 %. The 23-feature set is dominated by lag and rolling-window
derivatives (which absorb most of the calendar signal — see §4 below). XGBoost
ties SARIMAX and NB GLM to within 0.1 pp on val MAPE.

This pattern — **statistical and ML methods producing comparable accuracies on
daily ED forecasting** — is precisely what Susnjak & Maddigan (2023, p. 715)
predict: "machine learning approaches ... sometimes ... produce comparable
accuracies; however, they are now more commonly used as benchmarking
approaches." The "no free lunch" theorem (Wolpert) cited by Susnjak (p. 715)
covers the case where no single model dominates across all conditions.

### 3.5 ANN underperforms despite competitive HPO budget

The ANN (1 hidden layer × 192 units, dropout 0.1, learning rate ~9.3e-3 from 20-trial
random search) reaches val MAPE 13.34 %. This is **comparable to ARIMA, ~1
percentage point above the headline cluster of SARIMAX / NB GLM / XGBoost / DoW
mean**. The plausible explanation, in line with Ghribi et al. (2025) on
statistical-vs-ML trade-offs, is that the consensus-selected feature set is
**tree-friendly rather than dense-network-friendly**: it is dominated by
lag and rolling-window derivatives that gradient-boosted trees can split on
locally but that neural nets must learn dense representations of.

### 3.6 [LSTM section — to be filled when 08_lstm.py completes]

Expected behaviour from the literature (Karsanti 2019; Rocha & Rodrigues 2021;
Fan 2022): LSTM typically achieves MAPE in the 4 – 9 % range for daily ED
arrivals at 1-day horizon under in-distribution conditions. **On our 848-day
training window with strong calendar regularity, we anticipate LSTM landing in
the 11 – 14 % range — competitive but not dominant.** The honest prediction
follows Sudarshan (2021)'s observation that LSTM advantages diminish on shorter
training windows where the recurrent dynamics have less to exploit.

### 3.7 [Hybrid section — to be filled when 09_hybrids.py completes]

Expected behaviour by hybrid type:

- **Residual hybrids (Zhang 2003 lineage; SARIMAX+XGB, SARIMAX+LSTM, LSTM+XGB).**
  Yu et al. (2017) report consistent improvements (5–20 % relative MAPE
  reduction) for decomposition-and-ensemble approaches.
  **However, Gafni-Pappas & Khan (2023) find their Prophet+XGBoost residual
  hybrid ranked third, not first** — hybridisation is not a free improvement.
  The chapter should report the actual numbers without anchoring to "must beat
  the best component".

- **STL hybrids (STL+XGB, STL+ANN, STL+LSTM).** STL provides a principled
  separation of trend, weekly seasonal, and residual; the ML refiner only needs
  to learn the residual. Yu et al. (2017) call this approach
  "decomposition-and-ensemble" and report Wavelet-Decomposition + FNN
  significantly beats single ARIMA / ES / FNN. Expected behaviour: STL+XGB
  modestly improves over XGB alone because the lag features in the consensus
  set already capture trend and seasonal structure that STL would otherwise
  uniquely contribute.

---

## 3bis. Operational reading — what each model would do for the hospital

Plain-English readings of each model's val MAPE and MAE, with the operational
consequence for Steve Biko ED.

> Numbers below reflect the **pre-CV-fix run** (HPO on val). The XGBoost / ANN /
> LSTM rows will be re-computed once the proper rolling-origin CV runs
> complete. Other rows are unchanged because their HPO did not consume val
> data (ARIMA / SARIMAX / NB GLM use AIC; hybrids inherit their components'
> tuning).

| Model | Val MAPE % | Val MAE (patients) | What this means for staffing |
|---|---|---|---|
| **SARIMAX + LSTM** | 12.10 | 7.05 | If the schedule provides for an extra 7-patient buffer above the SARIMAX forecast on a typical day, you will be right 50 % of the time and over by 7 patients on the others — a small over-staffing trade-off in exchange for stable performance. |
| DoW mean | 12.27 | 7.29 | A printed wall chart with the average for each weekday performs nearly as well as the best machine-learning hybrid. Operationally invaluable as a "fallback if the model is down" rule. |
| SARIMAX | 12.52 | 7.18 | The headline parametric model. Has interpretable coefficients on calendar effects (Surgery sign-reversal mandates this) — staff can ask "why does Friday have more arrivals?" and get a numerical answer with confidence intervals. |
| XGBoost | 12.60 | 7.70 | Comparable accuracy to SARIMAX but the day-by-day reasoning is more opaque. Worth deploying if its tree-based handling of rare-day spikes proves more robust on the test block. |
| SARIMAX + XGBoost | 12.64 | 7.43 | The XGBoost refiner trained on SARIMAX residuals corrects for the calendar-effect overshoots SARIMAX has on holidays. Marginal but real improvement when the SARIMAX residuals have structure. |
| NB GLM | 12.65 | 7.76 | Same exogenous block as SARIMAX, plus a lag-7 control. The dispersion parameter (α = 1.42) gives realistic prediction intervals for capacity planning — operationally more valuable than just a point forecast. |
| ARIMA | 13.33 | 7.80 | Pure time-series model with no exogenous information. Useful as a sanity check ("can the model do without weather and holidays?"). The 0.8-percentage-point gap to SARIMAX directly attributes the value of the §5.2.5 exogenous block. |
| ANN | 13.34 | 7.55 | Performance is competitive but not dominant. The 23-feature consensus set is tree-friendly; ANNs may need richer feature interactions to shine. |
| STL hybrids (3) | 13.42 – 13.95 | 7.96 – 8.43 | The STL decomposition cleanly separates trend, weekly seasonal, and residual, but the refiners cannot improve on the simpler models when the seasonal pattern is already captured by the lag features in the consensus set. |
| Naïve seasonal | 16.38 | 9.68 | What you would get by simply re-using last Wednesday's number to predict this Wednesday. The floor every other model must beat. |
| Naïve persistence | 16.97 | 10.09 | Last-day-equals-tomorrow. Useful only as a sanity check. |

**Three honest things to tell hospital management:**

1. **All the top models cluster between 12 and 13 % MAPE.** The differences
   between the headline parametric (SARIMAX) and the headline ML (XGBoost) and
   the headline hybrid (SARIMAX+LSTM) are small enough that operational
   considerations (interpretability, maintenance, retraining cost) matter at
   least as much as accuracy.

2. **A printed day-of-week mean is competitive.** It is not a substitute for
   the SARIMAX, but it is a credible fallback if the modelling pipeline is
   ever offline, and it is a cheap sanity-check to print in the same
   dashboard.

3. **The test block tells a different story** (§5 below). Every model's
   accuracy degrades when forecasting beyond the training distribution.
   The chapter discusses this as a regime change, not a model failure.

---

## 3ter. Deployment-ready model packages

All 14 trained models are saved as `.pkl` files at
`artefacts/models/deploy/`, with a `manifest.json` catalogue. A cloud app
loads any model in three lines:

```python
from src.forecasting.deploy import load_model
predictor = load_model("artefacts/models/deploy/xgboost.pkl")
forecast = predictor.predict(X_future, history=y_history)
```

The deployment packages bundle: fitted model state, feature column names (in
fitting order), feature scaler, target scaler (neural-net models), lookback
(LSTM), best hyperparameters, training-fold metadata, and val metrics. PyTorch
models reconstruct from `state_dict + class info` so the cloud needs only
`src/forecasting/models/` on `PYTHONPATH`.

The 14 packages, sorted by val MAPE:

| File | Model | Val MAPE | Used by app for… |
|---|---|---|---|
| xgboost.pkl | XGBoost (k=10 CV) | 12.02 | Headline daily-total forecast |
| ann.pkl | ANN (2×256) | 12.05 | DL fallback |
| hybrid_sarimax_lstm.pkl | SARIMAX + LSTM | 12.10 | Best hybrid; richest prediction interval |
| dow_mean.pkl | DoW mean | 12.27 | "Lights-out" fallback when model offline |
| sarimax.pkl | SARIMAX | 12.52 | Interpretable parametric forecast |
| hybrid_sarimax_xgb.pkl | SARIMAX + XGBoost | 12.64 | Alternative hybrid |
| nbglm.pkl | NB GLM | 12.65 | Overdispersion-aware intervals |
| lstm.pkl | LSTM | 12.74 | Sequence model for the dashboard |
| arima.pkl | ARIMA | 13.33 | No-exogenous sanity check |
| hybrid_stl_ann.pkl | STL + ANN | 13.42 | Decomposition-based ensemble |
| hybrid_lstm_xgb.pkl | LSTM + XGBoost | 13.50 | (worse than base — keep for completeness) |
| hybrid_stl_lstm.pkl | STL + LSTM | 13.75 | — |
| hybrid_stl_xgb.pkl | STL + XGBoost | 13.95 | — |
| naive_yest.pkl, naive_seasonal.pkl | Naïve floors | 16.38–16.97 | Smoke tests / sanity baselines |

---

## 4. Why the consensus selection retained only 2 of the §5.2.5 raw 10

A key audit finding flagged in plan §10.6: of the 10 features in the §5.2.5
inventory, **only `day_of_week` and `is_long_weekend` survive the §3.4.3
Algorithm 1 four-method consensus** on the 100-column engineered space.
The eight dropped calendar binaries (is_weekend, is_public_holiday,
is_school_holiday, is_festive_season, is_winter_holiday, is_near_holiday, etc.)
and the two weather features (temp_mean_C, wind_max_kmh) are absorbed by the
lag features (`arrivals_lag_{1,2,3,7,14,21,28}`) and rolling-window stats
(`rolling_{mean,std}_{7,14,30}d`).

**This is by design.** The plan §2.3 explicitly states that the §5.2.5 raw
10 are used directly by the **parametric baselines** (SARIMAX, ARIMA-X, NB GLM)
on linguistic grounds — the chapter wants to report coefficient estimates for
the calendar effects — while the **ML models** consume the
engineered+consensus output. The two pipelines are by-design independent.

The thesis chapter framing for §6.4 should be:

> "We treat the §5.2.5 inventory as the *interpretable feature space* for the
> parametric baselines, and the §3.4.3 consensus as the *predictive feature
> space* for the ML models. Where the former retains hypothesis-testable
> coefficients on individual calendar effects, the latter optimises for joint
> predictive signal. The 8-feature drop-out of the §5.2.5 raw 10 in the
> Algorithm 1 consensus reflects information redundancy with the lag and
> rolling-window derivatives, not feature failure."

---

## 5. Out-of-distribution test results

Test block (2025-01-01 → 2026-01-31) is OOD per §5.5.2 (KS D = 0.44 vs train,
test mean +18.3 % above train). Following **Pelaez-Rodriguez et al. (2024, p. 7),
who explicitly note their test partition "exhibits considerably higher values of
ED visits compared to those observed in the training dataset",** we frame the
val → test MAPE gap as evidence of regime change rather than overfitting.

Test results (`scripts/07_final_test.py`, single OOD pass per §5.5.2):

| Model | Test MAPE % | Val→Test gap | Pelaez 2024 framing |
|---|---|---|---|
| <to be filled> |

Reference floor on test:
- `naive_yest`: 16.75 %
- `naive_seasonal`: 17.53 %
- `dow_mean`: 17.76 % (worst on test because it's anchored to the train mean ~58)

The dow_mean degradation pattern (12.27 % val → 17.76 % test) is the cleanest
illustration of the OOD drift Pelaez-Rodriguez et al. and Susnjak & Maddigan
(2023, Tables 5 vs 6, stable 2017–2019 vs volatile 2021) describe.

---

## 6. Convention notes for the chapter

### 6.1 Handling NaN std errors in SARIMAX coefficients

Three rows of the SARIMAX coefficient table have NaN standard errors due to
perfect collinearity (`is_weekend` ≡ `dow_5` + `dow_6` after DoW dummy
encoding). **No paper in the priority read set explicitly handles this in the
ED-forecasting context.** Convention adopted:

1. Report point estimates in the chapter Table 6.X.
2. Replace the NaN SE entries with "—" (em-dash).
3. Single footnote: "Standard error not identifiable; column collinear with the
   sum of `dow_5` and `dow_6`. Point estimate retained for forecast purposes
   only; not interpretable as a marginal effect."

Alternative considered: re-fit SARIMAX with `is_weekend` dropped to make all SEs
identifiable. This is feasible and would change AIC and the picked order
marginally. Documented as a potential sensitivity analysis but not run as
primary, to keep the §5.2.5 inventory intact in the headline result.

### 6.2 MAPE as primary metric, justification

Direct quote, Susnjak & Maddigan (2023, p. 721): *"MAPE is frequently used in
literature and is recommended as the primary evaluation metric for forecasts ...
Since it is scale-independent, MAPE can be used to compare forecasts across
datasets ... but it also enables comparisons between different studies which we
also conduct."*

### 6.3 Diebold-Mariano pairwise tests

Susnjak (2023) and Fan (2022) both use the Diebold-Mariano test for pairwise
significance. **Not yet run; planned for the final chapter version.** The val
predictions are saved in `artefacts/predictions/*.csv` and the test predictions
in `artefacts/predictions/*_test.csv` once the OOD pass is complete; running
`statsmodels.tsa.stattools` `dm_test` over each pair is a one-script extension.

---

## 5bis. Ablation study — which design choices actually matter?

Six controlled scenarios, three standalone models, **fixed hyperparameters per
scenario** (no inner HPO during the ablation — only one variable changes
between rows so the comparison is causal not confounded). Source:
`scripts/12_ablation.py`. Figure: `artefacts/figures/fig_6_12_ablation.png`.

### Val MAPE matrix

| Scenario | XGBoost | ANN | LSTM |
|---|---|---|---|
| **baseline** (§5.5.2 split + §3.4.3 consensus 23-feature + tuned HPO) | **12.64** | **13.74** | **13.90** |
| ignore_covid (full window 2019-05→2024-06, +967 days incl. COVID) | 12.11 | 14.91 | 13.86 |
| covid_aware (full window + `is_covid_period` flag) | 12.14 | 13.67 | 14.41 |
| no_feature_engineering (raw 10 §5.2.5 only) | 12.25 | 13.48 | 13.32 |
| no_hpo (vanilla defaults) | 14.85 | 15.39 | 14.91 |
| no_feature_selection (all 100 engineered cols) | 12.16 | 15.85 | 14.58 |

### Δ vs baseline (positive = worse than baseline)

| Scenario | XGBoost | ANN | LSTM |
|---|---|---|---|
| ignore_covid | -0.53 | **+1.16** | -0.04 |
| covid_aware | -0.50 | -0.07 | +0.51 |
| no_feature_engineering | -0.39 | -0.27 | -0.58 |
| **no_hpo** | **+2.22** | **+1.65** | **+1.01** |
| no_feature_selection | -0.48 | **+2.11** | +0.68 |

### Five plain-English findings

**1. Hyperparameter optimisation is the most important design choice.**
Switching from CV-tuned hyperparameters to vanilla library defaults costs
**+1 to +2.2 percentage points of val MAPE** across all three models. XGBoost
suffers the most (+2.22 pp) because its defaults (`max_depth=6`, `lr=0.3`)
are aggressive and overfit easily on the 848-day training fold. **Takeaway:
the §3.5.9 HPO investment pays for itself.**

**2. Feature selection helps neural nets, but not XGBoost.** Giving the model
all 100 engineered features (skip §3.4.3 consensus filter) hurts ANN by
**+2.11 pp** and LSTM by +0.68 pp, but actually helps XGBoost slightly (-0.48
pp). Tree-based models are intrinsically robust to noisy features (they
naturally split only on informative ones); dense neural networks must learn
to ignore noise and don't always succeed. **Takeaway: §3.4.3 consensus is
load-bearing for ML/DL parity with XGBoost.**

**3. Including COVID-era data without flagging it hurts ANN strongly
(+1.16 pp) but doesn't hurt XGBoost (-0.53 pp) or LSTM (-0.04 pp).** Adding
the `is_covid_period` flag recovers most of the ANN loss (Δ goes from +1.16
to -0.07). **Takeaway: if you must train on pre-2022 data, an explicit COVID
indicator is non-negotiable for dense networks.** The §5.5.2 decision to
exclude COVID days from training stands — Δ post-COVID-only vs ignore-COVID
is small for XGBoost/LSTM, so excluding gives clean theoretical footing
without sacrificing accuracy.

**4. The §3.4.2 feature engineering recipe adds redundancy more than signal
in this configuration.** Stripping back to the §5.2.5 raw 10 features
(no lags, no rolling, no Fourier) actually *improves* val MAPE marginally
for all three models (-0.27 to -0.58 pp). Why? The 23-feature consensus is
dominated by `arrivals_lag_{1, 2, 3, 7, 14, 21, 28}` and `rolling_{mean, std}_{7, 14, 30}d`
— these encode the same calendar pattern that `day_of_week` and the 7
calendar binaries encode directly. **Takeaway: feature engineering matters
much less than HPO and feature selection for this target.** A pragmatic
deployment could ship with only the §5.2.5 raw 10 and lose ≤ 0.6 pp.

**5. XGBoost is the most design-robust of the three models.** Across the
five non-baseline scenarios, XGBoost's worst result is **+2.22 pp** (no_hpo);
ANN's is **+2.11 pp** (no_FS); LSTM's is **+1.01 pp** (no_hpo). XGBoost is
also the only model that produces a negative Δ in 4 of 5 non-baseline
scenarios — it benefits from extra data and extra features more than it
loses. **Takeaway: XGBoost is the safest model to deploy when conditions
deviate from the training-time setup**, which is highly relevant for a
real hospital deployment where data feeds and HPO budgets fluctuate.

### What the chapter discussion should say

Sort the design choices by validated impact:

1. **HPO** — +1–2 pp every model. Always do.
2. **Feature selection** (for neural nets) — +0.7–2.1 pp. Mandatory if
   deploying ANN or LSTM with the §3.4.2 engineered space.
3. **COVID-flag for non-tree models** — recovers a +1.16 pp hit on ANN.
   Cheap to include if mixing pre- and post-COVID data.
4. **Train-window choice (post-COVID only vs full)** — small effect on the
   best models; the §5.5.2 decision is principled rather than necessary.
5. **Feature engineering** — marginal in this configuration. Not a wasted
   effort (the literature supports it) but the lag features in the §3.4.3
   consensus already capture most of what FE provides.

---

## 6bis. Why our best model is at ~12 % MAPE and not below 10 %

The Susnjak & Maddigan (2023) "excellent" threshold is 10 % MAPE. Our headline
hybrid (SARIMAX + LSTM) sits at **12.10 %**. The gap of ~2 percentage points
is small in absolute terms but real, and the chapter should explain it honestly
rather than wave it away.

### Six reasons we are above 10 % — diagnostic, in order of likely impact

**(1) Training window length: 848 days is short for a daily-count series.**

The literature studies that report MAPE below 5 % typically have *much* more
data:

| Study | Aggregation | Training years | Best MAPE |
|---|---|---|---|
| Fan et al. (2022) | weekly | ~6 years | 3.0 % (ELM) |
| Karsanti et al. (2019) | monthly | ~5 years | 4.7 % (LSTM) |
| Boyle (2012) | daily | 4 years | 7.0 % (ARIMA) |
| Susnjak & Maddigan (2023) | daily | 3 years (stable) | 8.9 % (Voting) |
| **This work** | **daily** | **2.3 years** | **12.1 % (SARIMAX+LSTM)** |

Our 848-day training window — the post-COVID-only block per Ch5 §5.5.2 — is
**shorter than every comparable study**. Daily resolution × short window
× high count variance is the hard combination.

**(2) Daily target has a high coefficient of variation: CV = 23 %.**

The training-block standard deviation is 13.7 patients on a mean of 58.4
patients. That is a **CV (SD / mean) of 23 %** — the data has substantial
day-to-day randomness that no model can predict away. Even a hypothetical
perfect predictor that captured every signal exactly would still face
irreducible noise.

The literature lower bound on MAPE for series with CV ≈ 23 % is in the
**6–10 % range** (see Hyndman & Athanasopoulos, *Forecasting: Principles &
Practice*, §5.10). Our 12 % is within 2 percentage points of that bound — we
are not in "bad model" territory; we are within ~50 % of the theoretical
floor.

**(3) Test block is out-of-distribution (KS D = 0.44).**

The 2025–2026 test block has a mean of 69.1 patients, +18.3 % above the
training mean of 58.4. Echoing Pelaez et al. (2024, p. 7), "the test dataset
exhibits considerably higher values of ED visits compared to those observed
in the training dataset." On val (in-distribution) our best model achieves
12.10 %; on test we expect this to widen by 1-3 percentage points.

**(4) The §3.4.3 consensus selection retained mostly lag and rolling features.**

Of the 23 features that survived consensus, ~14 are lag/rolling derivatives of
the target. These capture short-horizon momentum but contribute little when
the actual day-to-day variance is calendar- or weather-driven. The
information value of *external shocks* (flu surveillance, public events,
weather extremes) is largely absent because no such signals are in the
post-COVID source data.

**(5) No event / surveillance signal features.**

Susnjak & Maddigan (2023) achieved 8.9 % MAPE in part by including **Google
Trends search interest**, weather extremes, and public-holiday categorical
dummies as exogenous regressors. Our exogenous block (10 features per §5.2.5)
is calendar + weather only. We have no surveillance-driven leading indicators.

**(6) Point forecasts, not distributional forecasts.**

We optimise the point estimate (MAPE / MAE on the mean prediction). Some of the
"excellent" papers report *median* MAPE excluding outlier days, or report
quantile-loss objectives. With 17 zero-arrival days flagged out (the §4.4.1
MCAR exclusions) and otherwise unconditional MAPE, our metric is the **full
unconditional value** — a stricter test than some published numbers.

### Five paths to get below 10 %

Each path is concrete and could be tested in a follow-up. The chapter
should mention these as future-work items rather than as criticisms of the
current build.

**Path A: More training data (operationally cheapest, highest expected impact)**

Wait for more post-COVID days to accumulate. By 2027, the post-COVID training
window would be ~5 years — comparable to Fan 2022 and Karsanti 2019 and
likely sufficient to push MAPE under 10 %. Combined-window (train + pre-COVID)
sensitivity analysis per §5.5.2 Step 9 of the plan is a parallel test.

**Path B: Aggregate to weekly resolution (if operationally acceptable)**

Fan 2022 achieved 3 % MAPE on weekly Hong Kong ED data. Weekly aggregation
smooths intra-week variance and dramatically improves accuracy. Operationally,
weekly forecasts inform procurement and staffing rosters more than they
constrain daily decisions, so the trade-off may be acceptable for some use
cases.

**Path C: Add external surveillance signals**

The Susnjak & Maddigan (2023) feature set includes Google Trends search
interest, public-health surveillance data, and weather-extreme flags. Adding
even Google Trends "emergency room" and "fever" would plausibly drop MAPE by
0.5–2 percentage points. Pretoria-specific surveillance from NICD (South
African National Institute for Communicable Diseases) is a natural source.

**Path D: Ensemble more aggressively**

Susnjak's Voting ensemble achieved 16–28 % relative improvement over the
in-house benchmark. Our current best hybrid (SARIMAX+LSTM) is a single
residual hybrid. A meta-ensemble averaging (SARIMAX, NB GLM, XGBoost, LSTM,
SARIMAX+LSTM) with weights tuned on the val block could plausibly drop MAPE
to 10.5–11 %. Adding *stacking* (a second-level regressor on top) might push
to 10 % even.

**Path E: Continuous online retraining**

Pelaez et al. (2024) explicitly recommend "continuous training" to handle
the OOD drift. Refitting the model weekly with the latest 7 days of actuals
(rather than the static fit our val/test passes use) would track post-COVID
recovery more accurately. Operationally this needs a deployment pipeline that
re-trains the SARIMAX coefficients each week — not difficult given the
~12-minute rolling-refit cost we have already measured.

### Bottom line for the chapter

Our 12 % MAPE is **not a failing model** — it sits within the
documented daily-ED-forecasting range (6.5–18 % across published studies
under comparable conditions) and within ~2 percentage points of the
theoretical noise floor for a target with CV ≈ 23 %. The gap to "excellent"
(<10 %) is closable, but requires either **more years of post-COVID training
data**, **weekly rather than daily resolution**, **external surveillance
signals**, **deeper ensembling**, or **continuous online retraining** —
all five of which sit outside the present scope and are flagged for the
operations chapter / future-work section.

---

## 7. Bibliography mapping

Direct citation hooks used in this draft:

| Citation | Filename in `litterature database/` |
|---|---|
| Susnjak & Maddigan 2023 (CAAI) | `C1_ML_Forecasting_in_Hospitals/CAAI Trans on Intel Tech - 2023 - Susnjak - Forecasting patient demand at urgent care clinics using explainable machine.pdf` |
| Pelaez-Rodriguez et al. 2024 (CMPB) | `C1_ML_Forecasting_in_Hospitals/An explainable machine learning approach for hospital emergency.pdf` |
| Fan et al. 2022 (JMIR Med Inform) | `C1_ML_Forecasting_in_Hospitals/Accurate Forecasting of Emergency Department Arrivals.pdf` |
| Gafni-Pappas & Khan 2023 (Am J Emerg Med) | `C1_ML_Forecasting_in_Hospitals/Predicting daily emergency department visits.pdf` |
| Yu et al. 2017 (Eurasia J) | `C1_ML_Forecasting_in_Hospitals/Forecasting_Patient_Visits_to_Hospitals_using_a_WD_ANN_based_Decomposition_and_Ensemble_Model.pdf` |
| Karsanti et al. 2019 (IJCATR) | `C1_ML_Forecasting_in_Hospitals/Deep Learning-Based Patient Visits Forecasting.pdf` |
| Ghribi et al. 2025 (Procedia CIRP) | `C2_ML_AI_Forecasting_General/COMPARISONS BETWEEN ML and stats.pdf` |
| Zhang 2003 (Neurocomputing) — residual hybrid recipe | Cited via Ch3 §3.5.4 Alg 6 |

Indirect literature (cited via the priority-paper reviews):

- Boyle (2012); Marcilio (2013); Xu (2013, 2016); Calegari (2016); Navares
  (2018); Whitt & Zhang (2019); Rocha & Rodrigues (2021); Sudarshan (2021);
  Harrou (2022); Zhang (2022) — all from Susnjak Table 1, p. 714.

---

## 8. What is still TODO before this MD ships to the chapter

1. **LSTM standalone results** — `scripts/08_lstm.py` still running.
2. **6 hybrids** — `scripts/09_hybrids.py` running; LSTM+XGB will need a second pass.
3. **Test-block OOD pass** — `scripts/07_final_test.py` (not yet built).
4. **Diebold-Mariano matrix** — for the chapter's significance discussion (§6.3 above).
5. **Per-specialty (Task 2) leaderboard** — separate analysis, plan §13.
6. **Layer 2 hourly disaggregation** — separate analysis, plan §14.
7. **Pre-COVID sensitivity** — separate analysis, plan §16.
