"""Chapter 7 — Inventory (s, S) policy optimisation experiment.

Compares five approaches to setting the (s, S) policy parameters for the
30-item Steve Biko ED inventory panel:

  1. Baseline            — Silver et al. 2017 textbook formula (current
                            literature-anchored configuration)
  2. Grid Search          — exhaustive search over (alpha, beta, gamma)
                            multipliers on the reorder-point and order-up-to
  3. Random Search        — Bergstra & Bengio 2012 uniform sampling
  4. Bayesian (Optuna TPE) — Akiba et al. 2019
  5. Forecast-driven      — Chapter 6 XGBoost daily arrivals forecast feeds
                            a time-varying mean-consumption estimate

The objective is total annual cost (holding + ordering + stockout penalty
+ expiry).  Common Random Numbers are used across all five methods: every
configuration is evaluated on the same N_SEEDS draws of arrivals
distribution, lead-time variates, and procurement-failure events.

Data inputs (read-only):
  - items_master.csv      : 30-item panel (chapter7_simulation/)
  - arrivals_test_block   : 396-day Steve Biko ED arrivals
  - xgboost_rmse.csv      : Chapter 6 winning forecast on the test block

Outputs (written):
  - results/method_summary.csv     : per-method mean cost + 95% CI
  - results/optuna_trials.csv      : Optuna trial trace
  - results/random_trials.csv      : Random search trial trace
  - results/grid_trials.csv        : Grid search trace
  - results/cost_decomposition.csv : per-method cost components

The script writes everything under artefacts/chapter7/.  No state from the
chapter7_simulation/ folder is mutated.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"C:/Users/BIBINBUSINESS/OneDrive/Desktop/dataAnalysis/chapter7_simulation")
OUT_DIR = ROOT / "artefacts" / "chapter7" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITEMS_CSV    = DATA_DIR / "items_master.csv"
ARRIVALS_CSV = DATA_DIR / "arrivals_test_block.csv"
FORECAST_CSV = ROOT / "artefacts" / "predictions" / "test" / "xgboost_rmse.csv"

START_DATE = pd.Timestamp("2025-01-01")
SIM_DAYS = 396
N_SEEDS = 8                   # CRN — same seeds across all methods
SEEDS = list(range(42, 42 + N_SEEDS))

# Procurement-failure mechanism rates (Modisakeng 2020, same as v3.1 sim)
LEAD_TIME_MU = math.log(21.0)
LEAD_TIME_SIGMA = (math.log(90.0) - math.log(21.0)) / 1.6449
NON_PERFORMANCE_P = 0.25
NON_PERFORMANCE_MULT = 3.0
PAYMENT_DELAY_P = 0.12
PAYMENT_DELAY_MULT = 2.0
CONSUMPTION_DISPERSION = 1.4   # NB dispersion phi

# Specialty share of daily arrivals (Chapter 5 §5.5)
SPECIALTY_SHARES = {"medicine": 0.73, "ortho": 0.17, "lowvol": 0.10}


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
@dataclass
class Item:
    item_id: str
    abc_class: str
    used_by_specialty: str
    unit_price_zar: float
    units_per_patient_mean: float
    probability_of_use_per_patient: float
    lead_time_mean_days: float
    ordering_cost_zar: float
    holding_cost_rate_per_year: float
    stockout_penalty_zar: float
    shelf_life_months: float
    initial_stock_units: float


def load_items() -> list[Item]:
    df = pd.read_csv(ITEMS_CSV)
    out = []
    for _, r in df.iterrows():
        out.append(Item(
            item_id=r["item_id"],
            abc_class=r["abc_class"],
            used_by_specialty=r["used_by_specialty"],
            unit_price_zar=float(r["unit_price_zar"]),
            units_per_patient_mean=float(r["units_per_patient_mean"]),
            probability_of_use_per_patient=float(r["probability_of_use_per_patient"]),
            lead_time_mean_days=float(r["lead_time_mean_days"]),
            ordering_cost_zar=float(r["ordering_cost_zar"]),
            holding_cost_rate_per_year=float(r["holding_cost_rate_per_year"]),
            stockout_penalty_zar=float(r["stockout_penalty_zar"]),
            shelf_life_months=float(r["shelf_life_months"]),
            initial_stock_units=float(r["initial_stock_units"]),
        ))
    return out


def load_arrivals_actual() -> np.ndarray:
    df = pd.read_csv(ARRIVALS_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df["actual_arrivals"].astype(float).values[:SIM_DAYS]


def load_arrivals_forecast() -> np.ndarray:
    df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df["predicted"].astype(float).values[:SIM_DAYS]


# ----------------------------------------------------------------------
# (s, S) PARAMETER COMPUTATION
# ----------------------------------------------------------------------
def compute_ss(item: Item, mean_c: float, alpha: float, beta: float,
                gamma: float) -> tuple[float, float]:
    """Compute (s, S) for one item given a mean daily consumption.

    Parameterisation (multipliers on the textbook formula):
        s_textbook = mean_c * L + 1.65 * sqrt(var_c * L + mean_c^2 * var_L)
        s = alpha * (mean_c * L) + beta * safety_stock
        S = s + gamma * EOQ
    Baseline is alpha = beta = gamma = 1.
    """
    L = item.lead_time_mean_days
    var_L = (math.exp(LEAD_TIME_SIGMA**2) - 1) * math.exp(
        2 * LEAD_TIME_MU + LEAD_TIME_SIGMA**2
    )
    mean_c = max(mean_c, 0.001)
    var_c = mean_c * (1.0 + mean_c / CONSUMPTION_DISPERSION)   # NB variance

    safety = 1.65 * math.sqrt(var_c * L + mean_c**2 * var_L)
    s = alpha * mean_c * L + beta * safety

    annual_demand = mean_c * 365
    if item.unit_price_zar > 0 and item.holding_cost_rate_per_year > 0:
        eoq = math.sqrt(
            2 * annual_demand * item.ordering_cost_zar
            / (item.holding_cost_rate_per_year * item.unit_price_zar)
        )
    else:
        eoq = mean_c * 30

    S = s + gamma * eoq
    return s, S


# ----------------------------------------------------------------------
# STREAMLINED SIMULATOR
# ----------------------------------------------------------------------
def simulate_one_seed(
    items: list[Item],
    arrivals: np.ndarray,
    seed: int,
    alpha: float, beta: float, gamma: float,
    forecast_path: np.ndarray | None = None,
) -> dict:
    """Simulate one seed under the given (alpha, beta, gamma).

    If `forecast_path` is provided, the (s, S) for each item is recomputed
    every day using a rolling-window mean of the forecast over the next
    lead_time_mean_days; otherwise (s, S) is computed once from the
    historical mean consumption per item.
    """
    rng = np.random.default_rng(seed)

    # Per-item mean daily consumption based on historical arrivals
    mean_arrivals = arrivals.mean()
    item_mean_c = []
    for it in items:
        share = SPECIALTY_SHARES.get(it.used_by_specialty, 0.10)
        mean_c = (mean_arrivals * share *
                  it.probability_of_use_per_patient *
                  it.units_per_patient_mean)
        item_mean_c.append(mean_c)

    # Initialise per-item state
    n_items = len(items)
    stock = np.array([it.initial_stock_units for it in items], dtype=float)
    pending = [[] for _ in range(n_items)]   # list of (arrival_day, qty)
    shelf_lots = [[(it.initial_stock_units,
                     int(it.shelf_life_months * 30))]
                   for it in items]

    # Cost accumulators
    holding_zar = 0.0
    ordering_zar = 0.0
    stockout_zar = 0.0
    expiry_zar = 0.0
    n_orders = 0
    n_stockout_events = 0
    stockout_days = np.zeros(SIM_DAYS)

    # Precompute static (s, S) if no forecast
    static_ss = []
    for i, it in enumerate(items):
        s_, S_ = compute_ss(it, item_mean_c[i], alpha, beta, gamma)
        static_ss.append((s_, S_))

    for day in range(SIM_DAYS):
        arrivals_today = arrivals[day]

        # 1. Receive any pending orders whose arrival_day == today
        for i in range(n_items):
            new_pending = []
            for (arr_day, qty) in pending[i]:
                if arr_day <= day:
                    stock[i] += qty
                    shelf_lots[i].append(
                        (qty, int(items[i].shelf_life_months * 30))
                    )
                else:
                    new_pending.append((arr_day, qty))
            pending[i] = new_pending

        # 2. Consume
        for i, it in enumerate(items):
            share = SPECIALTY_SHARES.get(it.used_by_specialty, 0.10)
            expected = (arrivals_today * share *
                         it.probability_of_use_per_patient *
                         it.units_per_patient_mean)
            expected = max(expected, 0.001)
            # NB draw
            phi = CONSUMPTION_DISPERSION
            p = phi / (phi + expected)
            consumption = float(rng.negative_binomial(phi, p))

            if consumption > stock[i]:
                shortage = consumption - stock[i]
                stockout_zar += shortage * it.stockout_penalty_zar
                stock[i] = 0.0
                stockout_days[day] = 1.0
                n_stockout_events += 1
            else:
                stock[i] -= consumption
                # FIFO drain lots
                rem = consumption
                while rem > 0 and shelf_lots[i]:
                    qty0, age0 = shelf_lots[i][0]
                    take = min(qty0, rem)
                    shelf_lots[i][0] = (qty0 - take, age0)
                    rem -= take
                    if shelf_lots[i][0][0] <= 1e-9:
                        shelf_lots[i].pop(0)

            # 3. Age + expiry
            new_lots = []
            for (qty0, age0) in shelf_lots[i]:
                if age0 - 1 <= 0:
                    expiry_zar += qty0 * it.unit_price_zar
                    stock[i] = max(0.0, stock[i] - qty0)
                else:
                    new_lots.append((qty0, age0 - 1))
            shelf_lots[i] = new_lots

            # 4. Holding cost (daily)
            holding_zar += (stock[i] * it.unit_price_zar
                             * it.holding_cost_rate_per_year / 365.0)

            # 5. Determine (s, S) for the reorder decision
            if forecast_path is not None:
                # Forecast-driven: use mean of next L days of forecast
                L = int(round(it.lead_time_mean_days))
                end = min(SIM_DAYS, day + L)
                if end > day:
                    fc_window = forecast_path[day:end]
                    fc_mean = float(np.mean(fc_window))
                else:
                    fc_mean = float(np.mean(arrivals))
                share = SPECIALTY_SHARES.get(it.used_by_specialty, 0.10)
                mean_c_today = (fc_mean * share *
                                 it.probability_of_use_per_patient *
                                 it.units_per_patient_mean)
                s_, S_ = compute_ss(it, mean_c_today, alpha, beta, gamma)
            else:
                s_, S_ = static_ss[i]

            # 6. Reorder if stock + pending <= s
            in_pipeline = sum(q for (_, q) in pending[i])
            if stock[i] + in_pipeline <= s_:
                qty = max(1.0, S_ - stock[i] - in_pipeline)
                # Lead time draw
                if rng.random() < NON_PERFORMANCE_P:
                    L_draw = rng.lognormal(LEAD_TIME_MU,
                                             LEAD_TIME_SIGMA) * NON_PERFORMANCE_MULT
                elif rng.random() < PAYMENT_DELAY_P:
                    L_draw = rng.lognormal(LEAD_TIME_MU,
                                             LEAD_TIME_SIGMA) * PAYMENT_DELAY_MULT
                else:
                    L_draw = rng.lognormal(LEAD_TIME_MU, LEAD_TIME_SIGMA)
                arr = day + int(round(L_draw))
                pending[i].append((arr, qty))
                ordering_zar += it.ordering_cost_zar
                n_orders += 1

    total = holding_zar + ordering_zar + stockout_zar + expiry_zar
    return {
        "total_cost": total,
        "holding": holding_zar,
        "ordering": ordering_zar,
        "stockout": stockout_zar,
        "expiry": expiry_zar,
        "stockout_incidence_pct": 100.0 * stockout_days.mean(),
        "n_orders": n_orders,
        "n_stockout_events": n_stockout_events,
    }


def evaluate_config(
    items: list[Item],
    arrivals: np.ndarray,
    seeds: list[int],
    alpha: float, beta: float, gamma: float,
    forecast_path: np.ndarray | None = None,
) -> dict:
    """Mean across seeds + std-error."""
    rows = []
    for s in seeds:
        rows.append(simulate_one_seed(items, arrivals, s,
                                       alpha, beta, gamma,
                                       forecast_path=forecast_path))
    df = pd.DataFrame(rows)
    out = {f"{c}_mean": df[c].mean() for c in df.columns}
    out.update({f"{c}_std": df[c].std(ddof=1) for c in df.columns})
    return out


# ----------------------------------------------------------------------
# OPTIMISERS
# ----------------------------------------------------------------------
PARAM_BOX = {
    "alpha": (0.5, 1.5),
    "beta":  (0.3, 2.0),
    "gamma": (0.5, 2.5),
}


def grid_search(items, arrivals, seeds, n_per_axis=4):
    print(f"  grid: {n_per_axis}^3 = {n_per_axis**3} configs")
    alphas = np.linspace(*PARAM_BOX["alpha"], n_per_axis)
    betas  = np.linspace(*PARAM_BOX["beta"],  n_per_axis)
    gammas = np.linspace(*PARAM_BOX["gamma"], n_per_axis)
    trace = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                res = evaluate_config(items, arrivals, seeds, a, b, g)
                trace.append({"alpha": a, "beta": b, "gamma": g,
                               **res})
                print(f"    a={a:.2f} b={b:.2f} g={g:.2f} "
                       f"cost={res['total_cost_mean']:.0f}")
    trace = pd.DataFrame(trace)
    best = trace.loc[trace["total_cost_mean"].idxmin()]
    return best, trace


def random_search(items, arrivals, seeds, n_trials=25, rng_seed=2026):
    print(f"  random: {n_trials} trials")
    rng = np.random.default_rng(rng_seed)
    trace = []
    for t in range(n_trials):
        a = rng.uniform(*PARAM_BOX["alpha"])
        b = rng.uniform(*PARAM_BOX["beta"])
        g = rng.uniform(*PARAM_BOX["gamma"])
        res = evaluate_config(items, arrivals, seeds, a, b, g)
        trace.append({"trial": t, "alpha": a, "beta": b, "gamma": g,
                       **res})
        best_so_far = min(r["total_cost_mean"] for r in trace)
        print(f"    t{t:02d} a={a:.2f} b={b:.2f} g={g:.2f} "
               f"cost={res['total_cost_mean']:.0f}  best={best_so_far:.0f}")
    trace = pd.DataFrame(trace)
    best = trace.loc[trace["total_cost_mean"].idxmin()]
    return best, trace


def optuna_search(items, arrivals, seeds, n_trials=25):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"  optuna: {n_trials} trials")
    trace = []

    def objective(trial):
        a = trial.suggest_float("alpha", *PARAM_BOX["alpha"])
        b = trial.suggest_float("beta",  *PARAM_BOX["beta"])
        g = trial.suggest_float("gamma", *PARAM_BOX["gamma"])
        res = evaluate_config(items, arrivals, seeds, a, b, g)
        trace.append({"trial": trial.number,
                       "alpha": a, "beta": b, "gamma": g, **res})
        best_so_far = min(r["total_cost_mean"] for r in trace)
        print(f"    t{trial.number:02d} a={a:.2f} b={b:.2f} g={g:.2f} "
               f"cost={res['total_cost_mean']:.0f}  best={best_so_far:.0f}")
        return res["total_cost_mean"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=2026),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trace = pd.DataFrame(trace)
    best = trace.loc[trace["total_cost_mean"].idxmin()]
    return best, trace


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def t_half_width(std: float, n: int) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n))


def main():
    items = load_items()
    arrivals = load_arrivals_actual()
    forecast = load_arrivals_forecast()
    print(f"Loaded {len(items)} items, {len(arrivals)} days of arrivals, "
           f"{len(forecast)} days of forecast.")
    print(f"Seeds: {SEEDS}")
    print()

    summary = []

    # --- 1. Baseline (no optimisation) ---
    print("[1/5] Baseline — Silver 2017 textbook formula")
    base = evaluate_config(items, arrivals, SEEDS, 1.0, 1.0, 1.0)
    summary.append({
        "method": "Baseline", "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
        **base,
    })
    print(f"  baseline cost: {base['total_cost_mean']:.0f}\n")

    # --- 2. Forecast-driven (Chapter 6 XGBoost) ---
    print("[2/5] Forecast-driven — XGBoost time-varying (s, S)")
    fc = evaluate_config(items, arrivals, SEEDS, 1.0, 1.0, 1.0,
                          forecast_path=forecast)
    summary.append({
        "method": "Forecast-driven", "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
        **fc,
    })
    print(f"  forecast-driven cost: {fc['total_cost_mean']:.0f}\n")

    # --- 3. Grid search ---
    print("[3/5] Grid Search")
    g_best, g_trace = grid_search(items, arrivals, SEEDS, n_per_axis=3)
    g_trace.to_csv(OUT_DIR / "grid_trials.csv", index=False)
    summary.append({
        "method": "Grid Search",
        "alpha": g_best["alpha"], "beta": g_best["beta"],
        "gamma": g_best["gamma"],
        **{k: g_best[k] for k in g_best.index if k.endswith("_mean")
            or k.endswith("_std")},
    })
    print(f"  best: a={g_best['alpha']:.2f} b={g_best['beta']:.2f} "
           f"g={g_best['gamma']:.2f} cost={g_best['total_cost_mean']:.0f}\n")

    # --- 4. Random search ---
    print("[4/5] Random Search")
    r_best, r_trace = random_search(items, arrivals, SEEDS, n_trials=25)
    r_trace.to_csv(OUT_DIR / "random_trials.csv", index=False)
    summary.append({
        "method": "Random Search",
        "alpha": r_best["alpha"], "beta": r_best["beta"],
        "gamma": r_best["gamma"],
        **{k: r_best[k] for k in r_best.index if k.endswith("_mean")
            or k.endswith("_std")},
    })
    print(f"  best: a={r_best['alpha']:.2f} b={r_best['beta']:.2f} "
           f"g={r_best['gamma']:.2f} cost={r_best['total_cost_mean']:.0f}\n")

    # --- 5. Optuna TPE ---
    print("[5/5] Bayesian (Optuna TPE)")
    o_best, o_trace = optuna_search(items, arrivals, SEEDS, n_trials=25)
    o_trace.to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    summary.append({
        "method": "Bayesian (Optuna)",
        "alpha": o_best["alpha"], "beta": o_best["beta"],
        "gamma": o_best["gamma"],
        **{k: o_best[k] for k in o_best.index if k.endswith("_mean")
            or k.endswith("_std")},
    })
    print(f"  best: a={o_best['alpha']:.2f} b={o_best['beta']:.2f} "
           f"g={o_best['gamma']:.2f} cost={o_best['total_cost_mean']:.0f}\n")

    # --- 6. Forecast-driven + Optuna-tuned (combined headline) ---
    print("[6/6] Forecast-driven + Optuna-tuned (combined)")
    combo = evaluate_config(
        items, arrivals, SEEDS,
        float(o_best["alpha"]), float(o_best["beta"]), float(o_best["gamma"]),
        forecast_path=forecast,
    )
    summary.append({
        "method": "Forecast + Optuna",
        "alpha": o_best["alpha"], "beta": o_best["beta"],
        "gamma": o_best["gamma"],
        **combo,
    })
    print(f"  combined cost: {combo['total_cost_mean']:.0f}\n")

    # --- Write summary ---
    df = pd.DataFrame(summary)
    # 95% CI half-width
    df["cost_ci_half"] = df["total_cost_std"].apply(
        lambda s: t_half_width(s, N_SEEDS))
    df["cost_ci_low"]  = df["total_cost_mean"] - df["cost_ci_half"]
    df["cost_ci_high"] = df["total_cost_mean"] + df["cost_ci_half"]

    base_cost = df.loc[df["method"] == "Baseline", "total_cost_mean"].iloc[0]
    df["pct_reduction_vs_baseline"] = (
        (base_cost - df["total_cost_mean"]) / base_cost * 100.0
    )

    df.to_csv(OUT_DIR / "method_summary.csv", index=False)

    cost_decomp = df[["method", "holding_mean", "ordering_mean",
                       "stockout_mean", "expiry_mean", "total_cost_mean"]].copy()
    cost_decomp.to_csv(OUT_DIR / "cost_decomposition.csv", index=False)

    print()
    print("=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    show = df[["method", "alpha", "beta", "gamma",
                "total_cost_mean", "cost_ci_half",
                "stockout_incidence_pct_mean",
                "pct_reduction_vs_baseline"]].copy()
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"Wrote: {OUT_DIR / 'method_summary.csv'}")
    print(f"Wrote: {OUT_DIR / 'cost_decomposition.csv'}")


if __name__ == "__main__":
    main()
