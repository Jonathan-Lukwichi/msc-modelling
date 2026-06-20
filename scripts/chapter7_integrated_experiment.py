"""Chapter 7 — INTEGRATED hospital cost-minimisation experiment.

Aligned to the Chapter 3 mathematical formalisation:
  Eq 3.8   two-stage stochastic programme for (s, S)
  Eq 3.9   reorder point: s = D_bar * L + z_alpha * sigma_eps * sqrt(L)
  Eq 3.10  order-up-to: S* = argmin_S E[sum_t C_t(s, S)]
  Alg 8    Monte Carlo grid search for S over T-day horizon
  Eq 3.11  scheduling IP objective: min sum_{i,s} c_{i,s} x_{i,s}
  Eq 3.12  demand coverage
  Eq 3.13  max 45 weekly hours per BCEA s.9
  Eq 3.14  11-hour rest pair forbidden
  Eq 3.15  skills-mix
  Eq 3.16  budget
  Eq 3.17  hourly profile: lambda_{h,d} = y_d * p_{h,w(d)}
  Alg 9    IP construction with safety buffer kappa*sigma_eps*sqrt(h_s)

Total hospital cost minimised:
    C_total = C_holding + C_ordering + C_stockout + C_expiry
              + C_payroll + C_locum

Eight configurations compared on the SAME N_SEEDS via common random numbers.

Outputs to artefacts/chapter7/integrated/.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import milp, LinearConstraint, Bounds

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/BIBINBUSINESS/OneDrive/Desktop/dataAnalysis/chapter7_simulation")
OUT = ROOT / "artefacts" / "chapter7" / "integrated"
OUT.mkdir(parents=True, exist_ok=True)

ITEMS_CSV    = DATA / "items_master.csv"
STAFF_CSV    = DATA / "staff_master.csv"
ARRIVALS_CSV = DATA / "arrivals_test_block.csv"
FORECAST_TEST_CSV = ROOT / "artefacts" / "predictions" / "test" / "xgboost_rmse.csv"
# Validation predictions for residual-SD estimation
FORECAST_VAL_CSV  = ROOT / "artefacts" / "predictions" / "xgboost_rmse.csv"

SIM_DAYS = 396
N_SEEDS  = 5             # CRN — runtime constrained by IP per week
SEEDS    = list(range(42, 42 + N_SEEDS))

# Inventory — Modisakeng 2020 mechanism rates
LEAD_TIME_MU      = math.log(21.0)
LEAD_TIME_SIGMA   = (math.log(90.0) - math.log(21.0)) / 1.6449
NON_PERF_P        = 0.25
NON_PERF_MULT     = 3.0
PAY_DELAY_P       = 0.12
PAY_DELAY_MULT    = 2.0
CONS_PHI          = 1.4
Z_ALPHA           = 1.96         # 97.5% service level (Eq 3.9)
KAPPA             = 1.65         # safety buffer factor for scheduling (Alg 9)
MC_R              = 30           # MC replications for Algorithm 8 (R=1000 in chapter; reduced for runtime)
MC_T              = 90           # rolling horizon T=90 days (Eq 3.10)
S_GRID_N          = 6            # grid points for S* search

SPECIALTY_SHARES  = {"medicine": 0.73, "ortho": 0.17, "lowvol": 0.10}

# Scheduling (NDoH 2012 Norms + BCEA 1997)
NDOH_DAY_RATIO    = 6.0
NDOH_NIGHT_RATIO  = 10.0
SHIFT_HOURS       = 12.0
DAY_SHIFT_SHARE   = 0.65
NIGHT_SHIFT_SHARE = 0.35
MAX_WEEKLY_HOURS  = 45.0          # BCEA s.9 — LAWFUL cap (the headline)
ACTUAL_MAX_HOURS  = 58.0          # observed roster cap in chapter7 sim (overwork)
REST_HOURS        = 11.0
N_ACTIVE_NURSES   = 23            # 30 total - 7 vacant (Rispel 2014 vacancy 23%)
LOCUM_RATE        = 450.0
STOCKOUT_WORKAROUND_HOURS = 0.3  # coupling: stockout -> extra nurse-hours


# ----------------------------------------------------------------------
# DATA LOADERS
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


@dataclass
class Staff:
    staff_id: str
    category: str
    hourly_rate: float
    overtime_rate: float


def load_items():
    df = pd.read_csv(ITEMS_CSV)
    return [Item(
        item_id=r["item_id"], abc_class=r["abc_class"],
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
    ) for _, r in df.iterrows()]


def load_staff():
    df = pd.read_csv(STAFF_CSV)
    return [Staff(
        staff_id=r["staff_id"], category=r["category"],
        hourly_rate=float(r["hourly_rate_regular_zar"]),
        overtime_rate=float(r["hourly_rate_overtime_weekday_zar"]),
    ) for _, r in df.iterrows()]


def load_arrivals():
    df = pd.read_csv(ARRIVALS_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True).head(SIM_DAYS)


def load_forecast():
    df = pd.read_csv(FORECAST_TEST_CSV, parse_dates=["date"])
    return df.sort_values("date")["predicted"].astype(float).values[:SIM_DAYS]


def forecast_residual_sd() -> float:
    """sigma_eps for Eq 3.9 — SD of XGBoost forecast residuals on val block."""
    df = pd.read_csv(FORECAST_VAL_CSV, parse_dates=["date"])
    if "block" in df.columns:
        df = df[df["block"] == "val"]
    if "predicted" in df.columns and "actual" in df.columns:
        resid = df["actual"].astype(float) - df["predicted"].astype(float)
        return float(np.std(resid, ddof=1))
    return 9.35  # fallback: XGBoost val RMSE


# ----------------------------------------------------------------------
# ALGORITHM 8 — (s, S) optimisation by MC grid search
# ----------------------------------------------------------------------
def alg8_optimise_sS(item: Item, D_bar: float, sigma_eps: float,
                       rng_seed: int = 2026,
                       use_grid_S: bool = True) -> tuple[float, float]:
    """Implement Eq 3.9-3.10 + Algorithm 8.

    D_bar       : mean daily demand forecast for this item (units/day)
    sigma_eps   : SD of arrivals forecast residuals (in patients/day)
                  scaled to item units below.
    use_grid_S  : if False, return the textbook EOQ for S (faster, no MC).
    """
    L = item.lead_time_mean_days

    # Step 1: scale residual SD from patients to item units
    share = SPECIALTY_SHARES.get(item.used_by_specialty, 0.10)
    unit_factor = (share * item.probability_of_use_per_patient
                    * item.units_per_patient_mean)
    sigma_item = sigma_eps * unit_factor

    # Step 2: reorder point (Eq 3.9)
    s = D_bar * L + Z_ALPHA * sigma_item * math.sqrt(L)

    # EOQ as anchor for the S-grid
    annual = D_bar * 365
    if item.unit_price_zar > 0 and item.holding_cost_rate_per_year > 0:
        eoq = math.sqrt(2 * annual * item.ordering_cost_zar
                         / (item.holding_cost_rate_per_year
                             * item.unit_price_zar))
    else:
        eoq = D_bar * 30
    eoq = max(eoq, 1.0)

    if not use_grid_S:
        return s, s + eoq   # baseline path: S = s + EOQ, no MC

    # Step 3: MC grid search over S candidates
    rng = np.random.default_rng(rng_seed + hash(item.item_id) % 10_000)
    S_grid = np.linspace(0.4 * eoq, 2.0 * eoq, S_GRID_N) + s
    best_S = float(S_grid[0])
    best_cost = float("inf")
    for S_cand in S_grid:
        total = 0.0
        for _ in range(MC_R):
            # Sample demand path from forecast residuals + mean
            demand_path = np.maximum(
                D_bar + rng.normal(0.0, sigma_item, MC_T), 0.0)
            stock_t = s + 0.5 * (S_cand - s)
            cost = 0.0
            pending_qty = 0.0
            pending_arr = -1
            for t in range(MC_T):
                if pending_arr == t:
                    stock_t += pending_qty
                    pending_qty = 0.0
                    pending_arr = -1
                d = demand_path[t]
                if d > stock_t:
                    short = d - stock_t
                    cost += short * item.stockout_penalty_zar
                    stock_t = 0.0
                else:
                    stock_t -= d
                cost += (stock_t * item.unit_price_zar
                          * item.holding_cost_rate_per_year / 365.0)
                if stock_t <= s and pending_arr < 0:
                    pending_qty = max(1.0, S_cand - stock_t)
                    pending_arr = t + int(round(L))
                    cost += item.ordering_cost_zar
            total += cost
        mean_cost = total / MC_R
        if mean_cost < best_cost:
            best_cost = mean_cost
            best_S = float(S_cand)
    return s, best_S


# ----------------------------------------------------------------------
# ALGORITHM 9 — scheduling IP via scipy.optimize.milp
# ----------------------------------------------------------------------
def build_hourly_profile(arrivals_df: pd.DataFrame) -> np.ndarray:
    """Eq 3.17 — historical day-of-week × hour share table.

    Since we only have daily totals, we synthesise an hourly profile from
    a literature-anchored stylised shape: day-shift carries 65% of
    arrivals, night-shift 35%, uniform within each 12-h shift.
    """
    # 2 shifts (day, night); proportions per day-of-week stored as 7x2.
    profile = np.tile(
        np.array([[DAY_SHIFT_SHARE, NIGHT_SHIFT_SHARE]]),
        (7, 1)
    )
    return profile


def alg9_weekly_roster(
    week_forecast: np.ndarray,
    staff: list[Staff],
    sigma_eps: float,
    profile: np.ndarray,
    workaround_hours_per_day: np.ndarray,
    max_weekly_hours: float = MAX_WEEKLY_HOURS,
) -> dict:
    """Build and solve the IP (Eq 3.11-3.16) for a single week.

    Returns dict with payroll_cost, locum_cost, lawful coverage and
    diagnostics (mean weekly hours, BCEA breaches, required nurses).

    Decision variables: x_{i,s} in {0, 1}, i in staff, s in {day_d, night_d}
    for d in 0..6.  14 shifts per week.
    """
    n_staff = len(staff)
    days = 7
    shifts_per_day = 2     # day, night
    n_shifts = days * shifts_per_day
    n_vars = n_staff * n_shifts

    # ----- Demand per shift (Eq 3.12, Algorithm 9 step 1) -----
    d_min = np.zeros(n_shifts)
    for d in range(days):
        if d >= len(week_forecast):
            break
        y_d = week_forecast[d]
        wd = d % 7
        # Hourly lambda via Eq 3.17 — collapsed to per-shift here
        lam_day   = y_d * profile[wd, 0]   # day-shift patients
        lam_night = y_d * profile[wd, 1]
        # Add coupling workaround hours expressed as extra "patients"
        # (workaround_hours / nurse_ratio_per_patient_hour)
        # patients_equiv = workaround_hours / SHIFT_HOURS * NDOH_DAY_RATIO
        wk = workaround_hours_per_day[d] if d < len(workaround_hours_per_day) else 0
        lam_day   += wk * DAY_SHIFT_SHARE   * NDOH_DAY_RATIO / SHIFT_HOURS
        lam_night += wk * NIGHT_SHIFT_SHARE * NDOH_NIGHT_RATIO / SHIFT_HOURS
        # NDoH ratio gives nurse count needed; add safety buffer (Alg 9).
        # Safety buffer = kappa * sigma_eps * sqrt(shift_share) / ratio
        # (extra nurses to absorb forecast residual SD in the shift demand).
        nurses_day   = (lam_day   / NDOH_DAY_RATIO
                         + KAPPA * sigma_eps
                           * math.sqrt(DAY_SHIFT_SHARE) / NDOH_DAY_RATIO)
        nurses_night = (lam_night / NDOH_NIGHT_RATIO
                         + KAPPA * sigma_eps
                           * math.sqrt(NIGHT_SHIFT_SHARE) / NDOH_NIGHT_RATIO)
        d_min[d * 2]     = max(1.0, math.ceil(nurses_day))
        d_min[d * 2 + 1] = max(1.0, math.ceil(nurses_night))

    # ----- Cost coefficients c_{i,s} -----
    # Regular hours = SHIFT_HOURS; cost = hourly_rate * 12
    c = np.zeros(n_vars)
    for i, st in enumerate(staff):
        for s_idx in range(n_shifts):
            c[i * n_shifts + s_idx] = st.hourly_rate * SHIFT_HOURS

    # ----- Constraint A: demand coverage (Eq 3.12) sum_i x_{i,s} >= d_s -----
    A_cov = np.zeros((n_shifts, n_vars))
    for s_idx in range(n_shifts):
        for i in range(n_staff):
            A_cov[s_idx, i * n_shifts + s_idx] = 1.0
    cov_lb = d_min
    cov_ub = np.full(n_shifts, np.inf)

    # ----- Constraint B: max weekly hours (Eq 3.13) sum_s h*x <= 45 -----
    A_hrs = np.zeros((n_staff, n_vars))
    for i in range(n_staff):
        for s_idx in range(n_shifts):
            A_hrs[i, i * n_shifts + s_idx] = SHIFT_HOURS
    hrs_lb = np.zeros(n_staff)
    hrs_ub = np.full(n_staff, max_weekly_hours)

    # ----- Constraint C: 11-hour rest pairs (Eq 3.14) x_{i,s} + x_{i,s'} <= 1
    # Rest-pair: a day-shift on day d (ends 18:00) and night-shift on day d
    # (starts 18:00) violate the 11-hour rest rule.  Also night-shift on day d
    # (ends 06:00 next day) and day-shift on day d+1 (starts 06:00) violate.
    rest_pairs = []
    for d in range(days):
        rest_pairs.append((d * 2, d * 2 + 1))            # day -> night same day
        if d + 1 < days:
            rest_pairs.append((d * 2 + 1, (d + 1) * 2))  # night -> day next day
    n_rest = len(rest_pairs)
    A_rest = np.zeros((n_staff * n_rest, n_vars))
    rest_ub = np.ones(n_staff * n_rest)
    rest_lb = np.full(n_staff * n_rest, -np.inf)
    row = 0
    for i in range(n_staff):
        for (s1, s2) in rest_pairs:
            A_rest[row, i * n_shifts + s1] = 1.0
            A_rest[row, i * n_shifts + s2] = 1.0
            row += 1

    # Stack constraints
    A = np.vstack([A_cov, A_hrs, A_rest])
    lb = np.concatenate([cov_lb, hrs_lb, rest_lb])
    ub = np.concatenate([cov_ub, hrs_ub, rest_ub])

    # Integrality: all binary
    integrality = np.ones(n_vars)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    constraints = LinearConstraint(A, lb=lb, ub=ub)

    try:
        res = milp(c=c, constraints=constraints,
                    integrality=integrality, bounds=bounds,
                    options={"time_limit": 5.0})
    except Exception:
        res = None

    if res is None or not res.success:
        return _greedy_fallback(staff, d_min, max_weekly_hours)

    x = res.x.reshape(n_staff, n_shifts)
    payroll = float((x * SHIFT_HOURS *
                      np.array([s.hourly_rate for s in staff]).reshape(-1, 1)
                      ).sum())
    assigned = x.sum(axis=0)
    locum_hours = 0.0
    total_req = 0.0
    total_cov = 0.0
    for s_idx in range(n_shifts):
        req = d_min[s_idx]
        cov = assigned[s_idx]
        total_req += req
        total_cov += min(cov, req)
        if cov < req:
            locum_hours += (req - cov) * SHIFT_HOURS
    coverage = 100.0 * total_cov / total_req if total_req > 0 else 100.0

    # Diagnostics — BCEA + overwork
    weekly_hours_per_staff = x.sum(axis=1) * SHIFT_HOURS    # n_staff vector
    active_mask = weekly_hours_per_staff > 1e-3             # who actually worked
    n_active = int(active_mask.sum())
    mean_weekly_hours = (float(weekly_hours_per_staff[active_mask].mean())
                          if n_active > 0 else 0.0)
    bcea_breaches = int((weekly_hours_per_staff > MAX_WEEKLY_HOURS + 1e-6).sum())
    # Required nurses at lawful (45h) cap
    required_hours_total = total_req * SHIFT_HOURS
    required_nurses_lawful = math.ceil(required_hours_total / MAX_WEEKLY_HOURS)

    return {
        "payroll": payroll,
        "locum_cost": locum_hours * LOCUM_RATE,
        "locum_hours": locum_hours,
        "coverage_pct": coverage,
        "n_active": n_active,
        "mean_weekly_hours": mean_weekly_hours,
        "bcea_breaches": bcea_breaches,
        "required_hours": required_hours_total,
        "required_nurses_lawful": required_nurses_lawful,
        "total_assigned_hours": float(weekly_hours_per_staff.sum()),
    }


def _greedy_fallback(staff: list[Staff], d_min: np.ndarray,
                       max_h: float = MAX_WEEKLY_HOURS) -> dict:
    """Used when MILP fails."""
    n_shifts = len(d_min)
    staff_sorted = sorted(staff, key=lambda s: s.hourly_rate)
    weekly_hours = {s.staff_id: 0.0 for s in staff_sorted}
    payroll = 0.0
    total_req = 0.0
    total_cov = 0.0
    locum_hours = 0.0
    for s_idx in range(n_shifts):
        req = d_min[s_idx]
        total_req += req
        cov = 0
        for st in staff_sorted:
            if cov >= req:
                break
            if weekly_hours[st.staff_id] + SHIFT_HOURS <= max_h:
                weekly_hours[st.staff_id] += SHIFT_HOURS
                payroll += st.hourly_rate * SHIFT_HOURS
                cov += 1
        total_cov += min(cov, req)
        if cov < req:
            locum_hours += (req - cov) * SHIFT_HOURS
    hrs = np.array(list(weekly_hours.values()))
    active = hrs > 1e-3
    return {
        "payroll": payroll,
        "locum_cost": locum_hours * LOCUM_RATE,
        "locum_hours": locum_hours,
        "coverage_pct": 100.0 * total_cov / total_req if total_req > 0 else 100.0,
        "n_active": int(active.sum()),
        "mean_weekly_hours": float(hrs[active].mean()) if active.any() else 0.0,
        "bcea_breaches": int((hrs > MAX_WEEKLY_HOURS + 1e-6).sum()),
        "required_hours": total_req * SHIFT_HOURS,
        "required_nurses_lawful": math.ceil(total_req * SHIFT_HOURS / MAX_WEEKLY_HOURS),
        "total_assigned_hours": float(hrs.sum()),
    }


# ----------------------------------------------------------------------
# INTEGRATED SIMULATOR
# ----------------------------------------------------------------------
def simulate_one_seed(
    items: list[Item], staff: list[Staff],
    arrivals: np.ndarray, seed: int,
    sigma_eps: float,
    use_grid_S: bool = True,
    inv_forecast: np.ndarray | None = None,
    sched_forecast: np.ndarray | None = None,
    profile: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    mean_arrivals = arrivals.mean()
    if profile is None:
        profile = np.tile(np.array([[DAY_SHIFT_SHARE, NIGHT_SHIFT_SHARE]]),
                            (7, 1))

    # ------------ Set inventory (s, S) per item ------------
    item_D = []
    for it in items:
        share = SPECIALTY_SHARES.get(it.used_by_specialty, 0.10)
        D = max(mean_arrivals * share * it.probability_of_use_per_patient
                 * it.units_per_patient_mean, 0.001)
        item_D.append(D)
    static_ss = [
        alg8_optimise_sS(it, item_D[i], sigma_eps,
                          rng_seed=seed, use_grid_S=use_grid_S)
        for i, it in enumerate(items)
    ]

    # Inventory state
    n_items = len(items)
    stock = np.array([it.initial_stock_units for it in items], dtype=float)
    pending = [[] for _ in range(n_items)]
    shelf_lots = [[(it.initial_stock_units,
                     int(it.shelf_life_months * 30))]
                   for it in items]

    holding_zar = 0.0
    ordering_zar = 0.0
    stockout_zar = 0.0
    expiry_zar = 0.0
    n_orders = 0
    n_stockout_events = 0
    stockout_days = np.zeros(SIM_DAYS)
    daily_stockouts = np.zeros(SIM_DAYS, dtype=int)

    # ------------ Day-by-day inventory simulation ------------
    for day in range(SIM_DAYS):
        a_today = arrivals[day]
        # Receive pending
        for i in range(n_items):
            new_pend = []
            for arr_day, qty in pending[i]:
                if arr_day <= day:
                    stock[i] += qty
                    shelf_lots[i].append((qty,
                                            int(items[i].shelf_life_months * 30)))
                else:
                    new_pend.append((arr_day, qty))
            pending[i] = new_pend

        for i, it in enumerate(items):
            share = SPECIALTY_SHARES.get(it.used_by_specialty, 0.10)
            exp_c = max(a_today * share * it.probability_of_use_per_patient
                         * it.units_per_patient_mean, 0.001)
            phi = CONS_PHI
            p_nb = phi / (phi + exp_c)
            cons = float(rng.negative_binomial(phi, p_nb))
            if cons > stock[i]:
                short = cons - stock[i]
                stockout_zar += short * it.stockout_penalty_zar
                stock[i] = 0.0
                stockout_days[day] = 1.0
                n_stockout_events += 1
                daily_stockouts[day] += 1
            else:
                stock[i] -= cons
                rem = cons
                while rem > 0 and shelf_lots[i]:
                    q0, a0 = shelf_lots[i][0]
                    take = min(q0, rem)
                    shelf_lots[i][0] = (q0 - take, a0)
                    rem -= take
                    if shelf_lots[i][0][0] <= 1e-9:
                        shelf_lots[i].pop(0)
            new_lots = []
            for q0, a0 in shelf_lots[i]:
                if a0 - 1 <= 0:
                    expiry_zar += q0 * it.unit_price_zar
                    stock[i] = max(0.0, stock[i] - q0)
                else:
                    new_lots.append((q0, a0 - 1))
            shelf_lots[i] = new_lots
            holding_zar += (stock[i] * it.unit_price_zar
                             * it.holding_cost_rate_per_year / 365.0)

            # (s, S) decision — forecast-driven recomputes daily
            if inv_forecast is not None:
                L = int(round(it.lead_time_mean_days))
                end = min(SIM_DAYS, day + L)
                D_fc = (float(np.mean(inv_forecast[day:end])) * share
                         * it.probability_of_use_per_patient
                         * it.units_per_patient_mean
                         if end > day else item_D[i])
                D_fc = max(D_fc, 0.001)
                s_, S_ = alg8_optimise_sS(it, D_fc, sigma_eps,
                                            rng_seed=seed + day,
                                            use_grid_S=False)
            else:
                s_, S_ = static_ss[i]

            in_pipe = sum(q for _, q in pending[i])
            if stock[i] + in_pipe <= s_:
                qty = max(1.0, S_ - stock[i] - in_pipe)
                if rng.random() < NON_PERF_P:
                    Lr = rng.lognormal(LEAD_TIME_MU,
                                         LEAD_TIME_SIGMA) * NON_PERF_MULT
                elif rng.random() < PAY_DELAY_P:
                    Lr = rng.lognormal(LEAD_TIME_MU,
                                         LEAD_TIME_SIGMA) * PAY_DELAY_MULT
                else:
                    Lr = rng.lognormal(LEAD_TIME_MU, LEAD_TIME_SIGMA)
                pending[i].append((day + int(round(Lr)), qty))
                ordering_zar += it.ordering_cost_zar
                n_orders += 1

    # ------------ Weekly scheduling IP (Algorithm 9) ------------
    # Solve twice per week:
    #   (a) LAWFUL — 45h cap (BCEA s.9), this is the chapter's headline
    #   (b) ACTUAL — 58h cap (observed overwork roster), comparison only
    payroll_zar = 0.0
    locum_zar = 0.0
    lawful_cov_sum = 0.0
    actual_cov_sum = 0.0
    mean_h_sum = 0.0
    breach_sum = 0.0
    req_nurses_sum = 0.0
    n_weeks_count = 0
    n_weeks = SIM_DAYS // 7

    for w in range(n_weeks):
        start = w * 7
        end = start + 7
        if sched_forecast is not None:
            week_fc = sched_forecast[start:end]
        else:
            week_fc = np.full(7, mean_arrivals)
        wk_workaround = daily_stockouts[max(0, start - 1):end - 1].astype(float)
        if len(wk_workaround) < 7:
            wk_workaround = np.pad(wk_workaround,
                                     (0, 7 - len(wk_workaround)))
        wk_workaround = wk_workaround * STOCKOUT_WORKAROUND_HOURS

        # Lawful IP — 45h cap (headline)
        r_law = alg9_weekly_roster(week_fc, staff, sigma_eps,
                                     profile, wk_workaround,
                                     max_weekly_hours=MAX_WEEKLY_HOURS)
        # Actual IP — 58h cap (overtime allowed, comparison)
        r_act = alg9_weekly_roster(week_fc, staff, sigma_eps,
                                     profile, wk_workaround,
                                     max_weekly_hours=ACTUAL_MAX_HOURS)

        # Cost realised is from the ACTUAL roster (this is what the
        # hospital pays today) — overtime hours billed at overtime rate
        # (approximated as 1.5x for hours above 45)
        ot_share = max(0.0, r_act["mean_weekly_hours"] - MAX_WEEKLY_HOURS) \
                    / max(r_act["mean_weekly_hours"], 1.0)
        payroll_zar += r_act["payroll"] * (1.0 + 0.5 * ot_share)
        locum_zar   += r_act["locum_cost"]
        lawful_cov_sum += r_law["coverage_pct"]
        actual_cov_sum += r_act["coverage_pct"]
        mean_h_sum += r_act["mean_weekly_hours"]
        breach_sum += r_act["bcea_breaches"]
        req_nurses_sum += r_law["required_nurses_lawful"]
        n_weeks_count += 1

    total_inv = holding_zar + ordering_zar + stockout_zar + expiry_zar
    total_sch = payroll_zar + locum_zar
    lawful_cov = lawful_cov_sum / max(1, n_weeks_count)
    actual_cov = actual_cov_sum / max(1, n_weeks_count)
    mean_h_per_nurse = mean_h_sum / max(1, n_weeks_count)
    avg_required_nurses_lawful = req_nurses_sum / max(1, n_weeks_count)
    staffing_shortfall = max(0.0,
                              avg_required_nurses_lawful - N_ACTIVE_NURSES)
    overwork_pct = mean_h_per_nurse / MAX_WEEKLY_HOURS * 100.0
    bcea_breaches_per_nurse = breach_sum / max(1, n_weeks_count) / N_ACTIVE_NURSES
    return {
        "total_cost":      total_inv + total_sch,
        "inventory_cost":  total_inv,
        "scheduling_cost": total_sch,
        "holding":     holding_zar,
        "ordering":    ordering_zar,
        "stockout":    stockout_zar,
        "expiry":      expiry_zar,
        "payroll":     payroll_zar,
        "locum":       locum_zar,
        "n_stockout_events": n_stockout_events,
        "stockout_incidence_pct": 100.0 * stockout_days.mean(),
        "lawful_coverage_pct":    lawful_cov,
        "actual_coverage_pct":    actual_cov,
        "mean_weekly_hours":      mean_h_per_nurse,
        "overwork_pct":           overwork_pct,
        "bcea_breaches_per_nurse_wk": bcea_breaches_per_nurse,
        "required_nurses_lawful": avg_required_nurses_lawful,
        "staffing_shortfall_nurses": staffing_shortfall,
    }


def evaluate(items, staff, arrivals, seeds, sigma_eps, **kw):
    rows = [simulate_one_seed(items, staff, arrivals, s,
                               sigma_eps, **kw)
            for s in seeds]
    df = pd.DataFrame(rows)
    out = {f"{c}_mean": df[c].mean() for c in df.columns}
    out.update({f"{c}_std": df[c].std(ddof=1) for c in df.columns})
    return out


def t_half(std, n):
    return float(stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n))


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    items = load_items()
    staff = load_staff()
    arr_df = load_arrivals()
    arr = arr_df["actual_arrivals"].astype(float).values[:SIM_DAYS]
    fc = load_forecast()
    sigma_eps = forecast_residual_sd()
    print(f"Items: {len(items)}, Staff: {len(staff)}")
    print(f"Arrivals: {len(arr)} days, mean = {arr.mean():.1f}/day")
    print(f"Forecast (XGBoost): {len(fc)} days, mean = {fc.mean():.1f}/day")
    print(f"sigma_eps (forecast residual SD): {sigma_eps:.2f}")
    print(f"Seeds (CRN): {SEEDS}")
    print(f"MC reps R = {MC_R}, MC horizon T = {MC_T}, S grid n = {S_GRID_N}")
    print()

    profile = np.tile(np.array([[DAY_SHIFT_SHARE, NIGHT_SHIFT_SHARE]]),
                        (7, 1))

    # 6 configurations
    methods = [
        ("Baseline",
            dict(use_grid_S=False, inv_forecast=None,
                  sched_forecast=None, profile=profile)),
        ("Forecast -> inventory only",
            dict(use_grid_S=False, inv_forecast=fc,
                  sched_forecast=None, profile=profile)),
        ("Forecast -> scheduling only",
            dict(use_grid_S=False, inv_forecast=None,
                  sched_forecast=fc, profile=profile)),
        ("Forecast -> BOTH",
            dict(use_grid_S=False, inv_forecast=fc,
                  sched_forecast=fc, profile=profile)),
        ("Alg 8 (MC grid S) + hist roster",
            dict(use_grid_S=True, inv_forecast=None,
                  sched_forecast=None, profile=profile)),
        ("Alg 8 + Forecast roster (Eq 3.8-3.17 full chain)",
            dict(use_grid_S=True, inv_forecast=fc,
                  sched_forecast=fc, profile=profile)),
    ]

    summary = []
    for i, (name, kw) in enumerate(methods, 1):
        print(f"[{i}/{len(methods)}] {name}")
        res = evaluate(items, staff, arr, SEEDS, sigma_eps, **kw)
        summary.append({
            "method": name,
            "uses_alg8":  kw["use_grid_S"],
            "uses_inv_forecast":  kw["inv_forecast"] is not None,
            "uses_sched_forecast": kw["sched_forecast"] is not None,
            **res,
        })
        print(f"   inventory:  R{res['inventory_cost_mean']:>15,.0f}")
        print(f"   scheduling: R{res['scheduling_cost_mean']:>15,.0f}")
        print(f"   TOTAL:      R{res['total_cost_mean']:>15,.0f}")
        print(f"   LAWFUL coverage:  {res['lawful_coverage_pct_mean']:.1f}%  "
               f"(headline)")
        print(f"   ACTUAL coverage:  {res['actual_coverage_pct_mean']:.1f}%  "
               f"(with overtime, propped-up)")
        print(f"   mean weekly hrs:  {res['mean_weekly_hours_mean']:.1f}h "
               f"({res['overwork_pct_mean']:.0f}% of 45h)")
        print(f"   BCEA breaches/nurse/wk: {res['bcea_breaches_per_nurse_wk_mean']:.2f}")
        print(f"   staffing shortfall at lawful hrs: "
               f"{res['staffing_shortfall_nurses_mean']:.1f} nurses "
               f"(need ~{res['required_nurses_lawful_mean']:.0f}, "
               f"have {N_ACTIVE_NURSES} active)")
        print(f"   stockouts: {res['stockout_incidence_pct_mean']:.1f}%")
        print()

    df = pd.DataFrame(summary)
    df["total_ci_half"] = df["total_cost_std"].apply(
        lambda s: t_half(s, N_SEEDS))
    df["total_ci_low"]  = df["total_cost_mean"] - df["total_ci_half"]
    df["total_ci_high"] = df["total_cost_mean"] + df["total_ci_half"]
    base = df.loc[df["method"] == "Baseline", "total_cost_mean"].iloc[0]
    df["pct_reduction"] = (base - df["total_cost_mean"]) / base * 100

    df.to_csv(OUT / "integrated_summary.csv", index=False)

    print("=" * 100)
    print("INTEGRATED HOSPITAL COST RESULTS")
    print("=" * 100)
    show = df[["method", "total_cost_mean", "pct_reduction",
                "lawful_coverage_pct_mean", "actual_coverage_pct_mean",
                "mean_weekly_hours_mean", "overwork_pct_mean",
                "staffing_shortfall_nurses_mean",
                "stockout_incidence_pct_mean"]].copy()
    show.columns = ["method", "total_zar", "pct_red",
                     "lawful_cov", "actual_cov",
                     "mean_h", "overwork_pct",
                     "shortfall", "stockouts"]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.0f}"))
    print()
    print(f"Wrote: {OUT / 'integrated_summary.csv'}")


if __name__ == "__main__":
    main()
