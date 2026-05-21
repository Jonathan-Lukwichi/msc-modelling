"""Sample-importance weights for covariate-shift correction.

References
----------
- Gretton, Smola, Huang, Schmittfull, Borgwardt & Schölkopf (2009)
  "Covariate Shift by Kernel Mean Matching." in *Dataset Shift in Machine
  Learning* (Quiñonero-Candela et al., eds.), MIT Press.
- Sugiyama, Krauledat & Müller (2007) "Covariate shift adaptation by
  importance weighted cross validation." *JMLR* 8:985-1005.
- Yamada, Suzuki, Kanamori, Hachiya & Sugiyama (2013) "Relative density-
  ratio estimation for robust distribution comparison." *Neural
  Computation* 25(5):1324-1370.  (RuLSIF.)

Two estimators are provided:

  - ``kmm_weights``  -- Kernel Mean Matching with a Gaussian kernel and
                        median-distance bandwidth (Gretton 2009 default).
                        Solves a quadratic program; requires ``cvxopt``
                        but falls back to a closed-form approximation
                        when cvxopt is not installed.
  - ``rulsif_weights`` -- Relative unconstrained least-squares importance
                          fitting via the ``densratio`` package.
                          Bounded weights via the relative parameter
                          ``alpha`` (default 0.1).

Both produce non-negative weights normalised to sum to ``n_train`` so
they slot into XGBoost's ``sample_weight=`` or PyTorch's per-sample loss
multiplier without changing the gradient's effective magnitude.

These functions are the building blocks for ``sample_weight_fn`` passed
to ``RollingForecaster`` in Prompt 7's drift-aware refit script.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _median_pairwise_distance(X: np.ndarray) -> float:
    """Median heuristic for the Gaussian kernel bandwidth (Gretton 2009)."""
    n = min(len(X), 500)  # cap for tractability
    if n < 2:
        return 1.0
    idx = np.random.default_rng(0).choice(len(X), size=n, replace=False)
    sub = X[idx]
    # All pairwise sq-distances
    sq = (
        np.sum(sub ** 2, axis=1, keepdims=True)
        + np.sum(sub ** 2, axis=1, keepdims=True).T
        - 2.0 * sub @ sub.T
    )
    upper = sq[np.triu_indices_from(sq, k=1)]
    upper = upper[upper > 0]
    if len(upper) == 0:
        return 1.0
    return float(np.sqrt(np.median(upper)))


def _gaussian_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel matrix K[i, j] = exp(-||A_i - B_j||^2 / (2 * sigma^2))."""
    sq = (
        np.sum(A ** 2, axis=1, keepdims=True)
        + np.sum(B ** 2, axis=1, keepdims=True).T
        - 2.0 * A @ B.T
    )
    return np.exp(-sq / (2.0 * sigma ** 2))


def _as_matrix(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    return np.asarray(X, dtype=float)


def _normalise(w: np.ndarray, n: int) -> np.ndarray:
    """Clip negatives, then rescale so mean(w) ~= 1 (sum w = n)."""
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones(n)
    return w * (n / s)


# ---------------------------------------------------------------------------
# Kernel Mean Matching (Gretton et al. 2009)
# ---------------------------------------------------------------------------

def kmm_weights(
    X_train,
    X_recent,
    sigma: Literal["median"] | float = "median",
    B: float = 1000.0,
    eps: Optional[float] = None,
) -> np.ndarray:
    """KMM weights for X_train so its kernel mean matches X_recent's.

    Solves a closed-form approximation (regularised normal equations)
    when ``cvxopt`` is not available, otherwise a constrained QP per
    Gretton (2009). Output is an array of length ``len(X_train)``,
    non-negative, normalised so the mean is 1.

    Parameters
    ----------
    X_train : array-like, shape (n_train, d)
    X_recent : array-like, shape (n_recent, d) -- sample from the
        target distribution (e.g., the last 90 days of train + val).
    sigma : Gaussian bandwidth. "median" picks via the median heuristic.
    B : upper-bound on individual weights in the constrained QP.
    eps : ridge regularisation for the closed-form fallback. None -> auto.
    """
    Xtr = _as_matrix(X_train)
    Xre = _as_matrix(X_recent)
    n = len(Xtr)
    m = len(Xre)

    if sigma == "median":
        sigma = _median_pairwise_distance(np.vstack([Xtr, Xre]))

    K = _gaussian_kernel(Xtr, Xtr, sigma)
    kappa = (float(n) / m) * _gaussian_kernel(Xtr, Xre, sigma).sum(axis=1)

    try:
        import cvxopt
        from cvxopt import solvers, matrix
        solvers.options["show_progress"] = False
        P = matrix(K.astype(float))
        q = matrix(-kappa.astype(float))
        # Constraints:  0 <= w <= B  AND  | sum(w) - n | <= n * eps
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), np.full(n, B)]))
        sol = solvers.qp(P, q, G, h)
        w = np.asarray(sol["x"]).ravel()
    except (ImportError, Exception):
        if eps is None:
            eps = 1e-3 * float(np.trace(K)) / n
        w = np.linalg.solve(K + eps * np.eye(n), kappa)

    return _normalise(w, n)


# ---------------------------------------------------------------------------
# RuLSIF (Yamada et al. 2013) via the densratio package
# ---------------------------------------------------------------------------

def rulsif_weights(
    X_train,
    X_recent,
    alpha: float = 0.1,
    sigma_range: Optional[list[float]] = None,
    lambda_range: Optional[list[float]] = None,
) -> np.ndarray:
    """Relative density-ratio weights via ``densratio.densratio``.

    The relative density ratio is r_alpha(x) = p_recent(x) /
    (alpha * p_recent(x) + (1 - alpha) * p_train(x)), bounded by 1/alpha
    for alpha > 0 -- avoids the unbounded-weight pathology of vanilla
    importance sampling under shift.

    If the ``densratio`` package is not installed, falls back to KMM
    (logs a warning to stderr).
    """
    try:
        from densratio import densratio
    except ImportError:
        import sys
        print(
            "WARN: densratio not installed; falling back to KMM. "
            "pip install densratio>=0.3",
            file=sys.stderr,
        )
        return kmm_weights(X_train, X_recent)

    Xtr = _as_matrix(X_train)
    Xre = _as_matrix(X_recent)
    result = densratio(
        Xre, Xtr,                      # densratio signature: (numerator, denominator)
        alpha=alpha,
        sigma_range=sigma_range or "auto",
        lambda_range=lambda_range or "auto",
        kernel_num=min(100, len(Xtr)),
        verbose=False,
    )
    # densratio's compute_density_ratio(x) gives r(x) for any x.
    w = np.asarray(result.compute_density_ratio(Xtr), dtype=float).ravel()
    return _normalise(w, len(Xtr))


__all__ = ["kmm_weights", "rulsif_weights"]
