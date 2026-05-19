"""Rolling-origin expanding-window cross-validation (§3.6.1).

Initial training window covers the earliest portion of the train block. Forecast
horizon h = 7 days. After evaluation, the training window expands by 7 days and
the process repeats. No random shuffling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """A single rolling-origin fold."""
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    origin: pd.Timestamp


def rolling_origin(
    index: pd.DatetimeIndex,
    horizon_days: int = 7,
    step_days: int = 7,
    min_train_days: int = 365,
) -> Iterator[Fold]:
    """Yield expanding-window folds with a fixed horizon.

    Parameters
    ----------
    index : sorted DatetimeIndex of the available training series.
    horizon_days : test window length (days).
    step_days : advance step (days).
    min_train_days : initial training window size (days).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("rolling_origin expects a pandas DatetimeIndex")
    n = len(index)
    if n < min_train_days + horizon_days:
        raise ValueError(
            f"Not enough data: {n} < {min_train_days + horizon_days} "
            f"(min_train + horizon)"
        )
    fold_id = 0
    train_end_pos = min_train_days - 1
    while train_end_pos + horizon_days < n:
        train_idx = np.arange(0, train_end_pos + 1)
        test_idx = np.arange(train_end_pos + 1, train_end_pos + 1 + horizon_days)
        origin = index[train_end_pos]
        yield Fold(
            fold_id=fold_id,
            train_idx=train_idx,
            test_idx=test_idx,
            origin=origin,
        )
        fold_id += 1
        train_end_pos += step_days


def count_folds(
    index: pd.DatetimeIndex,
    horizon_days: int = 7,
    step_days: int = 7,
    min_train_days: int = 365,
) -> int:
    """Return the number of folds without materialising the generator."""
    n = len(index)
    if n < min_train_days + horizon_days:
        return 0
    return 1 + (n - min_train_days - horizon_days) // step_days


def subsampled_rolling_origin(
    index: pd.DatetimeIndex,
    n_folds: int,
    horizon_days: int = 7,
    step_days: int = 7,
    min_train_days: int = 365,
) -> list[Fold]:
    """Pick n_folds evenly spaced from the full rolling-origin sequence.

    Used by HPO routines (XGBoost, ANN, LSTM) when full 69-fold CV is too
    expensive. The §3.6.1 spec is still honoured in spirit — each retained fold
    is a true expanding-window rolling-origin fold; we only subsample to fit
    the time budget.
    """
    all_folds = list(rolling_origin(index, horizon_days=horizon_days,
                                      step_days=step_days,
                                      min_train_days=min_train_days))
    if not all_folds:
        return []
    if n_folds >= len(all_folds):
        return all_folds
    # Evenly spaced indices including first and last
    positions = np.linspace(0, len(all_folds) - 1, n_folds).round().astype(int)
    positions = sorted(set(positions.tolist()))
    return [all_folds[i] for i in positions]


def evaluate_with_cv(
    folds: list[Fold],
    X: pd.DataFrame, y: pd.Series,
    fit_predict_fn,
    score_fn,
) -> dict:
    """Run a model factory across the supplied folds and aggregate scores.

    Parameters
    ----------
    folds : list of Fold (from rolling_origin or subsampled_rolling_origin)
    X, y : aligned full-train-block features and target
    fit_predict_fn : callable (X_train, y_train, X_test) -> y_pred
    score_fn : callable (y_actual, y_pred) -> dict[str, float]

    Returns
    -------
    {"per_fold": list[dict], "mean": dict, "std": dict}
    """
    per_fold = []
    for fold in folds:
        X_tr = X.iloc[fold.train_idx]
        y_tr = y.iloc[fold.train_idx]
        X_te = X.iloc[fold.test_idx]
        y_te = y.iloc[fold.test_idx]
        y_pred = fit_predict_fn(X_tr, y_tr, X_te)
        scores = score_fn(y_te.values, np.asarray(y_pred))
        per_fold.append({"fold_id": fold.fold_id, "origin": fold.origin, **scores})
    df = pd.DataFrame(per_fold)
    metric_cols = [c for c in df.columns if c not in ("fold_id", "origin")]
    return {
        "per_fold": per_fold,
        "mean": {c: float(df[c].mean()) for c in metric_cols},
        "std": {c: float(df[c].std()) for c in metric_cols},
    }


if __name__ == "__main__":
    from .io import load_g1, Splits

    splits = Splits.from_config()
    g1 = load_g1()
    train_df = splits.slice(g1, "train")
    print(f"Train block: {len(train_df)} days")
    n = count_folds(train_df.index, horizon_days=7, step_days=7, min_train_days=365)
    print(f"Rolling-origin folds (h=7, step=7, min_train=365): {n}")
    for fold in rolling_origin(train_df.index, min_train_days=365):
        if fold.fold_id < 3 or fold.fold_id >= n - 2:
            print(
                f"  fold {fold.fold_id:>3d}: "
                f"train {len(fold.train_idx)} d, "
                f"test {len(fold.test_idx)} d, "
                f"origin {fold.origin.date()}"
            )
        elif fold.fold_id == 3:
            print("  ...")
