"""Internal utilities for the nue feature contribution module."""

import inspect
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from sklearn.metrics import root_mean_squared_error as _sklearn_rmse
except ImportError:
    def _sklearn_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # sklearn < 1.4
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(_sklearn_rmse(y_true, y_pred))


_METRIC_REGISTRY: dict[str, tuple[Callable, bool, bool]] = {
    # (fn, needs_proba, higher_is_better)
    "roc_auc":   (roc_auc_score,      True,  True),
    "recall":    (recall_score,        False, True),
    "precision": (precision_score,     False, True),
    "f1":        (f1_score,            False, True),
    "accuracy":  (accuracy_score,      False, True),
    "rmse":      (_rmse,               False, False),
    "mae":       (mean_absolute_error, False, False),
    "r2":        (r2_score,            False, True),
}

_DEFAULT_METRICS: dict[str, list[str]] = {
    "binary":     ["roc_auc"],
    "continuous": ["rmse"],
}


def _build_model(
    task: str,
    n_estimators: int,
    random_state: int | None,
    max_features: int,
) -> RandomForestClassifier | RandomForestRegressor:
    """Return an unfitted random forest for the given task.

    Parameters
    ----------
    task : str
        ``'binary'`` or ``'continuous'``.
    n_estimators : int
    random_state : int or None
    max_features : int
        Number of features to consider at each split. Kept identical for both
        the base and the augmented model so that any measured difference comes
        from the new features' signal, not from a larger per-split feature budget.

    Returns
    -------
    RandomForestClassifier or RandomForestRegressor
    """
    cls = RandomForestClassifier if task == "binary" else RandomForestRegressor
    return cls(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, max_features=max_features)


def _udf_accepts_df(fn: Callable) -> bool:
    """Return True if ``fn`` accepts a third positional argument.

    Parameters
    ----------
    fn : callable

    Returns
    -------
    bool
    """
    try:
        sig = inspect.signature(fn)
        positional = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 3
    except (ValueError, TypeError):
        return False


def _resolve_metrics(
    metrics: list[str | Callable],
    task: str,
) -> list[tuple[str, Callable, bool, bool, bool]]:
    """Resolve metrics to ``(name, fn, needs_proba, accepts_df, higher_is_better)`` tuples.

    Parameters
    ----------
    metrics : list of str or callable
    task : str
        ``'binary'`` or ``'continuous'``.

    Returns
    -------
    list of (str, callable, bool, bool, bool)
        ``(name, fn, needs_proba, accepts_df, higher_is_better)``.

    Raises
    ------
    ValueError
        Unknown metric string or metric incompatible with ``task``.
    TypeError
        Element is neither str nor callable.
    """
    resolved = []
    for m in metrics:
        if isinstance(m, str):
            if m not in _METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{m}'. "
                    f"Supported: {sorted(_METRIC_REGISTRY)}. "
                    "Pass a callable for custom metrics."
                )
            fn, needs_proba, higher_is_better = _METRIC_REGISTRY[m]
            if task == "continuous" and needs_proba:
                raise ValueError(
                    f"Metric '{m}' requires class probabilities "
                    "and cannot be used with regression targets."
                )
            resolved.append((m, fn, needs_proba, False, higher_is_better))
        elif callable(m):
            name = getattr(m, "__name__", repr(m))
            # UDFs receive predicted probabilities for binary classification
            # (more useful for business metrics) and predicted values for regression.
            needs_proba = task == "binary"
            accepts_df = _udf_accepts_df(m)
            # UDFs are assumed to be higher-is-better (e.g. business metrics like
            # "loss avoided"). Users with lower-is-better UDFs should negate the
            # return value of their function.
            resolved.append((name, m, needs_proba, accepts_df, True))
        else:
            raise TypeError(
                f"Each metric must be a str or callable, got {type(m).__name__}."
            )
    return resolved


def _get_cv(
    task: str,
    n_splits: int,
    random_state: int | None,
) -> StratifiedKFold | KFold:
    """Return the appropriate cross-validator for the task.

    Parameters
    ----------
    task : str
        ``'binary'`` or ``'continuous'``.
    n_splits : int
    random_state : int or None

    Returns
    -------
    StratifiedKFold or KFold
    """
    if task == "binary":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _score_fold(
    model: RandomForestClassifier | RandomForestRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metric_fns: list[tuple[str, Callable, bool, bool, bool]],
    df_fold: pd.DataFrame,
    task: str,
) -> np.ndarray:
    """Fit ``model`` and evaluate all metrics on the test fold.

    Parameters
    ----------
    model : unfitted RandomForest
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    metric_fns : list of (name, fn, needs_proba, accepts_df, higher_is_better)
    df_fold : pd.DataFrame
        Full ``df`` slice for test indices (all columns, reset index).
        Passed to UDFs that accept a third positional argument.
    task : str

    Returns
    -------
    np.ndarray, shape (n_metrics,)
    """
    model.fit(X_train, y_train)
    y_true = y_test.values

    if task == "binary":
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_class = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    scores = []
    for _, fn, needs_proba, accepts_df, _ in metric_fns:
        y_hat = (
            (y_pred_proba if needs_proba else y_pred_class)
            if task == "binary"
            else y_pred
        )
        score = fn(y_true, y_hat, df_fold) if accepts_df else fn(y_true, y_hat)
        scores.append(float(score))

    return np.array(scores)


def _build_results_df(
    metric_names: list[str],
    base_arr: np.ndarray,
    aug_arr: np.ndarray,
    null_arr: np.ndarray,
    significance_level: float,
    higher_is_better: list[bool],
) -> pd.DataFrame:
    """Assemble the results DataFrame with permutation-corrected significance.

    Parameters
    ----------
    metric_names : list of str
    base_arr : np.ndarray, shape (n_metrics, n_splits)
        Fold scores for the base model (base features only).
    aug_arr : np.ndarray, shape (n_metrics, n_splits)
        Fold scores for the augmented model (base + new features, real values).
    null_arr : np.ndarray, shape (n_metrics, n_splits)
        Fold scores for the null model (base + new features, new ones
        row-permuted). Used as the reference for the paired t-test so that
        the Random Forest diversification effect — where adding more features
        (even noise) improves ensemble diversity and thus performance — is
        controlled for. The p-value therefore tests genuine signal, not the
        structural benefit of a larger feature pool.
    significance_level : float
    higher_is_better : list of bool
        Whether a larger value is better for each metric. Used to gate
        ``significant``: a statistically significant *degradation* is not
        flagged as a significant improvement.

    Returns
    -------
    pd.DataFrame
        Columns: ``metric``, ``base_score``, ``augmented_score``, ``delta``,
        ``pct_change``, ``p_value``, ``significant``.
    """
    rows = []
    for i, name in enumerate(metric_names):
        b = base_arr[i]
        a = aug_arr[i]
        n = null_arr[i]
        base_mean = b.mean()
        aug_mean = a.mean()
        delta = aug_mean - base_mean
        pct_change = delta / abs(base_mean) * 100 if base_mean != 0 else np.nan

        # Compare augmented (real) vs null (permuted) — same feature count,
        # so any consistent advantage reflects signal, not diversification.
        p_value = 1.0 if np.allclose(a, n) else float(ttest_rel(a, n).pvalue)

        signal_delta = (a - n).mean()
        is_improvement = signal_delta > 0 if higher_is_better[i] else signal_delta < 0
        significant = bool(p_value < significance_level and is_improvement)

        rows.append({
            "metric":           name,
            "base_score":       base_mean,
            "augmented_score":  aug_mean,
            "delta":            delta,
            "pct_change":       pct_change,
            "p_value":          p_value,
            "significant":      significant,
        })
    return pd.DataFrame(rows)
