"""Public evaluation interface for the nue contribution module."""

from collections.abc import Callable
import numpy as np
import pandas as pd
from megumi.gyokuken.visualization_utils import infer_target_type
from .utils import (
    _DEFAULT_METRICS,
    _build_model,
    _build_results_df,
    _get_cv,
    _resolve_metrics,
    _score_fold,
)


def evaluate_contribution(
    df: pd.DataFrame,
    base_features: list[str],
    new_features: list[str],
    target: str,
    metrics: list[str | Callable] | None = None,
    n_splits: int = 5,
    n_estimators: int = 200,
    random_state: int | None = None,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Measure the contribution of new features to model metrics.

    Fits two random forests per cross-validation fold — one on
    ``base_features`` alone and one on ``base_features + new_features`` —
    and compares their performance via a paired t-test across folds.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features, ``target``, and any additional columns needed
        by custom metric callables.
    base_features : list of str
        Features the current model already uses.
    new_features : list of str
        Candidate features to evaluate. Must not overlap with ``base_features``.
    target : str
        Response variable. Supports binary classification and regression;
        multiclass raises ``ValueError``.
    metrics : list of str or callable or None, optional
        Metrics to evaluate. Strings are resolved from the built-in registry:
        ``"roc_auc"``, ``"recall"``, ``"precision"``, ``"f1"``, ``"accuracy"``
        (classification); ``"rmse"``, ``"mae"``, ``"r2"`` (regression).
        Callables are invoked as ``fn(y_true, y_pred)``; if they accept a
        third positional argument, they also receive a ``pd.DataFrame`` slice
        of ``df`` for the test fold (all columns, reset index), which is
        useful for business metrics that depend on extra columns.
        Defaults to ``["roc_auc"]`` for binary targets and ``["rmse"]``
        for regression.
    n_splits : int, optional
        Cross-validation folds. Default is 5.
    n_estimators : int, optional
        Trees per forest. Default is 200.
    random_state : int or None, optional
        Seed for reproducibility. Default is ``None``.
    significance_level : float, optional
        Alpha for the paired t-test. Default is 0.05.

    Returns
    -------
    pd.DataFrame
        One row per metric with columns:

        - ``metric`` — metric name
        - ``base_score`` — mean score across folds (base features only)
        - ``augmented_score`` — mean score across folds (base + new features)
        - ``delta`` — augmented_score − base_score
        - ``pct_change`` — percentage change relative to base_score
        - ``p_value`` — paired t-test p-value (two-sided)
        - ``significant`` — whether ``p_value < significance_level``

    Raises
    ------
    ValueError
        If ``target`` is multiclass, ``base_features`` or ``new_features``
        is empty, features overlap, or required columns are missing.

    Examples
    --------
    >>> result = evaluate_contribution(
    ...     df, base_features=["age", "income"],
    ...     new_features=["vendor_score"], target="default",
    ...     metrics=["roc_auc", "recall"], random_state=42
    ... )
    """
    if not base_features:
        raise ValueError("base_features cannot be empty.")
    if not new_features:
        raise ValueError("new_features cannot be empty.")

    overlap = set(base_features) & set(new_features)
    if overlap:
        raise ValueError(
            f"Features appear in both base_features and new_features: {sorted(overlap)}."
        )

    all_features = base_features + new_features
    missing_cols = set(all_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in df: {sorted(missing_cols)}.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in df.")

    task = infer_target_type(df[target])
    if task == "multiclass":
        raise ValueError(
            "Multiclass targets are not supported. "
            "Use a binary classification or regression target."
        )

    resolved_metrics = _resolve_metrics(
        metrics if metrics is not None else _DEFAULT_METRICS[task], task
    )
    metric_names = [name for name, *_ in resolved_metrics]
    higher_is_better = [hib for *_, hib in resolved_metrics]

    # Anchor max_features to sqrt(n_base_features), held constant for both the
    # base and the augmented model. Using a relative formula like 'sqrt' would
    # scale with total feature count and give the augmented model a larger
    # per-split budget even when the new features are pure noise, producing
    # spurious systematic improvements that survive cross-validation.
    # int() (floor) matches sklearn's own 'sqrt' implementation for classifiers.
    max_features = max(1, int(np.sqrt(len(base_features))))

    y = df[target]
    n_metrics = len(resolved_metrics)
    base_scores = np.zeros((n_metrics, n_splits))
    aug_scores  = np.zeros((n_metrics, n_splits))
    null_scores = np.zeros((n_metrics, n_splits))

    for fold_idx, (train_idx, test_idx) in enumerate(
        _get_cv(task, n_splits, random_state).split(df, y)
    ):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        df_fold = df.iloc[test_idx].reset_index(drop=True)

        # Use a fold-specific seed so each fold's models are independently
        # seeded, restoring the independence assumption of the paired t-test.
        # Both models within the same fold share the same seed so that any
        # measured difference comes from the feature set, not from RF variance.
        fold_seed = None if random_state is None else random_state * n_splits + fold_idx

        base_scores[:, fold_idx] = _score_fold(
            _build_model(task, n_estimators, fold_seed, max_features),
            df.iloc[train_idx][base_features],
            df.iloc[test_idx][base_features],
            y_train, y_test, resolved_metrics, df_fold, task,
        )
        aug_scores[:, fold_idx] = _score_fold(
            _build_model(task, n_estimators, fold_seed, max_features),
            df.iloc[train_idx][all_features],
            df.iloc[test_idx][all_features],
            y_train, y_test, resolved_metrics, df_fold, task,
        )

        # Null model: same feature count as augmented but new features are
        # row-permuted (joint permutation preserves inter-feature covariance
        # while destroying any predictive relationship with the target).
        # This controls for the Random Forest diversification effect: adding
        # more features — even noise — reduces per-split feature overlap across
        # trees, which can improve ensemble performance independently of signal.
        # By comparing aug vs null (not aug vs base), the t-test isolates
        # genuine signal from structural benefits of a larger feature pool.
        perm_rng = np.random.default_rng(fold_seed)
        perm_idx = perm_rng.permutation(len(df))
        df_null = df.copy()
        df_null[new_features] = df[new_features].to_numpy()[perm_idx]

        null_scores[:, fold_idx] = _score_fold(
            _build_model(task, n_estimators, fold_seed, max_features),
            df_null.iloc[train_idx][all_features],
            df_null.iloc[test_idx][all_features],
            y_train, y_test, resolved_metrics, df_fold, task,
        )

    return _build_results_df(
        metric_names, base_scores, aug_scores, null_scores,
        significance_level, higher_is_better,
    )
