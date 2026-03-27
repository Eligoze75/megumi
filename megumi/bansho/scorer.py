"""Public scoring interface for the bansho importance module."""

import pandas as pd

from megumi.gyokuken.visualization_utils import infer_target_type
from .utils import (
    _add_random_features,
    _assign_labels,
    _compute_shap_importances,
    _fit_model,
)


def score_features(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    df_val: pd.DataFrame | None = None,
    strategy: str = "tree",
    n_estimators: int = 200,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Rank features by SHAP importance relative to two random baselines.

    Fits a random forest on ``df``, injects two N(0,1) sentinel features
    (``RANDOM_1``, ``RANDOM_2``), and labels each original feature by how
    its mean absolute SHAP compares to those sentinels:

    - ``"predictive"`` — beats both sentinels
    - ``"marginal"``   — beats one sentinel
    - ``"noise"``      — beats neither sentinel

    Parameters
    ----------
    df : pd.DataFrame
        Training data. Must contain ``features`` and ``target`` columns.
    features : list of str
        Features to evaluate. ``RANDOM_1`` and ``RANDOM_2`` are reserved.
    target : str
        Response variable. Supports binary classification and regression;
        multiclass raises ``ValueError``.
    df_val : pd.DataFrame or None, optional
        Validation set used to compute SHAP values. When provided, the
        forest is fitted on ``df`` and SHAP is computed on ``df_val``,
        giving more conservative estimates. Defaults to ``None`` (SHAP
        computed on ``df``).
    strategy : str, optional
        Model type. Only ``"tree"`` (random forest) is supported.
    n_estimators : int, optional
        Number of trees. Default is ``200``.
    random_state : int or None, optional
        Seed for reproducibility. Default is ``None``.

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``predictive_power``, sorted by
        descending mean absolute SHAP. Sentinel features are excluded.

    Examples
    --------
    >>> score_features(df_train, features, target="y", df_val=df_val,
    ...                random_state=42)
    """
    if strategy != "tree":
        raise NotImplementedError(
            f"strategy='{strategy}' is not yet implemented. Use strategy='tree'."
        )

    reserved = {"RANDOM_1", "RANDOM_2"}
    overlap = reserved.intersection(features)
    if overlap:
        raise ValueError(
            f"Feature names {sorted(overlap)} are reserved for internal random "
            "baselines. Rename them before calling score_features."
        )

    if df_val is not None:
        missing = set(features) - set(df_val.columns)
        if missing:
            raise ValueError(
                f"df_val is missing the following feature columns: {sorted(missing)}."
            )

    task = infer_target_type(df[target])
    if task == "multiclass":
        raise ValueError(
            "Multiclass targets are not supported by bansho. "
            "Use a binary classification or regression target."
        )

    X_train = _add_random_features(df[features].copy(), random_state=random_state)
    all_features = list(X_train.columns)

    X_val = (
        _add_random_features(df_val[features].copy(), random_state=random_state)
        if df_val is not None
        else X_train
    )

    model = _fit_model(X_train, df[target], task=task, n_estimators=n_estimators, random_state=random_state)
    importances = _compute_shap_importances(model, X_val, task=task)

    return _assign_labels(importances, all_features)
