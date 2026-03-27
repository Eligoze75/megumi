"""Internal utilities for the bansho importance scoring module.

These helpers handle the mechanical steps of the pipeline — adding random
baselines, fitting the model, computing SHAP importances, and assigning
predictive power labels — keeping the public API in scorer.py clean.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import shap

_RANDOM_COLS = ("RANDOM_1", "RANDOM_2")

_LABELS = {
    "predictive": "predictive",
    "marginal": "marginal",
    "noise": "noise",
}


def _add_random_features(
    X: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Append two standard-normal random columns to X.

    The two synthetic columns act as lower-bound benchmarks: any real
    feature whose SHAP importance falls below both randoms carries no
    meaningful signal.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix. Not modified in place.
    random_state : int or None, optional
        Seed for reproducibility. Default is ``None``.

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` with two additional columns ``RANDOM_1`` and
        ``RANDOM_2`` drawn from N(0, 1).
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    X_ext = X.copy()
    X_ext[_RANDOM_COLS[0]] = rng.standard_normal(n)
    X_ext[_RANDOM_COLS[1]] = rng.standard_normal(n)
    return X_ext


def _fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    n_estimators: int,
    random_state: int | None,
) -> RandomForestClassifier | RandomForestRegressor:
    """Fit a random forest for the given task type.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix including the two random baseline columns.
    y : pd.Series
        Target variable.
    task : str
        ``'binary'`` or ``'continuous'``. ``'multiclass'`` raises
        ``ValueError``.
    n_estimators : int
        Number of trees in the forest.
    random_state : int or None
        Random seed passed to the forest.

    Returns
    -------
    RandomForestClassifier or RandomForestRegressor
        Fitted model.

    Raises
    ------
    ValueError
        If ``task`` is ``'multiclass'`` or unrecognised.
    """
    if task == "binary":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    elif task == "continuous":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"task='{task}' is not supported. "
            "bansho works with binary classification and regression targets only."
        )
    return model.fit(X, y)


def _compute_shap_importances(
    model: RandomForestClassifier | RandomForestRegressor,
    X: pd.DataFrame,
    task: str,
) -> np.ndarray:
    """Compute mean absolute SHAP values for every feature in X.

    Parameters
    ----------
    model : fitted RandomForestClassifier or RandomForestRegressor
        The model whose predictions are explained.
    X : pd.DataFrame
        Feature matrix used for SHAP explanation (same one used for
        fitting).
    task : str
        ``'binary'`` or ``'continuous'``. Determines which slice of the
        SHAP output to use for binary classifiers.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Mean absolute SHAP value for each column in ``X``, in the same
        column order.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if task == "binary":
        # Older shap returns a list [class_0_shap, class_1_shap];
        # newer shap returns a 3-D array (n_samples, n_features, n_classes).
        # In both cases we want the positive-class (index 1) slice so that
        # sv is always 2-D (n_samples, n_features) before aggregating.
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
    else:
        sv = shap_values

    return np.mean(np.abs(sv), axis=0)


def _assign_labels(
    importances: np.ndarray,
    all_features: list[str],
) -> pd.DataFrame:
    """Assign predictive-power labels relative to the two random baselines.

    The two random features act as a two-tier threshold:

    - ``"predictive"``  — importance > max(RANDOM_1, RANDOM_2)
    - ``"marginal"``    — min(RANDOM_1, RANDOM_2) < importance ≤ max(...)
    - ``"noise"``       — importance ≤ min(RANDOM_1, RANDOM_2)

    Parameters
    ----------
    importances : np.ndarray, shape (n_all_features,)
        Mean absolute SHAP for every column including the two randoms.
    all_features : list of str
        Column names matching ``importances`` in order; must contain
        ``RANDOM_1`` and ``RANDOM_2``.

    Returns
    -------
    pd.DataFrame
        Rows for original features only (randoms excluded), with columns
        ``feature`` and ``predictive_power``, sorted descending by
        importance.
    """
    imp_map = dict(zip(all_features, importances))
    shap_r1 = imp_map[_RANDOM_COLS[0]]
    shap_r2 = imp_map[_RANDOM_COLS[1]]

    high_thresh = max(shap_r1, shap_r2)
    low_thresh = min(shap_r1, shap_r2)

    original_features = [f for f in all_features if f not in _RANDOM_COLS]
    original_importances = np.array([imp_map[f] for f in original_features])

    labels = np.where(
        original_importances > high_thresh,
        _LABELS["predictive"],
        np.where(
            original_importances > low_thresh,
            _LABELS["marginal"],
            _LABELS["noise"],
        ),
    )

    order = np.argsort(original_importances)[::-1]
    return pd.DataFrame(
        {
            "feature": np.array(original_features)[order],
            "predictive_power": labels[order],
        }
    ).reset_index(drop=True)
