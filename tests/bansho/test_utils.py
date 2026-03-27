"""Unit tests for megumi.bansho.utils internal helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from megumi.bansho.utils import (
    _add_random_features,
    _assign_labels,
    _compute_shap_importances,
    _fit_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 200


@pytest.fixture
def binary_X_y():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, N), "b": rng.normal(0, 1, N)})
    y = pd.Series(rng.integers(0, 2, N))
    return X, y


@pytest.fixture
def regression_X_y():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, N), "b": rng.normal(0, 1, N)})
    y = pd.Series(rng.normal(5, 2, N))
    return X, y


@pytest.fixture
def fitted_classifier(binary_X_y):
    X, y = binary_X_y
    return _fit_model(X, y, task="binary", n_estimators=10, random_state=0)


@pytest.fixture
def fitted_regressor(regression_X_y):
    X, y = regression_X_y
    return _fit_model(X, y, task="continuous", n_estimators=10, random_state=0)


# ---------------------------------------------------------------------------
# _add_random_features
# ---------------------------------------------------------------------------


def test_add_random_features_adds_two_columns(binary_X_y):
    X, _ = binary_X_y
    result = _add_random_features(X, random_state=0)
    assert "RANDOM_1" in result.columns
    assert "RANDOM_2" in result.columns


def test_add_random_features_column_count(binary_X_y):
    X, _ = binary_X_y
    result = _add_random_features(X, random_state=0)
    assert result.shape[1] == X.shape[1] + 2


def test_add_random_features_preserves_row_count(binary_X_y):
    X, _ = binary_X_y
    result = _add_random_features(X, random_state=0)
    assert len(result) == len(X)


def test_add_random_features_does_not_modify_input(binary_X_y):
    X, _ = binary_X_y
    original_cols = list(X.columns)
    _add_random_features(X, random_state=0)
    assert list(X.columns) == original_cols


def test_add_random_features_reproducible(binary_X_y):
    X, _ = binary_X_y
    r1 = _add_random_features(X, random_state=42)
    r2 = _add_random_features(X, random_state=42)
    np.testing.assert_array_equal(r1["RANDOM_1"].values, r2["RANDOM_1"].values)
    np.testing.assert_array_equal(r1["RANDOM_2"].values, r2["RANDOM_2"].values)


def test_add_random_features_different_seeds_differ(binary_X_y):
    X, _ = binary_X_y
    r1 = _add_random_features(X, random_state=0)
    r2 = _add_random_features(X, random_state=1)
    assert not np.array_equal(r1["RANDOM_1"].values, r2["RANDOM_1"].values)


# ---------------------------------------------------------------------------
# _fit_model
# ---------------------------------------------------------------------------


def test_fit_model_binary_returns_classifier(binary_X_y):
    X, y = binary_X_y
    model = _fit_model(X, y, task="binary", n_estimators=10, random_state=0)
    assert isinstance(model, RandomForestClassifier)


def test_fit_model_continuous_returns_regressor(regression_X_y):
    X, y = regression_X_y
    model = _fit_model(X, y, task="continuous", n_estimators=10, random_state=0)
    assert isinstance(model, RandomForestRegressor)


def test_fit_model_respects_n_estimators(binary_X_y):
    X, y = binary_X_y
    model = _fit_model(X, y, task="binary", n_estimators=7, random_state=0)
    assert model.n_estimators == 7


def test_fit_model_unsupported_task_raises(binary_X_y):
    X, y = binary_X_y
    with pytest.raises(ValueError, match="not supported"):
        _fit_model(X, y, task="multiclass", n_estimators=10, random_state=0)


def test_fit_model_unknown_task_raises(binary_X_y):
    X, y = binary_X_y
    with pytest.raises(ValueError):
        _fit_model(X, y, task="ranking", n_estimators=10, random_state=0)


# ---------------------------------------------------------------------------
# _compute_shap_importances
# ---------------------------------------------------------------------------


def test_compute_shap_importances_shape_binary(fitted_classifier, binary_X_y):
    X, _ = binary_X_y
    importances = _compute_shap_importances(fitted_classifier, X, task="binary")
    assert importances.shape == (X.shape[1],)


def test_compute_shap_importances_shape_continuous(fitted_regressor, regression_X_y):
    X, _ = regression_X_y
    importances = _compute_shap_importances(fitted_regressor, X, task="continuous")
    assert importances.shape == (X.shape[1],)


def test_compute_shap_importances_non_negative_binary(fitted_classifier, binary_X_y):
    X, _ = binary_X_y
    importances = _compute_shap_importances(fitted_classifier, X, task="binary")
    assert np.all(importances >= 0)


def test_compute_shap_importances_non_negative_continuous(fitted_regressor, regression_X_y):
    X, _ = regression_X_y
    importances = _compute_shap_importances(fitted_regressor, X, task="continuous")
    assert np.all(importances >= 0)


def test_compute_shap_importances_returns_ndarray(fitted_classifier, binary_X_y):
    X, _ = binary_X_y
    importances = _compute_shap_importances(fitted_classifier, X, task="binary")
    assert isinstance(importances, np.ndarray)


# ---------------------------------------------------------------------------
# _assign_labels
# ---------------------------------------------------------------------------

# Controlled importances:
#   feat_a: 0.50 → above high_thresh (0.10) → predictive
#   feat_b: 0.09 → between low (0.08) and high (0.10) → marginal
#   feat_c: 0.02 → below low_thresh (0.08) → noise
#   RANDOM_1: 0.10, RANDOM_2: 0.08

_ALL_FEATURES = ["feat_a", "feat_b", "feat_c", "RANDOM_1", "RANDOM_2"]
_IMPORTANCES = np.array([0.50, 0.09, 0.02, 0.10, 0.08])


def test_assign_labels_output_columns():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    assert list(result.columns) == ["feature", "predictive_power"]


def test_assign_labels_excludes_random_features():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    assert "RANDOM_1" not in result["feature"].values
    assert "RANDOM_2" not in result["feature"].values


def test_assign_labels_row_count_equals_original_features():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    expected = len(_ALL_FEATURES) - 2   # minus the two randoms
    assert len(result) == expected


def test_assign_labels_valid_label_values():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    assert set(result["predictive_power"]).issubset({"predictive", "marginal", "noise"})


def test_assign_labels_predictive_correct():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    row = result.loc[result["feature"] == "feat_a", "predictive_power"].iloc[0]
    assert row == "predictive"


def test_assign_labels_marginal_correct():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    row = result.loc[result["feature"] == "feat_b", "predictive_power"].iloc[0]
    assert row == "marginal"


def test_assign_labels_noise_correct():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    row = result.loc[result["feature"] == "feat_c", "predictive_power"].iloc[0]
    assert row == "noise"


def test_assign_labels_sorted_descending_by_importance():
    result = _assign_labels(_IMPORTANCES, _ALL_FEATURES)
    # feat_a (0.50) should be first, feat_c (0.02) last
    assert result.iloc[0]["feature"] == "feat_a"
    assert result.iloc[-1]["feature"] == "feat_c"


def test_assign_labels_all_above_high_thresh_are_predictive():
    all_features = ["f1", "f2", "RANDOM_1", "RANDOM_2"]
    importances = np.array([0.9, 0.8, 0.05, 0.04])
    result = _assign_labels(importances, all_features)
    assert (result["predictive_power"] == "predictive").all()


def test_assign_labels_all_below_low_thresh_are_noise():
    all_features = ["f1", "f2", "RANDOM_1", "RANDOM_2"]
    importances = np.array([0.01, 0.02, 0.50, 0.40])
    result = _assign_labels(importances, all_features)
    assert (result["predictive_power"] == "noise").all()
