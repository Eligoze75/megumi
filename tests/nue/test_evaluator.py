"""Tests for megumi.nue.evaluate_contribution public API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from megumi.nue import evaluate_contribution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 500
BASE = [f"base_{i}" for i in range(5)]
NEW = [f"new_{i}" for i in range(3)]
NOISE = [f"noise_{i}" for i in range(3)]

EXPECTED_COLUMNS = [
    "metric", "base_score", "augmented_score",
    "delta", "pct_change", "p_value", "significant",
]


@pytest.fixture
def binary_df():
    X, y = make_classification(
        n_samples=N, n_features=len(BASE) + len(NEW),
        n_informative=len(BASE) + len(NEW), n_redundant=0,
        n_repeated=0, random_state=0,
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame(X, columns=BASE + NEW)
    df["target"] = y
    for col in NOISE:
        df[col] = rng.standard_normal(N)
    return df


@pytest.fixture
def regression_df():
    X, y = make_regression(
        n_samples=N, n_features=len(BASE) + len(NEW),
        n_informative=len(BASE) + len(NEW), noise=10, random_state=0,
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame(X, columns=BASE + NEW)
    df["price"] = y
    for col in NOISE:
        df[col] = rng.standard_normal(N)
    return df


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_returns_dataframe(binary_df):
    result = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=0)
    assert isinstance(result, pd.DataFrame)


def test_output_columns(binary_df):
    result = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=0)
    assert list(result.columns) == EXPECTED_COLUMNS


def test_one_row_per_metric(binary_df):
    result = evaluate_contribution(
        binary_df, BASE, NEW, "target",
        metrics=["roc_auc", "recall", "f1"],
        n_estimators=10, random_state=0,
    )
    assert len(result) == 3


def test_p_value_in_unit_interval(binary_df):
    result = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=0)
    assert result["p_value"].between(0, 1).all()


def test_significant_is_bool(binary_df):
    result = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=0)
    assert result["significant"].dtype == bool


def test_delta_equals_augmented_minus_base(binary_df):
    result = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=0)
    expected = result["augmented_score"] - result["base_score"]
    pd.testing.assert_series_equal(result["delta"], expected, check_names=False)


def test_metric_name_column(binary_df):
    result = evaluate_contribution(
        binary_df, BASE, NEW, "target",
        metrics=["roc_auc", "recall"],
        n_estimators=10, random_state=0,
    )
    assert list(result["metric"]) == ["roc_auc", "recall"]


# ---------------------------------------------------------------------------
# Task support
# ---------------------------------------------------------------------------


def test_binary_classification_runs(binary_df):
    evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=0)


def test_regression_runs(regression_df):
    evaluate_contribution(regression_df, BASE, NEW, "price", n_estimators=10, random_state=0)


def test_all_classification_metrics(binary_df):
    result = evaluate_contribution(
        binary_df, BASE, NEW, "target",
        metrics=["roc_auc", "recall", "precision", "f1", "accuracy"],
        n_estimators=10, random_state=0,
    )
    assert len(result) == 5


def test_all_regression_metrics(regression_df):
    result = evaluate_contribution(
        regression_df, BASE, NEW, "price",
        metrics=["rmse", "mae", "r2"],
        n_estimators=10, random_state=0,
    )
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Custom metrics (UDFs)
# ---------------------------------------------------------------------------


def test_udf_two_arg(binary_df):
    def hit_rate(y_true, y_pred):
        return float((y_true == y_pred).mean())

    result = evaluate_contribution(
        binary_df, BASE, NEW, "target",
        metrics=[hit_rate], n_estimators=10, random_state=0,
    )
    assert len(result) == 1
    assert result["metric"].iloc[0] == "hit_rate"


def test_udf_three_arg_receives_df_fold(binary_df):
    binary_df = binary_df.copy()
    binary_df["amount"] = 1000.0

    def loss_avoided(y_true, y_pred_proba, df_fold):
        flagged = y_pred_proba >= 0.5
        return float(df_fold.loc[y_true.astype(bool) & flagged, "amount"].sum())

    result = evaluate_contribution(
        binary_df, BASE, NEW, "target",
        metrics=[loss_avoided], n_estimators=10, random_state=0,
    )
    assert len(result) == 1
    assert result["base_score"].iloc[0] >= 0


def test_mixed_str_and_udf_metrics(binary_df):
    def custom(y_true, y_pred):
        return float((y_true == y_pred).mean())

    result = evaluate_contribution(
        binary_df, BASE, NEW, "target",
        metrics=["roc_auc", custom],
        n_estimators=10, random_state=0,
    )
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result(binary_df):
    r1 = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=7)
    r2 = evaluate_contribution(binary_df, BASE, NEW, "target", n_estimators=10, random_state=7)
    pd.testing.assert_frame_equal(r1, r2)


def test_same_seed_regression(regression_df):
    r1 = evaluate_contribution(regression_df, BASE, NEW, "price", n_estimators=10, random_state=7)
    r2 = evaluate_contribution(regression_df, BASE, NEW, "price", n_estimators=10, random_state=7)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Significance direction: pure noise must not be flagged as significant
# ---------------------------------------------------------------------------


def test_noise_not_significant_classification(binary_df):
    result = evaluate_contribution(
        binary_df, BASE, NOISE, "target",
        metrics=["roc_auc"],
        n_estimators=50, n_splits=5, random_state=0,
    )
    assert not result["significant"].any()


def test_noise_not_significant_regression(regression_df):
    result = evaluate_contribution(
        regression_df, BASE, NOISE, "price",
        metrics=["rmse"],
        n_estimators=50, n_splits=5, random_state=0,
    )
    assert not result["significant"].any()


# ---------------------------------------------------------------------------
# Input validation — raises
# ---------------------------------------------------------------------------


def test_empty_base_features_raises(binary_df):
    with pytest.raises(ValueError, match="base_features"):
        evaluate_contribution(binary_df, [], NEW, "target")


def test_empty_new_features_raises(binary_df):
    with pytest.raises(ValueError, match="new_features"):
        evaluate_contribution(binary_df, BASE, [], "target")


def test_overlapping_features_raises(binary_df):
    with pytest.raises(ValueError, match="both base_features and new_features"):
        evaluate_contribution(binary_df, BASE, BASE[:2] + NEW[:1], "target")


def test_missing_feature_column_raises(binary_df):
    with pytest.raises(ValueError, match="Columns not found"):
        evaluate_contribution(binary_df, BASE, ["does_not_exist"], "target")


def test_missing_target_raises(binary_df):
    with pytest.raises(ValueError, match="Target column"):
        evaluate_contribution(binary_df, BASE, NEW, "nonexistent")


def test_multiclass_raises(binary_df):
    df = binary_df.copy()
    df["target"] = np.tile([0, 1, 2], N // 3 + 1)[:N]
    with pytest.raises(ValueError, match="Multiclass"):
        evaluate_contribution(df, BASE, NEW, "target")


def test_unknown_metric_string_raises(binary_df):
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_contribution(binary_df, BASE, NEW, "target", metrics=["bad_metric"])


def test_non_callable_metric_raises(binary_df):
    with pytest.raises(TypeError, match="str or callable"):
        evaluate_contribution(binary_df, BASE, NEW, "target", metrics=[42])


def test_proba_metric_on_regression_raises(regression_df):
    with pytest.raises(ValueError, match="class probabilities"):
        evaluate_contribution(regression_df, BASE, NEW, "price", metrics=["roc_auc"])
