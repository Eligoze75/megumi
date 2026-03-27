"""Tests for megumi.bansho.score_features public API."""

import numpy as np
import pandas as pd
import pytest

from megumi.bansho import score_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300
_FEATURES = ["a", "b", "c", "d"]
_VALID_LABELS = {"predictive", "marginal", "noise"}


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, N),
            "b": rng.normal(0, 1, N),
            "c": rng.normal(0, 1, N),
            "d": rng.normal(0, 1, N),
            "target": rng.integers(0, 2, N),
        }
    )
    return df


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, N),
            "b": rng.normal(0, 1, N),
            "c": rng.normal(0, 1, N),
            "d": rng.normal(0, 1, N),
        }
    )
    df["target"] = df["a"] * 2 + df["b"] * 0.5 + rng.normal(0, 0.5, N)
    return df


@pytest.fixture
def multiclass_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, N),
            "b": rng.normal(0, 1, N),
            "label": rng.choice(["x", "y", "z"], N),
        }
    )


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_score_features_returns_dataframe(binary_df):
    result = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)
    assert isinstance(result, pd.DataFrame)


def test_score_features_output_columns(binary_df):
    result = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)
    assert list(result.columns) == ["feature", "predictive_power"]


def test_score_features_row_count_equals_features(binary_df):
    result = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)
    assert len(result) == len(_FEATURES)


def test_score_features_all_input_features_present(binary_df):
    result = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)
    assert set(result["feature"]) == set(_FEATURES)


def test_score_features_random_sentinels_not_in_output(binary_df):
    result = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)
    assert "RANDOM_1" not in result["feature"].values
    assert "RANDOM_2" not in result["feature"].values


def test_score_features_valid_label_values(binary_df):
    result = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)
    assert set(result["predictive_power"]).issubset(_VALID_LABELS)


# ---------------------------------------------------------------------------
# Task support
# ---------------------------------------------------------------------------


def test_score_features_binary_runs(binary_df):
    score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=0)


def test_score_features_regression_runs(regression_df):
    score_features(regression_df, _FEATURES, "target", n_estimators=10, random_state=0)


# ---------------------------------------------------------------------------
# df_val parameter
# ---------------------------------------------------------------------------


def test_score_features_df_val_accepted(binary_df):
    train = binary_df.iloc[:200]
    val = binary_df.iloc[200:]
    result = score_features(
        train, _FEATURES, "target", df_val=val, n_estimators=10, random_state=0
    )
    assert len(result) == len(_FEATURES)


def test_score_features_df_val_missing_column_raises(binary_df):
    train = binary_df.iloc[:200]
    val = binary_df.iloc[200:].drop(columns=["a"])
    with pytest.raises(ValueError, match="df_val is missing"):
        score_features(
            train, _FEATURES, "target", df_val=val, n_estimators=10, random_state=0
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_score_features_same_seed_same_result(binary_df):
    r1 = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=7)
    r2 = score_features(binary_df, _FEATURES, "target", n_estimators=10, random_state=7)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Input validation — raises
# ---------------------------------------------------------------------------


def test_score_features_multiclass_raises(multiclass_df):
    with pytest.raises(ValueError, match="Multiclass"):
        score_features(multiclass_df, ["a", "b"], "label", n_estimators=10, random_state=0)


def test_score_features_reserved_name_random1_raises(binary_df):
    df = binary_df.copy()
    df["RANDOM_1"] = 0.0
    with pytest.raises(ValueError, match="reserved"):
        score_features(df, _FEATURES + ["RANDOM_1"], "target", n_estimators=10, random_state=0)


def test_score_features_reserved_name_random2_raises(binary_df):
    df = binary_df.copy()
    df["RANDOM_2"] = 0.0
    with pytest.raises(ValueError, match="reserved"):
        score_features(df, _FEATURES + ["RANDOM_2"], "target", n_estimators=10, random_state=0)


def test_score_features_unsupported_strategy_raises(binary_df):
    with pytest.raises(NotImplementedError, match="strategy"):
        score_features(
            binary_df, _FEATURES, "target",
            strategy="linear", n_estimators=10, random_state=0,
        )
