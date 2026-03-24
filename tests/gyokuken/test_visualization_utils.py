"""Tests for megumi.gyokuken.visualization_utils."""

import numpy as np
import pandas as pd
import pytest

from megumi.gyokuken.visualization_utils import (
    bin_continuous_feature,
    infer_feature_type,
    infer_target_type,
    rare_label_encoder,
)


# ---------------------------------------------------------------------------
# rare_label_encoder
# ---------------------------------------------------------------------------


def test_rare_label_encoder_replaces_rare_labels():
    s = pd.Series(["a"] * 80 + ["b"] * 15 + ["rare"] * 5)
    result = rare_label_encoder(s, threshold=0.10)
    assert "rare" not in result.values
    assert "Other" in result.values


def test_rare_label_encoder_keeps_frequent_labels():
    s = pd.Series(["a"] * 80 + ["b"] * 15 + ["rare"] * 5)
    result = rare_label_encoder(s, threshold=0.10)
    assert "a" in result.values
    assert "b" in result.values


def test_rare_label_encoder_custom_fill_value():
    s = pd.Series(["a"] * 80 + ["rare"] * 20)
    result = rare_label_encoder(s, threshold=0.25, fill_value="Infrequent")
    assert "Infrequent" in result.values
    assert "Other" not in result.values


def test_rare_label_encoder_preserves_series_length():
    s = pd.Series(["a"] * 80 + ["rare"] * 20)
    result = rare_label_encoder(s, threshold=0.25)
    assert len(result) == len(s)


def test_rare_label_encoder_no_change_when_all_frequent():
    s = pd.Series(["a"] * 50 + ["b"] * 50)
    result = rare_label_encoder(s, threshold=0.05)
    assert set(result.unique()) == {"a", "b"}


def test_rare_label_encoder_multiple_rare_labels_merged():
    s = pd.Series(["common"] * 90 + ["rare1"] * 5 + ["rare2"] * 5)
    result = rare_label_encoder(s, threshold=0.08)
    # Both rare labels collapsed into one "Other"
    assert "rare1" not in result.values
    assert "rare2" not in result.values
    assert result.value_counts()["Other"] == 10


# ---------------------------------------------------------------------------
# bin_continuous_feature
# ---------------------------------------------------------------------------


def test_bin_continuous_feature_returns_categorical_dtype():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 1, 200))
    result = bin_continuous_feature(s, n_bins=5)
    assert isinstance(result.dtype, pd.CategoricalDtype)


def test_bin_continuous_feature_at_most_n_bins():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 1, 200))
    result = bin_continuous_feature(s, n_bins=5)
    assert result.nunique() <= 5


def test_bin_continuous_feature_preserves_series_length():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 1, 150))
    result = bin_continuous_feature(s, n_bins=5)
    assert len(result) == len(s)


def test_bin_continuous_feature_no_nulls_in_output():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 1, 100))
    result = bin_continuous_feature(s, n_bins=5)
    assert result.isna().sum() == 0


def test_bin_continuous_feature_handles_duplicate_edges():
    # Heavily concentrated data produces duplicate qcut edges — must not raise
    s = pd.Series([1.0] * 90 + list(np.linspace(2, 3, 10)))
    result = bin_continuous_feature(s, n_bins=5)
    assert result is not None
    assert len(result) == len(s)


# ---------------------------------------------------------------------------
# infer_feature_type
# ---------------------------------------------------------------------------


def test_infer_feature_type_high_cardinality_float_is_continuous():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 1, 200))
    assert infer_feature_type(s) == "continuous"


def test_infer_feature_type_object_dtype_is_categorical():
    s = pd.Series(["a", "b", "c"] * 50)
    assert infer_feature_type(s) == "categorical"


def test_infer_feature_type_low_cardinality_numeric_is_categorical():
    s = pd.Series([0, 1, 2] * 50)   # only 3 unique values
    assert infer_feature_type(s) == "categorical"


def test_infer_feature_type_categorical_dtype_is_categorical():
    s = pd.Series(pd.Categorical(["a", "b", "c"] * 50))
    assert infer_feature_type(s) == "categorical"


def test_infer_feature_type_boundary_at_max_categories():
    # Exactly max_categories unique values → categorical
    s = pd.Series(list(range(20)) * 10)
    assert infer_feature_type(s, max_categories=20) == "categorical"


def test_infer_feature_type_above_max_categories_is_continuous():
    # One more than max_categories → continuous
    s = pd.Series(list(range(21)) * 10)
    assert infer_feature_type(s, max_categories=20) == "continuous"


# ---------------------------------------------------------------------------
# infer_target_type
# ---------------------------------------------------------------------------


def test_infer_target_type_binary_numeric():
    s = pd.Series([0, 1, 0, 1, 1, 0])
    assert infer_target_type(s) == "binary"


def test_infer_target_type_binary_string():
    # Two unique string labels → binary, not multiclass
    s = pd.Series(["malignant", "benign", "malignant", "benign"])
    assert infer_target_type(s) == "binary"


def test_infer_target_type_multiclass_string():
    s = pd.Series(["cat", "dog", "bird", "cat", "dog"])
    assert infer_target_type(s) == "multiclass"


def test_infer_target_type_multiclass_integer():
    s = pd.Series([0, 1, 2] * 10)
    assert infer_target_type(s) == "multiclass"


def test_infer_target_type_categorical_dtype_multiclass():
    s = pd.Series(pd.Categorical(["x", "y", "z"] * 10))
    assert infer_target_type(s) == "multiclass"


def test_infer_target_type_continuous():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 1, 200))
    assert infer_target_type(s) == "continuous"
