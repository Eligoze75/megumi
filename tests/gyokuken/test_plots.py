"""Tests for megumi.gyokuken plotting functions."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from megumi.gyokuken import (
    plot_bivariate,
    plot_correlation,
    plot_distribution,
    plot_missing,
)


matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "income": rng.normal(50_000, 15_000, n),
            "category": rng.choice(
                ["A", "B", "C", "rare1", "rare2"],
                n,
                p=[0.40, 0.30, 0.20, 0.05, 0.05],
            ),
            "default": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def multiclass_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.uniform(0, 1, n),
            "label": rng.choice(["cat", "dog", "bird"], n),
        }
    )


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(0, 1, n),
            "y": rng.normal(5, 2, n),
        }
    )


@pytest.fixture
def missing_df():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        }
    )
    df.loc[rng.choice(n, 20, replace=False), "a"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "b"] = np.nan
    return df


# ---------------------------------------------------------------------------
# plot_bivariate — smoke tests
# ---------------------------------------------------------------------------


def test_plot_bivariate_continuous(binary_df):
    plot_bivariate(binary_df, feature="income", target="default")
    plt.close("all")


def test_plot_bivariate_categorical(binary_df):
    plot_bivariate(binary_df, feature="category", target="default")
    plt.close("all")


def test_plot_bivariate_integer_feature(binary_df):
    plot_bivariate(binary_df, feature="age", target="default")
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_bivariate — structural assertions
# ---------------------------------------------------------------------------


def test_plot_bivariate_creates_two_axes(binary_df):
    plot_bivariate(binary_df, feature="income", target="default")
    assert len(plt.gcf().axes) == 2
    plt.close("all")


def test_plot_bivariate_left_ylabel_contains_target_name(binary_df):
    plot_bivariate(binary_df, feature="income", target="default")
    ylabel = plt.gcf().axes[0].get_ylabel()
    assert "default" in ylabel
    plt.close("all")


def test_plot_bivariate_right_ylabel_contains_count(binary_df):
    plot_bivariate(binary_df, feature="income", target="default")
    ylabel = plt.gcf().axes[1].get_ylabel()
    assert "Count" in ylabel
    plt.close("all")


def test_plot_bivariate_title_contains_feature_and_target(binary_df):
    plot_bivariate(binary_df, feature="income", target="default")
    title = plt.gcf().axes[0].get_title()
    assert "income" in title
    assert "default" in title
    plt.close("all")


def test_plot_bivariate_xtick_count_respects_n_bins(binary_df):
    n_bins = 4
    plot_bivariate(binary_df, feature="income", target="default", n_bins=n_bins)
    tick_labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels() if t.get_text()]
    assert len(tick_labels) <= n_bins
    plt.close("all")


def test_plot_bivariate_missing_values_get_own_bucket(binary_df):
    df = binary_df.copy()
    df.loc[:20, "income"] = None
    plot_bivariate(df, feature="income", target="default")
    tick_labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert "Missing" in tick_labels
    plt.close("all")


def test_plot_bivariate_no_missing_no_bucket(binary_df):
    plot_bivariate(binary_df, feature="income", target="default")
    tick_labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
    assert "Missing" not in tick_labels
    plt.close("all")


def test_plot_bivariate_multiclass_raises(multiclass_df):
    with pytest.raises(ValueError, match="multiclass"):
        plot_bivariate(multiclass_df, feature="feature1", target="label")


def test_plot_bivariate_regression_warns(regression_df):
    with pytest.warns(UserWarning, match="continuous"):
        plot_bivariate(regression_df, feature="x1", target="y")
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_distribution — smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["histogram", "kde", "violin", "boxplot"])
def test_plot_distribution_binary(binary_df, kind):
    plot_distribution(binary_df, feature="income", target="default", kind=kind)
    plt.close("all")


@pytest.mark.parametrize("kind", ["histogram", "kde", "violin", "boxplot"])
def test_plot_distribution_multiclass(multiclass_df, kind):
    plot_distribution(multiclass_df, feature="feature1", target="label", kind=kind)
    plt.close("all")


@pytest.mark.parametrize("kind", ["histogram", "kde", "violin", "boxplot"])
def test_plot_distribution_regression(regression_df, kind):
    plot_distribution(regression_df, feature="x1", target="y", kind=kind)
    plt.close("all")


def test_plot_distribution_invalid_kind(binary_df):
    with pytest.raises(ValueError, match="kind must be one of"):
        plot_distribution(binary_df, feature="income", target="default", kind="pie")


# ---------------------------------------------------------------------------
# plot_distribution — structural assertions
# ---------------------------------------------------------------------------


def test_plot_distribution_title_contains_feature(binary_df):
    plot_distribution(binary_df, feature="income", target="default")
    assert "income" in plt.gcf().axes[0].get_title()
    plt.close("all")


def test_plot_distribution_histogram_xlabel_is_feature(binary_df):
    plot_distribution(binary_df, feature="income", target="default", kind="histogram")
    assert "income" in plt.gcf().axes[0].get_xlabel()
    plt.close("all")


def test_plot_distribution_kde_xlabel_is_feature(binary_df):
    plot_distribution(binary_df, feature="income", target="default", kind="kde")
    assert "income" in plt.gcf().axes[0].get_xlabel()
    plt.close("all")


def test_plot_distribution_violin_xlabel_is_target(binary_df):
    plot_distribution(binary_df, feature="income", target="default", kind="violin")
    assert "default" in plt.gcf().axes[0].get_xlabel()
    plt.close("all")


def test_plot_distribution_boxplot_xlabel_is_target(binary_df):
    plot_distribution(binary_df, feature="income", target="default", kind="boxplot")
    assert "default" in plt.gcf().axes[0].get_xlabel()
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_correlation — smoke tests
# ---------------------------------------------------------------------------


def test_plot_correlation_default(binary_df):
    plot_correlation(binary_df)
    plt.close("all")


def test_plot_correlation_spearman(binary_df):
    plot_correlation(binary_df, method="spearman")
    plt.close("all")


def test_plot_correlation_feature_subset(binary_df):
    plot_correlation(binary_df, features=["age", "income"])
    plt.close("all")


def test_plot_correlation_multiclass_raises(multiclass_df):
    with pytest.raises(ValueError, match="multiclass"):
        plot_correlation(multiclass_df, target="label")


# ---------------------------------------------------------------------------
# plot_correlation — structural assertions
# ---------------------------------------------------------------------------


def test_plot_correlation_title_contains_method(binary_df):
    plot_correlation(binary_df, method="spearman")
    assert "Spearman" in plt.gcf().axes[0].get_title()
    plt.close("all")


def test_plot_correlation_target_excluded_from_heatmap(binary_df):
    plot_correlation(binary_df, target="default")
    ylabels = [t.get_text() for t in plt.gcf().axes[0].get_yticklabels()]
    assert "default" not in ylabels
    plt.close("all")


def test_plot_correlation_heatmap_rows_match_features(binary_df):
    features = ["age", "income"]
    plot_correlation(binary_df, features=features)
    ylabels = [t.get_text() for t in plt.gcf().axes[0].get_yticklabels() if t.get_text()]
    assert len(ylabels) == len(features)
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_missing — smoke tests
# ---------------------------------------------------------------------------


def test_plot_missing_with_nan(missing_df):
    plot_missing(missing_df)
    plt.close("all")


def test_plot_missing_no_nan_no_plot(binary_df):
    plot_missing(binary_df)
    plt.close("all")


def test_plot_missing_feature_subset(missing_df):
    plot_missing(missing_df, features=["a", "b"])
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_missing — structural assertions
# ---------------------------------------------------------------------------


def test_plot_missing_nan_features_appear_in_chart(missing_df):
    plot_missing(missing_df)
    bar_labels = [t.get_text() for t in plt.gcf().axes[0].get_yticklabels()]
    assert "a" in bar_labels
    assert "b" in bar_labels
    plt.close("all")


def test_plot_missing_clean_feature_excluded_from_chart(missing_df):
    # Column 'c' has no missing values and must not appear
    plot_missing(missing_df)
    bar_labels = [t.get_text() for t in plt.gcf().axes[0].get_yticklabels()]
    assert "c" not in bar_labels
    plt.close("all")


def test_plot_missing_no_figure_created_when_no_nan(binary_df):
    plt.close("all")
    plot_missing(binary_df)
    assert len(plt.get_fignums()) == 0
