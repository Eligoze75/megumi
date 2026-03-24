"""Utility functions for the gyokuken visualization module.

These are internal helpers used by the plotting functions. They handle
data preparation tasks such as binning, rare-label encoding, and type
inference, keeping the plotting functions focused on rendering.
"""

import pandas as pd


def rare_label_encoder(series, threshold=0.05, fill_value="Other"):
    """
    Replace infrequent category labels with a collective label.

    Identifies category labels whose relative frequency in ``series``
    falls below ``threshold`` and replaces them all with ``fill_value``.
    Useful for reducing cardinality before grouping or plotting.

    Parameters
    ----------
    series : pd.Series
        Categorical or object series.
    threshold : float, optional
        Minimum relative frequency to keep a label. Labels appearing
        in less than ``threshold`` fraction of rows are grouped into
        ``fill_value``. Default is 0.05.
    fill_value : str, optional
        Label assigned to rare categories. Default is ``"Other"``.

    Returns
    -------
    pd.Series
        Series with rare labels replaced by ``fill_value``.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["a", "a", "a", "b", "b", "c"])
    >>> rare_label_encoder(s, threshold=0.25)
    0        a
    1        a
    2        a
    3        b
    4        b
    5    Other
    dtype: object
    """
    freq = series.value_counts(normalize=True)
    rare_labels = freq[freq < threshold].index
    return series.where(~series.isin(rare_labels), other=fill_value)


def bin_continuous_feature(series, n_bins=10):
    """
    Bin a continuous series into quantile-based intervals.

    Wraps ``pandas.qcut`` with automatic handling of duplicate bin edges,
    which occur in heavily skewed or concentrated distributions. Falls back
    to equal-width binning via ``pandas.cut`` when quantile binning fails
    even after dropping duplicate edges.

    Parameters
    ----------
    series : pd.Series
        Numeric series to bin.
    n_bins : int, optional
        Number of quantile bins. Default is 10.

    Returns
    -------
    pd.Categorical
        Ordered categorical series of interval labels with shape
        matching the input.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = pd.Series(rng.uniform(0, 1, 100))
    >>> binned = bin_continuous_feature(s, n_bins=5)
    >>> binned.nunique() <= 5
    True
    """
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.cut(series, bins=n_bins, duplicates="drop")


def infer_feature_type(series, max_categories=20):
    """
    Infer whether a feature is continuous or categorical/discrete.

    A series is treated as categorical if it has an object or categorical
    dtype, or if it is numeric but has at most ``max_categories`` unique
    non-null values.

    Parameters
    ----------
    series : pd.Series
        Feature series to inspect.
    max_categories : int, optional
        Maximum number of unique values for a numeric series to be
        treated as categorical/discrete. Default is 20.

    Returns
    -------
    str
        ``'continuous'`` if the feature is numeric with cardinality
        above ``max_categories``, ``'categorical'`` otherwise.

    Examples
    --------
    >>> import pandas as pd
    >>> infer_feature_type(pd.Series([1.1, 2.2, 3.3] * 20))
    'continuous'
    >>> infer_feature_type(pd.Series(["a", "b", "a", "c"]))
    'categorical'
    >>> infer_feature_type(pd.Series([1, 2, 3, 1, 2]))
    'categorical'
    """
    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if series.nunique() <= max_categories:
        return "categorical"
    return "continuous"


def infer_target_type(series, max_multiclass=20):
    """
    Infer the modelling type of a target variable.

    Classification is assumed when the target has few unique values or a
    non-numeric dtype. The distinction between binary and multiclass is
    made by counting unique non-null values. Numeric targets with high
    cardinality are treated as regression targets.

    Parameters
    ----------
    series : pd.Series
        Target variable series.
    max_multiclass : int, optional
        Maximum number of unique values for a numeric series to be
        treated as multiclass rather than continuous. Default is 20.

    Returns
    -------
    str
        One of:

        - ``'binary'`` — exactly 2 unique non-null values.
        - ``'multiclass'`` — object/categorical dtype, or integer with
          3 to ``max_multiclass`` unique values.
        - ``'continuous'`` — numeric with more than ``max_multiclass``
          unique values.

    Examples
    --------
    >>> import pandas as pd
    >>> infer_target_type(pd.Series([0, 1, 0, 1]))
    'binary'
    >>> infer_target_type(pd.Series(["cat", "dog", "bird"]))
    'multiclass'
    >>> import numpy as np
    >>> infer_target_type(pd.Series(np.random.default_rng(0).uniform(0, 1, 100)))
    'continuous'
    """
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary"
    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        return "multiclass"
    if n_unique <= max_multiclass:
        return "multiclass"
    return "continuous"
