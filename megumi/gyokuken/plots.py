"""
Visualization functions for feature analysis.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .visualization_utils import (
    bin_continuous_feature,
    infer_feature_type,
    infer_target_type,
    rare_label_encoder,
)


def plot_bivariate(
    df,
    feature,
    target,
    n_bins=5,
    rare_threshold=0.05,
    bar_color="#CCCCCC",
    line_color="#0066FF",
    figsize=(10, 7),
    ax=None,
):
    """
    Plot mean target rate per feature bucket alongside observation counts.

    For continuous features the column is split into quantile bins using
    :func:`~megumi.gyokuken.visualization_utils.bin_continuous_feature`.
    For categorical or low-cardinality discrete features, rare labels are
    grouped into ``"Other"`` via
    :func:`~megumi.gyokuken.visualization_utils.rare_label_encoder`.

    The left y-axis shows the mean value of ``target`` per bucket (line),
    together with the overall population mean as a dashed reference line.
    The right y-axis shows the count of observations per bucket (bars).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both ``feature`` and ``target`` columns.
    feature : str
        Name of the feature column to analyse.
    target : str
        Name of the target column. Must not be a multiclass target.
    n_bins : int, optional
        Number of quantile bins for continuous features. Default is 5.
    rare_threshold : float, optional
        Minimum relative frequency before a category label is grouped
        into ``"Other"``. Applies only to categorical features.
        Default is 0.05.
    bar_color : str, optional
        Colour of the count bars. Default is ``"#CCCCCC"``.
    line_color : str, optional
        Colour of the mean-target line and left y-axis. Default is
        ``"#0066FF"``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(12, 5)``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` a new figure is created. A twin
        x-axis is always created internally, so the returned tuple
        always contains two axes objects regardless.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_mean : matplotlib.axes.Axes
        Left y-axis (mean target per bucket).
    ax_count : matplotlib.axes.Axes
        Right y-axis (observation counts per bucket).

    Raises
    ------
    ValueError
        If ``target`` is a multiclass variable. Bivariate analysis
        requires a numeric target (binary or continuous) to compute
        meaningful per-bucket means.

    Notes
    -----
    Missing values in ``feature`` are never silently dropped. Instead
    they form their own ``"Missing"`` bucket appended after all regular
    bins, rendered in a visually distinct colour. This allows the caller
    to inspect whether missingness is informative with respect to the
    target.

    For regression targets a warning is emitted as a reminder that the
    right-axis values represent mean target magnitude rather than an
    event rate, so cross-bucket differences carry a different meaning
    than in the binary case.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "age": rng.integers(18, 70, 500),
    ...     "default": rng.integers(0, 2, 500),
    ... })
    >>> fig, ax_mean, ax_count = plot_bivariate(df, feature="age", target="default")
    """
    target_type = infer_target_type(df[target])

    if target_type == "multiclass":
        raise ValueError(
            f"plot_bivariate does not support multiclass targets."
            f"\n'{target}' has {df[target].nunique()} unique classes.\n"
            "Computing a per-bucket mean is only meaningful for binary or "
            "continuous targets.\nUse plot_distribution to compare feature "
            "distributions across multiple classes."
        )

    if target_type == "continuous":
        warnings.warn(
            f"Target '{target}' appears to be continuous (regression).\n"
            "The right axis shows mean target magnitude per bin, not an "
            "event rate.\nBins with similar counts but different mean values "
            "indicate a predictive relationship.",
            UserWarning,
            stacklevel=2,
        )

    feature_series = df[feature]
    target_series = df[target]

    non_null_mask = feature_series.notna()
    feature_type = infer_feature_type(feature_series[non_null_mask])

    if feature_type == "continuous":
        buckets = bin_continuous_feature(
            feature_series[non_null_mask], n_bins=n_bins
        )
    else:
        buckets = rare_label_encoder(
            feature_series[non_null_mask].astype(str), threshold=rare_threshold
        )

    temp = pd.DataFrame(
        {"bucket": buckets, "target": target_series.loc[buckets.index]}
    )

    agg = temp.groupby("bucket", observed=True).agg(
        count=("target", "count"),
        mean_target=("target", "mean"),
    )
    agg.index = agg.index.astype(str)

    null_mask = feature_series.isna()
    has_missing = null_mask.any()
    if has_missing:
        missing_targets = target_series[null_mask]
        missing_row = pd.DataFrame(
            {
                "count": [int(null_mask.sum())],
                "mean_target": [missing_targets.mean()],
            },
            index=["Missing"],
        )
        agg = pd.concat([agg, missing_row])

    overall_mean = df[target].mean()

    if ax is None:
        fig, ax_mean = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        ax_mean = ax

    ax_count = ax_mean.twinx()

    x = np.arange(len(agg))

    bar_colors = [
        "#A0A0A0" if label == "Missing" else bar_color
        for label in agg.index
    ]
    ax_count.bar(x, agg["count"], color=bar_colors, alpha=0.45, label="Count", zorder=1)
    ax_count.set_ylabel("Observation Count", fontsize=11)

    ax_mean.plot(
        x,
        agg["mean_target"],
        color=line_color,
        marker="o",
        linewidth=2,
        label=f"Mean {target}",
        zorder=3,
    )
    ax_mean.axhline(
        overall_mean,
        color="#FF9900",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        label=f"Population mean {target} ({overall_mean:.3f})",
        zorder=3,
    )
    ax_mean.set_ylabel(f"Mean {target}", fontsize=11)
    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels(agg.index, rotation=45, ha="right", fontsize=9)
    ax_mean.set_xlabel(feature, fontsize=11)
    ax_mean.set_zorder(ax_count.get_zorder() + 1)
    ax_mean.patch.set_visible(False)

    ax_mean.set_title(
        f"{feature}  vs.  mean {target}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    handles_l, labels_l = ax_mean.get_legend_handles_labels()
    handles_r, labels_r = ax_count.get_legend_handles_labels()
    ax_mean.legend(
        handles_l + handles_r,
        labels_l + labels_r,
        loc="upper left",
        fontsize=9,
        framealpha=0.8,
    )

    fig.tight_layout()
    plt.show()


def plot_distribution(
    df,
    feature,
    target,
    kind="histogram",
    n_target_bins=4,
    figsize=(10, 5),
    palette=None,
    ax=None,
):
    """
    Plot the distribution of a feature grouped by the target variable.

    Supports four plot types controlled by ``kind``. The target is
    automatically handled according to its inferred type (binary,
    multiclass, or continuous):

    - ``'binary'`` / ``'multiclass'`` — target values are used directly
      as group labels.
    - ``'continuous'`` — the target is binned into ``n_target_bins``
      quantile groups before colouring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both ``feature`` and ``target`` columns.
    feature : str
        Name of the feature column whose distribution is plotted.
    target : str
        Name of the target column used for grouping.
    kind : {'histogram', 'kde', 'violin', 'boxplot'}, optional
        Type of distribution plot. Default is ``'histogram'``.
    n_target_bins : int, optional
        Number of quantile bins used when ``target`` is continuous.
        Default is 4.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(10, 5)``.
    palette : str or sequence, optional
        Seaborn/matplotlib colour palette. If ``None`` the seaborn
        default is used.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported plot types.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "income": rng.normal(50_000, 15_000, 300),
    ...     "default": rng.integers(0, 2, 300),
    ... })
    >>> fig, ax = plot_distribution(df, feature="income", target="default")
    """
    _VALID_KINDS = {"histogram", "kde", "violin", "boxplot"}
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
        )

    target_type = infer_target_type(df[target])

    if target_type == "continuous":
        target_col = pd.qcut(
            df[target], q=n_target_bins, duplicates="drop"
        ).astype(str)
        target_label = f"{target} (quantile bins)"
    else:
        target_col = df[target].astype(str)
        target_label = target

    plot_df = df[[feature]].copy()
    plot_df["_target"] = target_col.values

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if kind == "histogram":
        sns.histplot(
            data=plot_df,
            x=feature,
            hue="_target",
            palette=palette,
            ax=ax,
            alpha=0.55,
        )
    elif kind == "kde":
        sns.kdeplot(
            data=plot_df,
            x=feature,
            hue="_target",
            palette=palette,
            ax=ax,
            fill=True,
            alpha=0.35,
        )
    elif kind == "violin":
        n_groups = plot_df["_target"].nunique()
        effective_palette = palette if palette is not None else sns.color_palette(n_colors=n_groups)
        sns.violinplot(
            data=plot_df,
            x="_target",
            y=feature,
            hue="_target",
            palette=effective_palette,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel(target_label, fontsize=11)
        # ax.tick_params(axis="x", rotation=30)
    elif kind == "boxplot":
        n_groups = plot_df["_target"].nunique()
        effective_palette = palette if palette is not None else sns.color_palette(n_colors=n_groups)
        sns.boxplot(
            data=plot_df,
            x="_target",
            y=feature,
            hue="_target",
            palette=effective_palette,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel(target_label, fontsize=11)
        # ax.tick_params(axis="x", rotation=30)

    ax.set_title(
        f"Distribution of  {feature}  by  {target_label}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    if kind in {"histogram", "kde"}:
        ax.set_xlabel(feature, fontsize=11)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(target_label)

    fig.tight_layout()
    plt.show()
    


def plot_correlation(
    df,
    features=None,
    target=None,
    method="pearson",
    palette="Wistia",
    figsize=None,
    ax=None,
):
    """
    Plot a lower-triangle correlation heatmap for numeric features.

    Computes pairwise correlations among ``features`` (or all numeric
    columns) and renders an annotated seaborn heatmap. The upper triangle
    is masked to avoid redundancy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns are correlated.
    features : list of str or None, optional
        Subset of column names to include. If ``None`` all numeric
        columns in ``df`` are used. Default is ``None``.
    target : str or None, optional
        Name of the target column. If provided and the target is
        multiclass, a ``ValueError`` is raised. When ``features`` is
        ``None`` the target column is automatically excluded from the
        correlation matrix. Default is ``None``.
    method : {'pearson', 'spearman', 'kendall'}, optional
        Correlation coefficient to compute. Default is ``'pearson'``.
    palette : str, optional
        Matplotlib colormap name for the heatmap. Default is
        ``'Wistia'``.
    figsize : tuple of float or None, optional
        Figure size ``(width, height)`` in inches. If ``None`` the size
        is derived automatically from the number of features.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``target`` is provided and is a multiclass variable.
        Pairwise linear correlation is not meaningful when the target
        has more than two unordered classes.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("ABCD"))
    >>> fig, ax = plot_correlation(df)
    """
    if target is not None:
        target_type = infer_target_type(df[target])
        if target_type == "multiclass":
            raise ValueError(
                f"plot_correlation does not support multiclass targets.\n"
                f"'{target}' has {df[target].nunique()} unique classes.\n"
                "Pairwise correlation is not meaningful for unordered "
                "multi-class targets.\nUse plot_distribution to compare "
                "feature distributions across classes."
            )

    numeric_df = df.select_dtypes(include=np.number)
    if features is not None:
        numeric_df = numeric_df[features]
    elif target is not None and target in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target])

    corr = numeric_df.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    n = len(corr)
    if figsize is None:
        figsize = (max(8, n * 0.8), max(6, n * 0.7))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        corr,
        mask=mask,
        cmap=palette,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(
        f"Correlation Matrix  ({method.capitalize()})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()
    plt.show()
    


def plot_missing(
    df,
    features=None,
    figsize=None,
    ax=None,
):
    """
    Plot the percentage of missing values per feature as a horizontal bar chart.

    Only features with at least one missing value are shown. Bars are
    sorted in descending order of missingness so the most problematic
    features appear at the top.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect for missing values.
    features : list of str or None, optional
        Subset of column names to inspect. If ``None`` all columns in
        ``df`` are used. Default is ``None``.
    figsize : tuple of float or None, optional
        Figure size ``(width, height)`` in inches. If ``None`` the
        height is derived from the number of features with missing values.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        ``None`` if no missing values are found.
    ax : matplotlib.axes.Axes or None
        ``None`` if no missing values are found.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "a": [1, None, 3, None, 5],
    ...     "b": [None, 2, 3, 4, 5],
    ...     "c": [1, 2, 3, 4, 5],
    ... })
    >>> fig, ax = plot_missing(df)
    """
    data = df if features is None else df[features]

    missing_pct = (data.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        return

    missing_pct = missing_pct.sort_values(ascending=True)

    n = len(missing_pct)
    if figsize is None:
        figsize = (10, max(4, n * 0.45))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.barh(
        missing_pct.index,
        missing_pct.values,
        color="#4D4D4D",
        # alpha=0.9,
        edgecolor="white",
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Missing Values (%)", fontsize=11)
    ax.set_title("Missing Values by Feature", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, min(100, missing_pct.max() * 1.18))
    ax.axvline(x=5, color="#9400D3", linestyle="--", linewidth=1.5, alpha=0.7, label="5% threshold")
    ax.axvline(x=25, color="#FF9900", linestyle="--", linewidth=1, alpha=0.7, label="25% threshold")
    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    plt.show()
    
