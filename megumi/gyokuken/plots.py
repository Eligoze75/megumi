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
    Plot mean target rate per feature bucket with observation counts.

    Left axis: mean ``target`` per bucket (line) + population mean reference.
    Right axis: observation count per bucket (bars). Continuous features are
    quantile-binned; categoricals have rare labels grouped into ``"Other"``.
    Missing values get their own bucket. Raises ``ValueError`` for multiclass
    targets; emits a warning for continuous targets.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
    target : str
        Binary or continuous target column.
    n_bins : int, optional
        Quantile bins for continuous features. Default is 5.
    rare_threshold : float, optional
        Frequency below which a category is grouped into ``"Other"``.
        Default is 0.05.
    bar_color : str, optional
        Default is ``"#CCCCCC"``.
    line_color : str, optional
        Default is ``"#0066FF"``.
    figsize : tuple of float, optional
        Default is ``(10, 7)``.
    ax : matplotlib.axes.Axes or None, optional

    Returns
    -------
    fig, ax_mean, ax_count
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
    Plot feature distribution grouped by target.

    Supports ``histogram``, ``kde``, ``violin``, and ``boxplot``. Binary and
    multiclass targets are used as group labels directly; continuous targets
    are quantile-binned before colouring. Raises ``ValueError`` for unknown
    ``kind`` values.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
    target : str
    kind : {'histogram', 'kde', 'violin', 'boxplot'}, optional
        Default is ``'histogram'``.
    n_target_bins : int, optional
        Quantile bins when target is continuous. Default is 4.
    figsize : tuple of float, optional
        Default is ``(10, 5)``.
    palette : str or sequence, optional
    ax : matplotlib.axes.Axes or None, optional

    Returns
    -------
    fig, ax
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

    Upper triangle is masked to avoid redundancy. When ``features`` is
    ``None``, all numeric columns are used and ``target`` is excluded
    automatically. Raises ``ValueError`` if ``target`` is multiclass.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str or None, optional
        Columns to include. Default is ``None`` (all numeric columns).
    target : str or None, optional
        Excluded from the matrix when ``features`` is ``None``.
    method : {'pearson', 'spearman', 'kendall'}, optional
        Default is ``'pearson'``.
    palette : str, optional
        Default is ``'Wistia'``.
    figsize : tuple of float or None, optional
        Auto-sized from feature count if ``None``.
    ax : matplotlib.axes.Axes or None, optional

    Returns
    -------
    fig, ax
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
    Plot missing value percentages per feature as a horizontal bar chart.

    Only features with at least one missing value are shown, sorted by
    severity. Returns ``None, None`` silently if no missingness is found.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str or None, optional
        Columns to inspect. Default is ``None`` (all columns).
    figsize : tuple of float or None, optional
        Auto-sized from feature count if ``None``.
    ax : matplotlib.axes.Axes or None, optional

    Returns
    -------
    fig, ax — both ``None`` if no missing values are found.
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
    
