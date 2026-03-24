"""gyokuken — visual feature analysis tools.

Named after Megumi Fushiguro's most reliable shikigami (玉犬), the Divine
Dogs, which are used for tracking and sensing. This module provides tools
to track and sense the true nature of your features.

Functions
---------
plot_bivariate :
    Mean target rate per feature bucket with observation counts.
plot_distribution :
    Feature distribution grouped by the target variable.
plot_correlation :
    Lower-triangle correlation heatmap.
plot_missing :
    Horizontal bar chart of missing value percentages.

Utilities
---------
rare_label_encoder :
    Group infrequent category labels into a single collective label.
bin_continuous_feature :
    Quantile-bin a continuous series into labelled intervals.
infer_feature_type :
    Detect whether a feature is continuous or categorical/discrete.
infer_target_type :
    Detect whether a target is binary, multiclass, or continuous.
"""

from .plots import (
    plot_bivariate,
    plot_correlation,
    plot_distribution,
    plot_missing,
)
from .visualization_utils import (
    bin_continuous_feature,
    infer_feature_type,
    infer_target_type,
    rare_label_encoder,
)

__all__ = [
    "plot_bivariate",
    "plot_distribution",
    "plot_correlation",
    "plot_missing",
    "rare_label_encoder",
    "bin_continuous_feature",
    "infer_feature_type",
    "infer_target_type",
]
