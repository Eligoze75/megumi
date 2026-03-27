"""bansho — SHAP-based feature importance scoring.

Named after Megumi Fushiguro's shikigami *Banshō* (万象, Max Elephant) —
a heavy, water-releasing shikigami. This module uses machine learning and
SHAP values to flood the feature space and reveal which features carry
real predictive power.

Two synthetic random features are introduced as baselines before fitting
a random forest. Features are ranked by their mean absolute SHAP value
and labelled relative to those baselines:

- ``"predictive"``  — beats both random features
- ``"marginal"``    — beats one random feature
- ``"noise"``       — beats neither random feature

Functions
---------
score_features :
    Rank input features by SHAP importance and label each one relative
    to two random baselines.
"""

from .scorer import score_features

__all__ = ["score_features"]
