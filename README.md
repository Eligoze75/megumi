# megumi

[![PyPI version](https://img.shields.io/pypi/v/megumi.svg)](https://pypi.org/project/megumi/)
[![Python versions](https://img.shields.io/pypi/pyversions/megumi.svg)](https://pypi.org/project/megumi/)
[![CI](https://github.com/Eligoze75/megumi/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Eligoze75/megumi/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/Eligoze75/megumi/branch/master/graph/badge.svg)](https://codecov.io/gh/Eligoze75/megumi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**megumi** is a Python package for feature selection in machine learning workflows.

Inspired by *Jujutsu Kaisen's* Megumi Fushiguro, this package helps you to work strategically: not about using every feature available, but about choosing the right ones.

---

## What it does

Feature selection is one of the most impactful steps in building a machine learning model, and also one of the easiest to rush. `megumi` gives you the tools to do it properly, across three areas:

**1. Visual exploration:** Understand your features before modelling. See how each feature relates to the target, spot distributions that separate classes, catch missing value patterns, and identify multicollinearity, all in a single function call.

**2. Importance scoring:** Go beyond intuition. Use machine learning-based methods to quantify which features actually carry predictive power.

**3. Contribution analysis:** *(coming soon)* Understand what each feature adds to your model and whether keeping it improves performance in practice.

---

## Modules

### `gyokuken` - Visual feature analysis

Named after Megumi's shikigami *gyokuken* (玉犬, the Divine Dogs), used for tracking and sensing. This module helps you track and sense the true nature of your features.

| Function | Description |
|---|---|
| `plot_bivariate` | Mean target rate per feature bucket vs. observation counts. Adapts automatically to continuous and categorical features. Missing values get their own bucket. |
| `plot_distribution` | Feature distribution grouped by target. Supports histogram, KDE, violin, and boxplot. |
| `plot_correlation` | Lower triangle correlation heatmap. Supports Pearson, Spearman, and Kendall. |
| `plot_missing` | Horizontal bar chart of missing value percentages, sorted by severity. |

Usage example:

```python
from megumi.gyokuken import plot_bivariate, plot_distribution, plot_correlation, plot_missing

plot_bivariate(df, feature="age", target="default")
plot_distribution(df, feature="income", target="default", kind="violin")
plot_correlation(df)
plot_missing(df)
```

All visualisations adapt to the target type automatically: binary classification, multiclass classification, or regression.

### `bansho`: SHAP-based feature importance scoring

Named after Megumi's shikigami *Banshō* (万象, Max Elephant) (a heavy, water releasing shikigami). This module uses machine learning and SHAP values to reveal which features carry real predictive power.

Two synthetic random features (`RANDOM_1`, `RANDOM_2`) are introduced as baselines before fitting a vanilla machine learning model. Every input feature is ranked by its mean absolute SHAP value and labelled in relation to those baselines:

| Label | Meaning |
|---|---|
| `predictive` | Mean \|SHAP\| beats both random features - a genuinely informative feature. |
| `marginal` | Mean \|SHAP\| beats one random feature - weak signal, use with caution. |
| `noise` | Mean \|SHAP\| beats neither random feature - no detectable predictive power. |

| Function | Description |
|---|---|
| `score_features` | Fit a vanilla model, compute SHAP values, and return a ranked DataFrame of features labelled by predictive power. |

Usage example:

```python
from sklearn.model_selection import train_test_split
from megumi.bansho import score_features

df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

result = score_features(df_train, features=["age", "income", "zip"], target="default",
                        df_val=df_val, random_state=42)
# returns:
#      feature predictive_power
# 0     income       predictive
# 1        age         marginal
# 2        zip            noise
```

Passing `df_val` is recommended: the forest is fitted on the training set and SHAP values are computed on the held-out set, producing more conservative importance estimates. If omitted, SHAP is computed on the training set directly.

Supports binary classification and regression targets. The `strategy` parameter is reserved for future model types (e.g. `"linear"`); currently only `"tree"` (random forest) is available.

---

## Installation

```bash
pip install megumi
```

Or, to set up a development environment using conda:

```bash
conda env create -f environment.yml
conda activate megumi-dev
```

---

## Status

`megumi` is under active development. Two modules are available: `gyokuken` for visual feature exploration and `bansho` for SHAP-based importance scoring. A third module for contribution analysis is planned. Contributions and feedback are welcome.

---

> *"With this treasure, I summon..."*

amazing feature selection?
