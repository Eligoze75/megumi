# megumi

**megumi** is a Python package for feature selection in machine learning workflows.

Inspired by *Jujutsu Kaisen's* Megumi Fushiguro, this package helps you to work strategically: not about using every feature available, but about choosing the right ones.

---

## What it does

Feature selection is one of the most impactful steps in building a machine learning model, and also one of the easiest to rush. `megumi` gives you the tools to do it properly, across three areas:

**1. Visual exploration:** Understand your features before modelling. See how each feature relates to the target, spot distributions that separate classes, catch missing value patterns, and identify multicollinearity — all in a single function call.

**2. Importance scoring:** *(coming soon)* Go beyond intuition. Use machine learning-based methods to quantify which features actually carry predictive power.

**3. Contribution analysis:** *(coming soon)* Understand what each feature adds to your model and whether keeping it improves performance in practice.

---

## Modules

### `gyokuken` — Visual feature analysis

Named after Megumi's shikigami *gyokuken* (玉犬, the Divine Dogs), used for tracking and sensing. This module helps you track and sense the true nature of your features.

| Function | Description |
|---|---|
| `plot_bivariate` | Mean target rate per feature bucket vs. observation counts. Adapts automatically to continuous and categorical features. Missing values get their own bucket. |
| `plot_distribution` | Feature distribution grouped by target. Supports histogram, KDE, violin, and boxplot. |
| `plot_correlation` | Lower-triangle correlation heatmap. Supports Pearson, Spearman, and Kendall. |
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

`megumi` is under active development. The `gyokuken` visual module is the first of several planned modules. Contributions and feedback are welcome.

---

> *"With this treasure, I summon..."*

amazing feature selection?
