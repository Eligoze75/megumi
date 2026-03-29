"""nue — feature contribution analysis.

Named after Megumi Fushiguro's shikigami *Nue* (鵺), a chimeric creature
used to survey from above and strike with precision. This module answers:

*"If I add these features, how much improvement do I get?"*

Two random forests are fitted per cross-validation fold - one on the
base features alone, one on base + candidate features - and their
performance is compared across metrics using a paired t-test. Both
sklearn metrics and user-defined callables are supported, including
business metrics that draw on extra columns in the dataset (e.g. loan
amount for expected-loss calculations in credit risk).

Functions
---------
evaluate_contribution :
    Compare model metrics before and after adding candidate features
    and report whether the improvement is statistically significant.
"""

from .evaluator import evaluate_contribution

__all__ = ["evaluate_contribution"]
