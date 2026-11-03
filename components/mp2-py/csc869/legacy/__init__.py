"""MP1 legacy package: Naive Bayes, k-fold CV, and metrics."""

from .naive_bayes import MixedNaiveBayes, run_legacy_cv

__all__ = ["MixedNaiveBayes", "run_legacy_cv"]
