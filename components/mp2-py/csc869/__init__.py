"""CSC 869 Mini Project 2 -- Python implementation.

The package is deliberately split into small, testable modules:

``csc869.data``       raw Adult loading, cleaning, fold generation
``csc869.preprocess`` leak-free preprocessing (fit inside each fold)
``csc869.sampling``   none / upsampling / downsampling inside a fold
``csc869.models``     the five classifier builders and DT tuning
``csc869.metrics``    accuracy, precision, recall, F1, ROC-AUC, PR-AUC
``csc869.cv``         the 5-fold experiment matrix
``csc869.curves``     ROC / PR / learning-curve figures
``csc869.test_eval``  full-training-set evaluation on adult.test
``csc869.error_analysis``  200-record error analysis and Venn diagram
``csc869.eda``        exploratory figures
``csc869.legacy``     MP1 Naive Bayes + k-fold + metrics carried forward
"""

__version__ = "1.0.0"

POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
