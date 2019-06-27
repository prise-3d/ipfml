"""
Contains all metrics used for model results comparisons
"""

# main imports
import numpy as np


def coefficient_of_determination(y, predicted):
    """Returns the coefficient of determination between `y` and `y_predicted` values

    Args:
        y: array of reference values
        predicted: array of predicted values

    Returns:
        coefficient of determination

    Example:

    >>> from ipfml import metrics
    >>> import numpy as np
    >>> y = np.arange(10)
    >>> predicted = np.arange(10) * 1.05
    >>> coeff = metrics.coefficient_of_determination(y, predicted)
    >>> int(coeff)
    1
    """

    y = np.asarray(y)
    predicted = np.asarray(predicted)

    y_mean = y.mean()

    numerator_sum = 0
    denominator_sum = 0

    for id_val, val in enumerate(y):
        numerator_sum += (predicted[id_val] - y_mean) * (
            predicted[id_val] - y_mean)
        denominator_sum += (val - y_mean) * (val - y_mean)

    return numerator_sum / denominator_sum
