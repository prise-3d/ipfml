"""
Utils functions of ipfml package (array normalization)
"""

import numpy as np
import math
import sys

from scipy.integrate import simps


def normalize_arr(arr):
    """Normalize data of 1D array shape

    Args:
        arr: array data of 1D shape

    Returns:
        Normalized 1D array

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(11)
    >>> arr_normalized = utils.normalize_arr(arr)
    >>> arr_normalized[1]
    0.1
    """

    output_arr = []
    max_value = max(arr)
    min_value = min(arr)

    for v in arr:
        try:
            output_arr.append((v - min_value) / (max_value - min_value))
        except ZeroDivisionError:
            output_arr.append((v - min_value) /
                              (max_value - min_value + sys.float_info.epsilon))

    return output_arr


def normalize_arr_with_range(arr, min, max):
    '''Normalize data of 1D array shape

    Args:
        arr: array data of 1D shape

    Returns:
        Normalized 1D Numpy array

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(11)
    >>> arr_normalized = utils.normalize_arr_with_range(arr, 0, 20)
    >>> arr_normalized[1]
    0.05
    '''

    output_arr = []

    for v in arr:
        try:
            output_arr.append((v - min) / (max - min))
        except ZeroDivisionError:
            output_arr.append((v - min) / (max - min + sys.float_info.epsilon))

    return output_arr


def normalize_2D_arr(arr):
    """Return array normalize from its min and max values

    Args:
        arr: 2D Numpy array

    Returns:
        Normalized 2D Numpy array

    Example:

    >>> from PIL import Image
    >>> from ipfml import utils, processing
    >>> img = Image.open('./images/test_img.png')
    >>> img_mscn = processing.rgb_to_mscn(img)
    >>> img_normalized = utils.normalize_2D_arr(img_mscn)
    >>> img_normalized.shape
    (200, 200)
    """

    # getting min and max value from 2D array
    max_value = arr.max(axis=1).max()
    min_value = arr.min(axis=1).min()

    # normalize each row
    output_array = []
    width, height = arr.shape

    for row_index in range(0, height):
        values = arr[row_index, :]
        output_array.append(
            normalize_arr_with_range(values, min_value, max_value))

    return np.asarray(output_array)


def integral_area_trapz(y_values, dx):
    """Returns area under curves from provided data points using Trapezium rule

    Args:
        y_values: y values of curve
        dx: number of unit for x axis

    Returns:
        Area under curves obtained from these points

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> y_values = np.array([5, 20, 4, 18, 19, 18, 7, 4])
    >>> area = utils.integral_area_trapz(y_values, dx=5)
    >>> area
    452.5
    """

    return np.trapz(y_values, dx=dx)


def integral_area_simps(y_values, dx):
    """Returns area under curves from provided data points using Simpsons rule

    Args:
        y_values: y values of curve
        dx: number of unit for x axis

    Returns:
        Area under curves obtained from these points

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> y_values = np.array([5, 20, 4, 18, 19, 18, 7, 4])
    >>> area = utils.integral_area_simps(y_values, dx=5)
    >>> area
    460.0
    """

    return simps(y_values, dx=dx)


def get_indices_of_highest_values(arr, n):
    """Returns indices of n highest values from list or 1D numpy array

    Args:
        arr: List of numpy array
        n: number of highest elements wanted

    Returns:
        `n` indices of highest values

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> indices = utils.get_indices_of_highest_values(arr, 2)
    >>> indices
    array([9, 8])
    """
    return np.array(arr).argsort()[-n:][::-1]


def get_indices_of_lowest_values(arr, n):
    """Returns indices of n highest values from list or 1D numpy array

    Args:
        arr: List of numpy array
        n: number of highest elements wanted

    Returns:
        `n` indices of highest values

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> indices = utils.get_indices_of_lowest_values(arr, 2)
    >>> indices
    array([0, 1])
    """
    return np.array(arr).argsort()[::-1][-n:][::-1]


def get_entropy(arr):
    """Returns the computed entropy from arr

    Args:
        arr: numpy array

    Returns:
        entropy score computed

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> entropy = utils.get_entropy(arr)
    >>> int(entropy)
    0
    """

    arr = np.array(arr)
    eigen_values = []
    sum_eigen_values = (arr * arr).sum()

    for val in arr:
        eigen_values.append(val * val)

    v = []

    for val in eigen_values:
        v.append(val / sum_eigen_values)

    entropy = 0

    for val in v:
        if val > 0:
            entropy += val * math.log(val)

    entropy *= -1

    entropy /= math.log(len(v))

    return entropy


def get_entropy_without_i(arr, i):
    """Returns the computed entropy from arr without contribution of i

    Args:
        arr: numpy array
        i: column index

    Returns:
        entropy score computed

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> entropy = utils.get_entropy_without_i(arr, 3)
    >>> int(entropy)
    0
    """

    arr = np.array([v for index, v in enumerate(arr) if index != i])

    return get_entropy(arr)


def get_entropy_contribution_of_i(arr, i):
    """Returns the entropy contribution i column

    Args:
        arr: numpy array
        i: column index

    Returns:
        entropy contribution score computed

    Example:

    >>> from ipfml import utils
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> entropy = utils.get_entropy_contribution_of_i(arr, 3)
    >>> int(entropy)
    0
    """

    return get_entropy(arr) - get_entropy_without_i(arr, i)
