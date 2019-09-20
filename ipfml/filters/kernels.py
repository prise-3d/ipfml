"""
Kernel to apply on images using convolution
"""

# main imports
import numpy as np
import sys


def plane_mean(window):
    """Plane mean kernel to use with convolution process on image

    Args:
        window: the window part to use from image

    Returns:
        Normalized residual error from mean plane

    Example:

    >>> from ipfml.filters.kernels import plane_mean
    >>> import numpy as np
    >>> window = np.arange(9).reshape([3, 3])
    >>> result = plane_mean(window)
    >>> (result < 0.0001)
    True
    """

    window = np.array(window)

    width, height = window.shape

    # prepare data
    nb_elem = width * height
    xs = [int(i / height) for i in range(nb_elem)]
    ys = [i % height for i in range(nb_elem)]
    zs = np.array(window).flatten().tolist()

    # get residual (error) from mean plane computed
    tmp_A = []
    tmp_b = []

    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])

    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit = (A.T * A).I * A.T * b

    errors = b - A * fit
    residual = np.linalg.norm(errors)

    return residual


# return difference between min and max errors
def plane_max_error(window):
    """Plane max error kernel to use with convolution process on image

    Args:
        window: the window part to use from image

    Returns:
        Difference between max and min error from mean plane

    Example:

    >>> from ipfml.filters.kernels import plane_max_error
    >>> import numpy as np
    >>> window = np.arange(9).reshape([3, 3])
    >>> result = plane_max_error(window)
    >>> (result < 0.0001)
    True
    """

    window = np.array(window)

    width, height = window.shape

    # prepare data
    nb_elem = width * height
    xs = [int(i / height) for i in range(nb_elem)]
    ys = [i % height for i in range(nb_elem)]
    zs = np.array(window).flatten().tolist()

    # get residual (error) from mean plane computed
    tmp_A = []
    tmp_b = []

    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])

    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit = (A.T * A).I * A.T * b

    errors = b - A * fit

    # get absolute values from errors
    errors = abs(np.array(errors))

    return (errors.max() - errors.min())


def _bilateral_diff(window, func):
    """Main bilateral difference kernel to use with convolution process on image
       Apply difference pixel to pixel and keep max on min difference before applying mean

    Args:
        window: the window part to use from image
        func: max or min function to get difference between pixels

    Returns:
        mean of max or min difference of pixels
    """

    window = np.array(window)

    width, height = window.shape

    total_row_diff_list = []
    total_col_diff_list = []

    for i in range(width):

        row_diff_list = []
        col_diff_list = []

        for j in range(height):

            diff_row = 0

            if i == 0:
                diff_row = abs(window[i][j] - window[i + 1][j])

            elif i == width - 1:
                diff_row = abs(window[i][j] - window[i - 1][j])

            else:
                diff1 = abs(window[i][j] - window[i - 1][j])
                diff2 = abs(window[i][j] - window[i + 1][j])
                diff_row = func(diff1, diff2)

            if j == 0:
                diff_col = abs(window[i][j] - window[i][j + 1])

            elif j == height - 1:
                diff_col = abs(window[i][j] - window[i][j - 1])

            else:
                diff1 = abs(window[i][j] - window[i][j - 1])
                diff2 = abs(window[i][j] - window[i][j + 1])
                diff_col = func(diff1, diff2)

            row_diff_list.append(diff_row)
            col_diff_list.append(diff_col)

        total_row_diff_list.append(sum(row_diff_list) / len(row_diff_list))
        total_col_diff_list.append(sum(col_diff_list) / len(col_diff_list))

    row_diff = sum(total_row_diff_list) / len(total_row_diff_list)
    col_diff = sum(total_col_diff_list) / len(total_col_diff_list)

    return func(row_diff, col_diff)


def max_bilateral_diff(window):
    """Bilateral difference kernel to use with convolution process on image
       Apply difference pixel to pixel and keep max difference before applying mean

    Args:
        window: the window part to use from image

    Returns:
        mean of max difference of pixels

    Example:

    >>> from ipfml.filters.kernels import max_bilateral_diff
    >>> import numpy as np
    >>> window = np.arange(9).reshape([3, 3])
    >>> result = max_bilateral_diff(window)
    >>> result
    3.0
    """

    return _bilateral_diff(window, max)


def min_bilateral_diff(window):
    """Bilateral difference kernel to use with convolution process on image
       Apply difference pixel to pixel and keep min difference before applying mean

    Args:
        window: the window part to use from image

    Returns:
        mean of min difference of pixels

    Example:

    >>> from ipfml.filters.kernels import min_bilateral_diff
    >>> import numpy as np
    >>> window = np.arange(9).reshape([3, 3])
    >>> result = min_bilateral_diff(window)
    >>> result
    1.0
    """

    return _bilateral_diff(window, min)
