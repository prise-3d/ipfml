from skimage.measure import compare_ssim, compare_psnr
from ipfml.exceptions import NumpyShapeComparisonException

"""
Full-reference Image Quality Assessment (FR-IQA) methods
"""

def _prepare_arrays(arr1, arr2):
    img_true = np.asarray(img_true, dtype=float_32)
    img_test = np.asarray(img_test, dtype=float_32)

    if img_true.shape != img_test.shape:
        raise NumpyShapeComparisonException

    return img_true, img_test


def mse(img_true, img_test):
    """Returns Mean-Squared Error between two Numpy arrays

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed MSE score

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same

    Example:
        >>> from ipfml import utils
        >>> import numpy as np
        >>> arr1 = np.arange(10)
        >>> arr2 = np.arange(5, 10)
        >>> mse = utils.mse(arr1, arr2)
        >>> mse
        100
    """

    img_true, img_test = _prepare_arrays(img_true, img_test)

    return np.mean(np.square(img_true - img_test), dtype=np.float64)


def rmse(img_true, img_test):
    """Returns Mean-Squared Error between two Numpy arrays

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed MSE score

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same

    Example:
        >>> from ipfml import utils
        >>> import numpy as np
        >>> arr1 = np.arange(10)
        >>> arr2 = np.arange(5, 10)
        >>> rmse = utils.rmse(arr1, arr2)
        >>> rmse
        100
    """

    return np.sqrt(mse(img_true, img_test))


def mae(img_true, img_test):
    """Returns Mean-Squared Error between two Numpy arrays

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed MSE score

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same

    Example:
        >>> from ipfml import utils
        >>> import numpy as np
        >>> arr1 = np.arange(10)
        >>> arr2 = np.arange(5, 10)
        >>> mse = utils.mse(arr1, arr2)
        >>> mse
        100
    """

    img_true, img_test = _prepare_arrays(img_true, img_test)

    return np.mean(np.absolute(img_true - img_test), dtype=np.float64)


def pnsr(img_true, img_test):
    """Returns the computed Peak Signal to Noise Ratio (PSNR) between two images

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed PSNR score
    """

    return compare_psnr(img_true, img_test)


def ms_ssim(img_true, img_test):
    """
    Implemented later..
    """
    pass


def vif(img_true, img_test):
    """
    Implemented later..
    """
    pass
