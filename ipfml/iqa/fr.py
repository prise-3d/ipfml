"""
Full-reference Image Quality Assessment (FR-IQA) methods
"""

from skimage.measure import compare_ssim, compare_psnr
from ipfml.exceptions import NumpyShapeComparisonException

import numpy as np


def _prepare_arrays(img_true, img_test, p_dtype='float32'):
    """
    Prepare image data

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same
    """

    img_true = np.asarray(img_true, dtype=p_dtype)
    img_test = np.asarray(img_test, dtype=p_dtype)

    if img_true.shape != img_test.shape:
        raise NumpyShapeComparisonException

    return img_true, img_test


def mse(img_true, img_test):
    """Returns Mean-Squared Error score between two Numpy arrays

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed MSE score

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same

    Example:
        >>> from ipfml.iqa import fr
        >>> import numpy as np
        >>> arr1 = np.arange(10)
        >>> arr2 = np.arange(5, 15)
        >>> mse_score = fr.mse(arr1, arr2)
        >>> mse_score
        25.0
    """

    img_true, img_test = _prepare_arrays(img_true, img_test)

    return np.mean(np.square(img_true - img_test), dtype=np.float64)


def rmse(img_true, img_test):
    """Returns Root Mean-Squared Error score between two Numpy arrays

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed RMSE score

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same

    Example:
        >>> from ipfml.iqa import fr
        >>> import numpy as np
        >>> arr1 = np.arange(10)
        >>> arr2 = np.arange(5, 15)
        >>> rmse_score = fr.rmse(arr1, arr2)
        >>> rmse_score
        5.0
    """

    return np.sqrt(mse(img_true, img_test))


def mae(img_true, img_test):
    """Returns Mean Absolute Error between two Numpy arrays

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed MAE score

    Raises:
        NumpyShapeComparisonException: if shape of images are not the same

    Example:
        >>> from ipfml.iqa import fr
        >>> import numpy as np
        >>> arr1 = np.arange(10)
        >>> arr2 = np.arange(5, 15)
        >>> mae_score = fr.mae(arr1, arr2)
        >>> mae_score
        5.0
    """

    img_true, img_test = _prepare_arrays(img_true, img_test)

    return np.mean(np.absolute(img_true - img_test), dtype=np.float64)


def psnr(img_true, img_test):
    """Returns the computed Peak Signal to Noise Ratio (PSNR) between two images

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed PSNR score

    Example:
        >>> from ipfml.iqa import fr
        >>> import numpy as np
        >>> arr1 = np.arange(100).reshape(10, 10)
        >>> arr2 = np.arange(5, 105).reshape(10, 10)
        >>> psnr_score = fr.psnr(arr1, arr2)
        >>> int(psnr_score)
        34
    """

    img_true, img_test = _prepare_arrays(img_true, img_test, 'uint8')

    return compare_psnr(img_true, img_test)


def ssim(img_true, img_test):
    """Returns the computed Structural Similarity (SSIM) between two images

    Args:
        img_true: Image, numpy array of any dimension
        img_test: Image, numpy array of any dimension

    Returns:
        Computed SSIM score

    Example:
        >>> from ipfml.iqa import fr
        >>> import numpy as np
        >>> arr1 = np.arange(100).reshape(10, 10)
        >>> arr2 = np.arange(5, 105).reshape(10, 10)
        >>> ssim_score = fr.ssim(arr1, arr2)
        >>> int(ssim_score)
        0
    """

    img_true, img_test = _prepare_arrays(img_true, img_test)

    if img_true.ndim == 3:
        return compare_ssim(img_true, img_test, multichannel=True)
    else:
        return compare_ssim(img_true, img_test)


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
