"""
Functions which can be used to extract information from image
"""

from numpy.linalg import svd
from scipy import misc

import numpy as np
from sklearn import preprocessing
from skimage import io, color

import cv2


def get_SVD(image):
    """Transforms Image using SVD compression

    Args:
        image: image to convert into SVD compression

    Return:
        U, s, V obtained from SVD compression

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> U, s, V = metrics.get_SVD(img)
    >>> U.shape
    (200, 200, 3)
    >>> len(s)
    200
    >>> V.shape
    (200, 3, 3)
    """
    return svd(image, full_matrices=False)


def get_SVD_s(image):
    """Transforms Image into SVD and returns only 's' part

    Args:
        image: image to convert

    Returns:
        vector of singular values obtained from SVD compression

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> s = metrics.get_SVD_s(img)
    >>> len(s)
    200
    """
    U, s, V = svd(image, full_matrices=False)
    return s


def get_SVD_U(image):
    """Transforms Image into SVD and returns only 'U' part

    Args:
        image: image to convert

    Returns:
        U matrix from SVD compression

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> U = metrics.get_SVD_U(img)
    >>> U.shape
    (200, 200, 3)
    """

    U, s, V = svd(image, full_matrices=False)
    return U


def get_SVD_V(image):
    """Transforms Image into SVD and returns only 'V' part

    Args:
        image: image to convert

    Returns:
        V matrix obtained from SVD compression

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> V = metrics.get_SVD_V(img)
    >>> V.shape
    (200, 3, 3)
    """

    U, s, V = svd(image, full_matrices=False)
    return V


def get_LAB(image):
    """Transforms RGB  Image into Lab

    Args:
        image: image to convert

    Returns:
        Lab information

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> Lab = metrics.get_LAB(img)
    >>> Lab.shape
    (200, 200, 3)
    """

    return color.rgb2lab(image)


def get_LAB_L(image):
    """Transforms RGB Image into Lab and returns L

    Args:
        image: image to convert

    Returns:
        The L chanel from Lab information

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> L = metrics.get_LAB_L(img)
    >>> L.shape
    (200, 200)
    """

    lab = get_LAB(image)
    return lab[:, :, 0]


def get_LAB_a(image):
    """Transforms RGB Image into LAB and returns a

    Args:
        image: image to convert

    Returns:
        The a chanel from Lab information

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> a = metrics.get_LAB_a(img)
    >>> a.shape
    (200, 200)
    """

    lab = get_LAB(image)
    return lab[:, :, 1]


def get_LAB_b(image):
    """Transforms RGB Image into LAB and returns b

    Args:
        image: image to convert

    Returns:
        The b chanel from Lab information

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> b = metrics.get_LAB_b(img)
    >>> b.shape
    (200, 200)
    """

    lab = get_LAB(image)
    return lab[:, :, 2]


def get_XYZ(image):
    """Transforms RGB Image into XYZ

    Args:
        image: image to convert

    Returns:
        XYZ information obtained from transformation

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> metrics.get_XYZ(img).shape
    (200, 200, 3)
    """

    return color.rgb2xyz(image)


def get_XYZ_X(image):
    """Transforms RGB Image into XYZ and returns X

    Args:
        image: image to convert

    Returns:
        The X chanel from XYZ information

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> x = metrics.get_XYZ_X(img)
    >>> x.shape
    (200, 200)
    """

    xyz = color.rgb2xyz(image)
    return xyz[:, :, 0]


def get_XYZ_Y(image):
    """Transforms RGB Image into XYZ and returns Y

    Args:
        image: image to convert

    Returns:
        The Y chanel from XYZ information

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> y = metrics.get_XYZ_Y(img)
    >>> y.shape
    (200, 200)
    """

    xyz = color.rgb2xyz(image)
    return xyz[:, :, 1]


def get_XYZ_Z(image):
    """Transforms RGB Image into XYZ and returns Z

    Args:
        image: image to convert

    Returns:
        The Z chanel from XYZ information

    Raises:
        ValueError: If `nb_bits` has unexpected value. `nb_bits` needs to be in interval [1, 8].

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> z = metrics.get_XYZ_Z(img)
    >>> z.shape
    (200, 200)
    """

    xyz = color.rgb2xyz(image)
    return xyz[:, :, 2]


def get_low_bits_img(image, nb_bits=4):
    """Returns Image or Numpy array with data information reduced using only low bits

    Args:
        image: image to convert
        nb_bits: optional parameter which indicates the number of bits to keep

    Returns:
        Numpy array with reduced values

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_img = metrics.get_low_bits_img(img, 5)
    >>> low_bits_img.shape
    (200, 200, 3)
    """

    if nb_bits <= 0:
        raise ValueError(
            "unexpected value of number of bits to keep. @nb_bits needs to be positive and greater than 0."
        )

    if nb_bits > 8:
        raise ValueError(
            "Unexpected value of number of bits to keep. @nb_bits needs to be in interval [1, 8]."
        )

    img_arr = np.array(image)

    bits_values = sum([pow(2, i - 1) for i in range(1, nb_bits + 1)])

    return img_arr & bits_values


def get_bits_img(image, interval):
    """Returns only bits specified into the interval

    Args:
        image: image to convert using this interval of bits value to keep
        interval: (begin, end) of bits values

    Returns:
        Numpy array with reduced values

    Raises:
        ValueError: If min value from interval is not >= 1.
        ValueError: If max value from interval is not <= 8.
        ValueError: If min value from interval >= max value.

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> bits_img = metrics.get_bits_img(img, (2, 5))
    >>> bits_img.shape
    (200, 200, 3)
    """

    img_arr = np.array(image)
    begin, end = interval

    if begin < 1:
        raise ValueError(
            "Unexpected value of interval. Interval min value needs to be >= 1."
        )

    if end > 8:
        raise ValueError(
            "Unexpected value of interval. Interval min value needs to be <= 8."
        )

    if begin >= end:
        raise ValueError("Unexpected interval values order.")

    bits_values = sum([pow(2, i - 1) for i in range(begin, end + 1)])

    return img_arr & bits_values


def gray_to_mscn(image):
    """Convert Grayscale Image into Mean Subtracted Contrast Normalized (MSCN)

    Args:
        image: grayscale image

    Returns:
        MSCN matrix obtained from transformation

    Usage:

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> img = metrics.get_LAB_L(img)
    >>> img_mscn = metrics.gray_to_mscn(img)
    >>> img_mscn.shape
    (200, 200)
    """

    s = 7 / 6
    blurred = cv2.GaussianBlur(image, (7, 7),
                               s)  # apply gaussian blur to the image
    blurred_sq = blurred * blurred
    sigma = cv2.GaussianBlur(image * image, (7, 7), s)
    sigma = abs(sigma - blurred_sq)**0.5
    sigma = sigma + 1.0 / 255  # avoid DivideByZero Exception
    mscn = (image - blurred) / sigma  # MSCN(i, j) image

    return mscn
