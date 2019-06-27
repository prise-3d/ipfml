"""
Functions which can be used to extract information from image or reduce it
"""

# main imports
import os
import random
import numpy as np

# image processing imports
from numpy.linalg import svd
from scipy import misc
from sklearn import preprocessing
from skimage import io, color
import cv2
from PIL import Image

# ipfml imports
from ipfml.processing import compression


def get_LAB(image):
    """Transforms RGB  Image into Lab

    Args:
        image: image to convert

    Returns:
        Lab information

    Usage:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> Lab = transform.get_LAB(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> L = transform.get_LAB_L(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> a = transform.get_LAB_a(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> b = transform.get_LAB_b(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> transform.get_XYZ(img).shape
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> x = transform.get_XYZ_X(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> y = transform.get_XYZ_Y(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> z = transform.get_XYZ_Z(img)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_img = transform.get_low_bits_img(img, 5)
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> bits_img = transform.get_bits_img(img, (2, 5))
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
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> img = transform.get_LAB_L(img)
    >>> img_mscn = transform.gray_to_mscn(img)
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


def rgb_to_mscn(image):
    """Convert RGB Image into Mean Subtracted Contrast Normalized (MSCN)

    Args:
        image: 3D RGB image Numpy array or PIL RGB image

    Returns:
        2D Numpy array with MSCN information

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> img_mscn = transform.rgb_to_mscn(img)
    >>> img_mscn.shape
    (200, 200)
    """

    # check if PIL image or not
    img_arr = np.array(image)

    # convert rgb image to gray
    im = np.array(color.rgb2gray(img_arr) * 255, 'uint8')

    return gray_to_mscn(im)


def get_mscn_coefficients(image):
    """Compute the Mean Substracted Constrast Normalized coefficients of an image

    Args:
        image: PIL Image, Numpy array or path of image

    Returns:
        MSCN coefficients

    Raises:
        FileNotFoundError: If `image` is set as str path and image was not found
        ValueError: If `image` numpy shape are not correct

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml.processing import transform
    >>> image_values = Image.open('./images/test_img.png')
    >>> mscn_coefficients = transform.get_mscn_coefficients(image_values)
    >>> mscn_coefficients.shape
    (200, 200)
    """

    if isinstance(image, str):
        if os.path.exists(image):
            # open image directly as grey level image
            imdist = cv2.imread(image, 0)
        else:
            raise FileNotFoundError('Image not found in your system')

    elif isinstance(image, np.ndarray):
        # convert if necessary to grey level numpy array
        if image.ndim == 2:
            imdist = image
        if image.ndim == 3:
            imdist = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('Incorrect image shape')
    else:
        # if PIL Image
        image = np.asarray(image)

        if image.ndim == 2:
            imdist = image
        if image.ndim == 3:
            imdist = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('Incorrect image shape')

    imdist = imdist.astype(np.float64)
    imdist = imdist / 255.0

    # calculating MSCN coefficients
    mu = cv2.GaussianBlur(
        imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(
        imdist * imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    sigma = np.sqrt(abs((sigma - mu_sq)))
    structdis = (imdist - mu) / (sigma + 1)
    return structdis


def get_LAB_L_SVD(image):
    """Returns Singular values from LAB L Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        U, s, V information obtained from SVD compression using Lab

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> U, s, V = transform.get_LAB_L_SVD(img)
    >>> U.shape
    (200, 200)
    >>> len(s)
    200
    >>> V.shape
    (200, 200)
    """
    L = get_LAB_L(image)
    return compression.get_SVD(L)


def get_LAB_L_SVD_s(image):
    """Returns s (Singular values) SVD from L of LAB Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        vector of singular values

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> s = transform.get_LAB_L_SVD_s(img)
    >>> len(s)
    200
    """
    L = get_LAB_L(image)
    return compression.get_SVD_s(L)


def get_LAB_L_SVD_U(image):
    """Returns U SVD from L of LAB Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        U matrix of SVD compression

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> U = transform.get_LAB_L_SVD_U(img)
    >>> U.shape
    (200, 200)
    """
    L = get_LAB_L(image)
    return compression.get_SVD_U(L)


def get_LAB_L_SVD_V(image):
    """Returns V SVD from L of LAB Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        V matrix of SVD compression

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> V = transform.get_LAB_L_SVD_V(img)
    >>> V.shape
    (200, 200)
    """

    L = get_LAB_L(image)
    return compression.get_SVD_V(L)


def rgb_to_grey_low_bits(image, nb_bits=4):
    """Convert RGB Image into grey image using only 4 low bits values

    Args:
        image: 3D RGB image Numpy array or PIL RGB image
        nb_bits: optional parameter which indicates the number of bits to keep (default 4)

    Returns:
        2D Numpy array with low bits information kept

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_grey_img = transform.rgb_to_grey_low_bits(img, 5)
    >>> low_bits_grey_img.shape
    (200, 200)
    """

    img_arr = np.array(image)
    grey_block = np.array(color.rgb2gray(img_arr) * 255, 'uint8')

    return get_low_bits_img(grey_block, nb_bits)


def rgb_to_LAB_L_low_bits(image, nb_bits=4):
    """Convert RGB Image into Lab L channel image using only 4 low bits values

    Args:
        image: 3D RGB image Numpy array or PIL RGB image
        nb_bits: optional parameter which indicates the number of bits to keep (default 4)

    Returns:
        2D Numpy array with low bits information kept

    Example:

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_Lab_l_img = transform.rgb_to_LAB_L_low_bits(img, 5)
    >>> low_bits_Lab_l_img.shape
    (200, 200)
    """

    L_block = np.asarray(get_LAB_L(image), 'uint8')

    return get_low_bits_img(L_block, nb_bits)


def rgb_to_LAB_L_bits(image, interval):
    """Returns only bits from LAB L canal specified into the interval

    Args:
        image: image to convert using this interval of bits value to keep
        interval: (begin, end) of bits values

    Returns:
        2D Numpy array with reduced values

    >>> from PIL import Image
    >>> from ipfml.processing import transform
    >>> img = Image.open('./images/test_img.png')
    >>> bits_Lab_l_img = transform.rgb_to_LAB_L_bits(img, (2, 6))
    >>> bits_Lab_l_img.shape
    (200, 200)
    """

    L_block = np.asarray(get_LAB_L(image), 'uint8')

    return get_bits_img(L_block, interval)
