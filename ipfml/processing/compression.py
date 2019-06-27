"""
Functions for image compression and extraction
"""

# image processing imports
from numpy.linalg import svd


def get_SVD(image):
    """Transforms Image using SVD compression

    Args:
        image: image to convert into SVD compression

    Return:
        U, s, V obtained from SVD compression

    Usage:

    >>> from PIL import Image
    >>> from ipfml.processing import compression
    >>> img = Image.open('./images/test_img.png')
    >>> U, s, V = compression.get_SVD(img)
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
    >>> from ipfml.processing import compression
    >>> img = Image.open('./images/test_img.png')
    >>> s = compression.get_SVD_s(img)
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
    >>> from ipfml.processing import compression
    >>> img = Image.open('./images/test_img.png')
    >>> U = compression.get_SVD_U(img)
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
    >>> from ipfml.processing import compression
    >>> img = Image.open('./images/test_img.png')
    >>> V = compression.get_SVD_V(img)
    >>> V.shape
    (200, 3, 3)
    """

    U, s, V = svd(image, full_matrices=False)
    return V
