# module file which contains all image metrics used in project

from numpy.linalg import svd
from scipy import misc

import numpy as np
from sklearn import preprocessing
from skimage import io, color

import cv2

def get_image_path(image):
    """
    @brief Returns file path of PIL Image
    @param PIL Image
    @return image path

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> path = metrics.get_image_path(img)
    >>> 'images/test_img.png' in path
    True
    """

    if hasattr(image, 'filename'):
        file_path = image.filename
    else:
        raise Exception("Image provided is not correct, required filename property...")

    return file_path

def get_SVD(image):
    """
    @brief Transforms Image into SVD
    @param image to convert
    @return U, s, V image decomposition

    Usage :

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
    """
    @brief Transforms Image into SVD and returns only 's' part
    @param image to convert
    @return s

    Usage :

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
    """
    @brief Transforms Image into SVD and returns only 'U' part
    @param image to convert
    @return U

    Usage :

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
    """
    @brief Transforms Image into SVD and returns only 'V' part
    @param image to convert
    @return V

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
    """
    @brief Transforms PIL RGB Image into LAB
    @param image to convert
    @return Lab information

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> Lab = metrics.get_LAB(img)
    >>> Lab.shape
    (200, 200, 3)
    """

    return color.rgb2lab(image)

def get_LAB_L(image):
    """
    @brief Transforms PIL RGB Image into LAB and returns L
    @param image to convert
    @return Lab information

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> L = metrics.get_LAB_L(img)
    >>> L.shape
    (200, 200)
    """

    lab = get_LAB(image)
    return lab[:, :, 0]

def get_LAB_A(image):
    """
    @brief Transforms PIL RGB Image into LAB and returns A
    @param image to convert
    @return Lab information

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> A = metrics.get_LAB_A(img)
    >>> A.shape
    (200, 200)
    """

    lab = get_LAB(image)
    return lab[:, :, 1]

def get_LAB_B(image):
    """
    @brief Transforms PIL RGB Image into LAB and returns B
    @param image to convert
    @return Lab information

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> B = metrics.get_LAB_B(img)
    >>> B.shape
    (200, 200)
    """

    lab = get_LAB(image)
    return lab[:, :, 2]

def get_XYZ(image):
    """
    @brief Transforms PIL RGB Image into XYZ
    @param image to convert
    @return Lab information

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> metrics.get_XYZ(img).shape
    (200, 200, 3)
    """

    return color.rgb2xyz(image)

def get_XYZ_X(image):
    """
    @brief Transforms PIL RGB Image into XYZ and returns X
    @param image to convert
    @return Lab information

    Usage :

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
    """
    @brief Transforms PIL RGB Image into XYZ and returns Y
    @param image to convert
    @return Lab information

    Usage :

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
    """
    @brief Transforms PIL RGB Image into XYZ and returns Z
    @param image to convert
    @return Lab information

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> z = metrics.get_XYZ_Z(img)
    >>> z.shape
    (200, 200)
    """

    xyz = color.rgb2xyz(image)
    return xyz[:, :, 2]

def get_low_bits_img(image, bind=15):
    """
    @brief Returns Image or Numpy array with data information reduced using only low bits
    @param image to convert
    @bind optional : bits to keep using & Bitwise operator
    @return Numpy array with reduced values

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_img = metrics.get_low_bits_img(img)
    >>> low_bits_img.shape
    (200, 200, 3)
    """

    img_arr = np.array(image)

    return img_arr & bind

def get_bits_img(image, interval):
    """
    @brief Returns only bits specified into the interval
    @param image to convert using this interval of bits value to keep
    @param interval (begin, end) of bits values
    @return Numpy array with reduced values

    Usage :

    >>> from PIL import Image
    >>> from ipfml import metrics
    >>> img = Image.open('./images/test_img.png')
    >>> bits_img = metrics.get_bits_img(img, (2, 5))
    >>> bits_img.shape
    (200, 200, 3)
    """

    img_arr = np.array(image)
    begin, end = interval

    bits_values = sum([pow(2, i - 1) for i in range(begin, end + 1)])

    return img_arr & bits_values


def gray_to_mscn(image):
    """
    @brief Convert Grayscale Image into Mean Subtracted Contrast Normalized (MSCN)
    @param grayscale image numpy array or PIL RGB image

    Usage :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('./images/test_img.png')
    >>> img_mscn = image_processing.rgb_to_mscn(img)
    >>> img_mscn.shape
    (200, 200)
    """

    s = 7/6
    blurred = cv2.GaussianBlur(image, (7, 7), s) # apply gaussian blur to the image
    blurred_sq = blurred * blurred
    sigma = cv2.GaussianBlur(image * image, (7, 7), s)
    sigma = abs(sigma - blurred_sq) ** 0.5
    sigma = sigma + 1.0/255 # avoid DivideByZero Exception
    mscn = (image - blurred)/sigma # MSCN(i, j) image

    return mscn
