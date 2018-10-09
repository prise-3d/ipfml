# module file which contains all image metrics used in project

from numpy.linalg import svd
from scipy import misc

import numpy as np
from sklearn import preprocessing
from skimage import io, color

def get_SVD(image):
    """
    @brief Transforms Image into SVD
    @param image to convert
    @return U, s, V image decomposition
    """
    return svd(image, full_matrices=False)

def get_SVD_s(image):
    """
    @brief Transforms Image into SVD and returns only 's' part
    @param image to convert
    @return s
    """
    U, s, V = svd(image, full_matrices=False)
    return s

def get_SVD_U(image):
    """
    @brief Transforms Image into SVD and returns only 'U' part
    @param image to convert
    @return U
    """
    U, s, V = svd(image, full_matrices=False)
    return U

def get_SVD_V(image):
    """
    @brief Transforms Image into SVD and returns only 'V' part
    @param image to convert
    @return V
    """
    U, s, V = svd(image, full_matrices=False)
    return V

def get_LAB(image):
    """
    @brief Transforms PIL RGB Image into LAB 
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    return color.rgb2lab(rgb)

def get_LAB_L(image):
    """
    @brief Transforms PIL RGB Image into LAB and returns L
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    lab = color.rgb2lab(rgb)
    return lab[:, :, 0]

def get_LAB_A(image):
    """
    @brief Transforms PIL RGB Image into LAB and returns A
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    lab = color.rgb2lab(rgb)
    return lab[:, :, 1]

def get_LAB_B(image):
    """
    @brief Transforms PIL RGB Image into LAB and returns B
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    lab = color.rgb2lab(rgb)
    return lab[:, :, 2]

def get_XYZ(image):
    """
    @brief Transforms PIL RGB Image into XYZ
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    return color.rgb2xyz(rgb)

def get_XYZ_X(image):
    """
    @brief Transforms PIL RGB Image into XYZ and returns X
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    xyz = color.rgb2xyz(rgb)
    return xyz[:, :, 0]

def get_XYZ_Y(image):
    """
    @brief Transforms PIL RGB Image into XYZ and returns Y
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    xyz = color.rgb2xyz(rgb)
    return xyz[:, :, 1]

def get_XYZ_Z(image):
    """
    @brief Transforms PIL RGB Image into XYZ and returns Z
    @param image to convert
    @return Lab information
    """
    rgb = io.imread(image.filename)
    xyz = color.rgb2xyz(rgb)
    return xyz[:, :, 2]
