# module file which contains all image metrics used in project

from numpy.linalg import svd
from scipy import misc

import numpy as np
from sklearn import preprocessing

def get_SVD(image):
    """
    @brief Transforms PIL Image into SVD
    @param image to convert
    @return U, s, V image decomposition
    """
    return svd(image, full_matrices=False)

def get_SVD_s(image):
    """
    @brief Transforms PIL Image into SVD and returns only 's' part
    @param image to convert
    @return s
    """
    U, s, V = svd(image, full_matrices=False)
    return s

def get_SVD_U(image):
    """
    @brief Transforms PIL Image into SVD and returns only 'U' part
    @param image to convert
    @return U
    """
    U, s, V = svd(image, full_matrices=False)
    return U

def get_SVD_V(image):
    """
    @brief Transforms PIL Image into SVD and returns only 'V' part
    @param image to convert
    @return V
    """
    U, s, V = svd(image, full_matrices=False)
    return V
