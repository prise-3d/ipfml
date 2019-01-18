"""
Module which contains all customs Exceptions used into ipfml package
"""


class NumpyDimensionComparisonException(Exception):
    """
    Numpy dimensions comparison Exception raised if two numpy arrays provided do not have same dimensions
    """

    def __init__(self):
        Exception.__init__(
            self,
            "Numpy arrays provided for comparisons do not have same dimensions"
        )


class NumpyShapeComparisonException(Exception):
    """
    Numpy shape comparison Exception raised if two numpy arrays provided do not have same shape extactly
    """

    def __init__(self):
        Exception.__init__(
            self,
            "Numpy arrays provided for comparisons do not have same shape")
