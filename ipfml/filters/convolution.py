"""
Convolution functions to apply on images
"""

# main imports
import numpy as np


def convolution2D(image, kernel, kernel_size=(5, 5)):
    """Apply 2D convolution on image using specific kernel from `ipfml.filters.kernels`

    Args:
        image: 2D image to apply convolution on
        kernel: specific kernel from `ipfml.filters.kernels` to use
        kernel_size: window size to use (default (5, 5))

    Returns:
        2D numpy array obtained from image using kernel

    Example:

    >>> from ipfml.filters.convolution import convolution2D
    >>> from ipfml.filters.kernels import plane_mean
    >>> import numpy as np
    >>> image = np.arange(81).reshape([9, 9])
    >>> convolved_image = convolution2D(image, plane_mean, (3, 3)) 
    >>> convolved_image.shape
    (7, 7)
    """

    img = np.array(image)

    width, height = img.shape

    kernel_width, kernel_height = kernel_size

    if kernel_width % 2 == 0 or kernel_height % 2 == 0:
        raise ValueError("Invalid kernel size, need to be of odd size")

    padding_height = (kernel_width - 1) / 2
    padding_width = (kernel_width - 1) / 2

    img_diff = []
    for i in range(width):

        if i >= padding_width and i < (width - padding_width):

            row_diff = []

            for j in range(height):

                if j >= padding_height and j < (height - padding_height):

                    # pixel in the center of kernel window size, need to extract window from img
                    window = img[int(i - padding_width):int(i + padding_width +
                                                            1),
                                 int(j - padding_height
                                     ):int(j + padding_height + 1)]

                    diff = kernel(window)
                    row_diff.append(diff)

            img_diff.append(row_diff)

    return np.array(img_diff)
