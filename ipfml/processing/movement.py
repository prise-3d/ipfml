"""
All movements that can be applied on image such as rotations, fusions, flips
"""

# main imports
import numpy as np

# image processing imports
from PIL import Image
from skimage import transform as sk_transform

# ipfml imports
from ipfml.exceptions import NumpyShapeComparisonException


def fusion_images(images, pil=True):
    '''Fusion array of images into single image

    Args:
        images: array of images (PIL Image or Numpy array)
        pil: block type returned as PIL Image (default True)

    Returns:
        merged image from array of images

    Raises:
        ValueError: if `images` is not an array or is empty
        NumpyShapeComparisonException: if `images` array contains images with different shapes

    Example:

    >>> import numpy as np
    >>> from ipfml.processing import movement
    >>> image_values_1 = np.random.randint(255, size=(800, 800, 3))
    >>> image_values_2 = np.random.randint(255, size=(800, 800, 3))
    >>> merged_image = movement.fusion_images([image_values_1, image_values_2], pil=False)
    >>> merged_image.shape
    (800, 800, 3)
    '''

    mode = 'RGB'
    dim = 1

    if len(images) == 0:
        raise ValueError('Empty array of images provided...')

    # convert image in numpy array (perhaps not necessary)
    images = [np.asarray(img) for img in images]
    image_array = images[0]

    if image_array.ndim != 3:
        mode = 'L'
        width, height = image_array.shape
    else:
        width, height, dim = image_array.shape

    # raise exception if all images do not have same shape
    if not np.array([image_array.shape == a.shape for a in images]).all():
        raise NumpyShapeComparisonException()

    if dim == 1:
        image_mean = np.empty([width, height])
    else:
        image_mean = np.empty([width, height, dim])

    nb_images = len(images)

    # construction of mean image from rotation
    for i in range(width):
        for j in range(height):

            if dim == 1:
                grey_value = 0

                # for each image we merge pixel values
                for img in images:
                    grey_value += img[i][j]

                image_mean[i][j] = grey_value / nb_images

            else:
                for k in range(dim):
                    canal_value = 0

                    # for each image we merge pixel values
                    for img in images:
                        canal_value += img[i][j][k]

                    image_mean[i][j][k] = canal_value / nb_images

    image_mean = np.array(image_mean, 'uint8')

    if pil:
        return Image.fromarray(image_mean, mode)
    else:
        return image_mean


def rotate_image(image, angle=90, pil=True):
    """Rotate image using specific angle

    Args:
        image: PIL Image or Numpy array
        angle: Angle value of the rotation
        pil: block type returned as PIL Image (default True)

    Returns:
        Image with rotation applied

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml.processing import movement
    >>> image_values = Image.open('./images/test_img.png')
    >>> rotated_image = movement.rotate_image(image_values, 90, pil=False)
    >>> rotated_image.shape
    (200, 200, 3)
    """

    mode = 'RGB'
    image_array = np.asarray(image)

    if image_array.ndim != 3:
        mode = 'L'

    rotated_image = np.array(
        sk_transform.rotate(image_array, angle) * 255, 'uint8')

    if pil:
        return Image.fromarray(rotated_image, mode)
    else:
        return rotated_image
