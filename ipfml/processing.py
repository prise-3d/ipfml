"""
Functions to quickly extract reduced information from image
"""

from PIL import Image
import random

import cv2
from skimage import transform, color
from scipy import signal
from numpy.linalg import svd
from sklearn.decomposition import FastICA, IncrementalPCA

import numpy as np
import ipfml.metrics as metrics
from ipfml.exceptions import NumpyShapeComparisonException

import os


def get_LAB_L_SVD(image):
    """Returns Singular values from LAB L Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        U, s, V information obtained from SVD compression using Lab

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> U, s, V = processing.get_LAB_L_SVD(img)
    >>> U.shape
    (200, 200)
    >>> len(s)
    200
    >>> V.shape
    (200, 200)
    """
    L = metrics.get_LAB_L(image)
    return metrics.get_SVD(L)


def get_LAB_L_SVD_s(image):
    """Returns s (Singular values) SVD from L of LAB Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        vector of singular values

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> s = processing.get_LAB_L_SVD_s(img)
    >>> len(s)
    200
    """
    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_s(L)


def get_LAB_L_SVD_U(image):
    """Returns U SVD from L of LAB Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        U matrix of SVD compression

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> U = processing.get_LAB_L_SVD_U(img)
    >>> U.shape
    (200, 200)
    """
    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_U(L)


def get_LAB_L_SVD_V(image):
    """Returns V SVD from L of LAB Image information

    Args:
        image: PIL Image or Numpy array

    Returns:
        V matrix of SVD compression

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> V = processing.get_LAB_L_SVD_V(img)
    >>> V.shape
    (200, 200)
    """

    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_V(L)


def rgb_to_mscn(image):
    """Convert RGB Image into Mean Subtracted Contrast Normalized (MSCN)

    Args:
        image: 3D RGB image Numpy array or PIL RGB image

    Returns:
        2D Numpy array with MSCN information

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> img_mscn = processing.rgb_to_mscn(img)
    >>> img_mscn.shape
    (200, 200)
    """

    # check if PIL image or not
    img_arr = np.array(image)

    # convert rgb image to gray
    im = np.array(color.rgb2gray(img_arr) * 255, 'uint8')

    return metrics.gray_to_mscn(im)


def rgb_to_grey_low_bits(image, nb_bits=4):
    """Convert RGB Image into grey image using only 4 low bits values

    Args:
        image: 3D RGB image Numpy array or PIL RGB image
        nb_bits: optional parameter which indicates the number of bits to keep (default 4)

    Returns:
        2D Numpy array with low bits information kept

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_grey_img = processing.rgb_to_grey_low_bits(img, 5)
    >>> low_bits_grey_img.shape
    (200, 200)
    """

    img_arr = np.array(image)
    grey_block = np.array(color.rgb2gray(img_arr) * 255, 'uint8')

    return metrics.get_low_bits_img(grey_block, nb_bits)


def rgb_to_LAB_L_low_bits(image, nb_bits=4):
    """Convert RGB Image into Lab L channel image using only 4 low bits values

    Args:
        image: 3D RGB image Numpy array or PIL RGB image
        nb_bits: optional parameter which indicates the number of bits to keep (default 4)

    Returns:
        2D Numpy array with low bits information kept

    Example:

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_Lab_l_img = processing.rgb_to_LAB_L_low_bits(img, 5)
    >>> low_bits_Lab_l_img.shape
    (200, 200)
    """

    L_block = np.asarray(metrics.get_LAB_L(image), 'uint8')

    return metrics.get_low_bits_img(L_block, nb_bits)


def rgb_to_LAB_L_bits(image, interval):
    """Returns only bits from LAB L canal specified into the interval

    Args:
        image: image to convert using this interval of bits value to keep
        interval: (begin, end) of bits values

    Returns:
        2D Numpy array with reduced values

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> bits_Lab_l_img = processing.rgb_to_LAB_L_bits(img, (2, 6))
    >>> bits_Lab_l_img.shape
    (200, 200)
    """

    L_block = np.asarray(metrics.get_LAB_L(image), 'uint8')

    return metrics.get_bits_img(L_block, interval)


def divide_in_blocks(image, block_size, pil=True):
    '''Divide image into equal size blocks

    Args:
        image: PIL Image or Numpy array
        block: tuple (width, height) representing the size of each dimension of the block
        pil: block type returned as PIL Image (default True)

    Returns:
        list containing all 2D Numpy blocks (in RGB or not)

    Raises:
        ValueError: If `image_width` or `image_height` are not compatible to produce correct block sizes

    Example:

    >>> import numpy as np
    >>> from PIL import Image
    >>> from ipfml import processing
    >>> from ipfml import metrics
    >>> image_values = np.random.randint(255, size=(800, 800, 3))
    >>> blocks = divide_in_blocks(image_values, (20, 20))
    >>> len(blocks)
    1600
    >>> blocks[0].width
    20
    >>> blocks[0].height
    20
    >>> img_l = Image.open('./images/test_img.png')
    >>> L = metrics.get_LAB_L(img_l)
    >>> blocks_L = divide_in_blocks(L, (100, 100))
    >>> len(blocks_L)
    4
    >>> blocks_L[0].width
    100
    '''

    blocks = []
    mode = 'RGB'

    # convert in Numpy array
    image_array = np.array(image)

    # check dimension of input image
    if image_array.ndim != 3:
        mode = 'L'
        image_width, image_height = image_array.shape
    else:
        image_width, image_height, _ = image_array.shape

    # check size compatibility
    width, height = block_size

    if (image_width % width != 0):
        raise ValueError("Width size issue, block size not compatible")

    if (image_height % height != 0):
        raise ValueError("Height size issue, block size not compatible")

    nb_block_width = image_width / width
    nb_block_height = image_height / height

    for i in range(int(nb_block_width)):

        begin_x = i * width

        for j in range(int(nb_block_height)):

            begin_y = j * height

            # getting sub block information
            current_block = image_array[begin_x:(begin_x + width), begin_y:(
                begin_y + height)]

            if pil:
                blocks.append(
                    Image.fromarray(current_block.astype('uint8'), mode))
            else:
                blocks.append(current_block)

    return blocks


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
    >>> from ipfml import processing
    >>> image_values_1 = np.random.randint(255, size=(800, 800, 3))
    >>> image_values_2 = np.random.randint(255, size=(800, 800, 3))
    >>> merged_image = processing.fusion_images([image_values_1, image_values_2], pil=False)
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
    >>> from ipfml import processing
    >>> image_values = Image.open('./images/test_img.png')
    >>> rotated_image = processing.rotate_image(image_values, 90, pil=False)
    >>> rotated_image.shape
    (200, 200, 3)
    """

    mode = 'RGB'
    image_array = np.asarray(image)

    if image_array.ndim != 3:
        mode = 'L'

    rotated_image = np.array(
        transform.rotate(image_array, angle) * 255, 'uint8')

    if pil:
        return Image.fromarray(rotated_image, mode)
    else:
        return rotated_image


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
    >>> from ipfml import processing
    >>> image_values = Image.open('./images/test_img.png')
    >>> mscn_coefficients = processing.get_mscn_coefficients(image_values)
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


def svd_reconstruction(image, interval):
    """Reconstruct an image from SVD compression using specific interval of Singular Values

    Args:
        image: PIL Image, Numpy array or path of 3D image
        interval: Interval used for reconstruction

    Returns:
        Reconstructed image

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml import processing
    >>> image_values = Image.open('./images/test_img.png')
    >>> reconstructed_image = processing.svd_reconstruction(image_values, (100, 200))
    >>> reconstructed_image.shape
    (200, 200)
    """

    begin, end = interval
    lab_img = metrics.get_LAB_L(image)
    lab_img = np.array(lab_img, 'uint8')

    U, s, V = svd(lab_img, full_matrices=True)

    # reconstruction using specific interval
    smat = np.zeros((end - begin, end - begin), dtype=complex)
    smat[:, :] = np.diag(s[begin:end])
    output_img = np.dot(U[:, begin:end], np.dot(smat, V[begin:end, :]))

    return output_img


def fast_ica_reconstruction(image, components):
    """Reconstruct an image from SVD compression using specific number of components to use

    Args:
        image: PIL Image, Numpy array or path of 3D image
        components: Number of components used for reconstruction

    Returns:
        Reconstructed image

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml import processing
    >>> image_values = Image.open('./images/test_img.png')
    >>> reconstructed_image = processing.fast_ica_reconstruction(image_values, 25)
    >>> reconstructed_image.shape
    (200, 200)
    """

    lab_img = metrics.get_LAB_L(image)
    lab_img = np.array(lab_img, 'uint8')

    ica = FastICA(n_components=50)
    # run ICA on image
    ica.fit(lab_img)
    # reconstruct image with independent components
    image_ica = ica.fit_transform(lab_img)
    restored_image = ica.inverse_transform(image_ica)

    return restored_image


def ipca_reconstruction(image, components, _batch_size=25):
    """Reconstruct an image from SVD compression using specific number of components to use and batch size

    Args:
        image: PIL Image, Numpy array or path of 3D image
        components: Number of components used for reconstruction
        batch_size: Batch size used for learn (default 25)

    Returns:
        Reconstructed image

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml import processing
    >>> image_values = Image.open('./images/test_img.png')
    >>> reconstructed_image = processing.ipca_reconstruction(image_values, 20)
    >>> reconstructed_image.shape
    (200, 200)
    """
    lab_img = metrics.get_LAB_L(image)
    lab_img = np.array(lab_img, 'uint8')

    transformer = IncrementalPCA(
        n_components=components, batch_size=_batch_size)

    transformed_image = transformer.fit_transform(lab_img)
    restored_image = transformer.inverse_transform(transformed_image)

    return restored_image
