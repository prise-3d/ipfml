from PIL import Image
from matplotlib import cm
import random

from skimage import color
import numpy as np
import ipfml.metrics as metrics
import cv2

from scipy import signal

def get_LAB_L_SVD(image):
    """
    @brief Returns Singular values from LAB L Image information
    @param image - PIL Image or Numpy array
    @return U, s, V information obtained from SVD compression using Lab

    Usage :

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
    """
    @brief Returns s (Singular values) SVD from L of LAB Image information
    @param image - PIL Image or numpy array
    @return vector of singular values

    Usage :

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
    """
    @brief Returns U SVD from L of LAB Image information
    @param image - PIL Image
    @return U matrix of SVD compression

    Usage :

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
    """
    @brief Returns V SVD from L of LAB Image information
    @param image - PIL Image or numpy array
    @return V matrix of SVD compression

    Usage :

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> V = processing.get_LAB_L_SVD_V(img)
    >>> V.shape
    (200, 200)
    """

    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_V(L)


def divide_in_blocks(image, block_size, pil=True):
    '''
    @brief Divide image into equal size blocks
    @param image - PIL Image or numpy array
    @param block - tuple (width, height) representing the size of each dimension of the block
    @param pil - kind block type (PIL by default or Numpy array)
    @return list containing all 2D numpy blocks (in RGB or not)

    Usage :

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

    # convert in numpy array
    image_array = np.array(image)

    # check dimension of input image
    if image_array.ndim != 3:
        mode = 'L'
        image_width, image_height = image_array.shape
    else:
        image_width, image_height, _ = image_array.shape

    # check size compatibility
    width, height = block_size

    if(image_width % width != 0):
        raise "Width size issue, block size not compatible"

    if(image_height % height != 0):
        raise "Height size issue, block size not compatible"

    nb_block_width = image_width / width
    nb_block_height = image_height / height

    for i in range(int(nb_block_width)):

        begin_x = i * width

        for j in range(int(nb_block_height)):

            begin_y = j * height

            # getting sub block information
            current_block = image_array[begin_x:(begin_x + width), begin_y:(begin_y + height)]

            if pil:
                blocks.append(Image.fromarray(current_block.astype('uint8'), mode))
            else:
                blocks.append(current_block)

    return blocks


def normalize_arr(arr):
    '''
    @brief Normalize data of 1D array shape
    @param arr - array data of 1D shape
    @return Normalized 1D array

    Usage :

    >>> from ipfml import processing
    >>> import numpy as np
    >>> arr = np.arange(11)
    >>> arr_normalized = processing.normalize_arr(arr)
    >>> arr_normalized[1]
    0.1
    '''

    output_arr = []
    max_value = max(arr)
    min_value = min(arr)

    for v in arr:
         output_arr.append((v - min_value) / (max_value - min_value))

    return output_arr


def normalize_arr_with_range(arr, min, max):
    '''
    @brief Normalize data of 1D array shape
    @param arr - array data of 1D shape

    Usage :

    >>> from ipfml import processing
    >>> import numpy as np
    >>> arr = np.arange(11)
    >>> arr_normalized = processing.normalize_arr_with_range(arr, 0, 20)
    >>> arr_normalized[1]
    0.05
    '''

    output_arr = []

    for v in arr:
        output_arr.append((v - min) / (max - min))

    return output_arr


def normalize_2D_arr(arr):
    """
    @brief Return array normalize from its min and max values
    @param arr - 2D numpy array
    @return normalized 2D array

    Usage :

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> img_mscn = processing.rgb_to_mscn(img)
    >>> img_normalized = processing.normalize_2D_arr(img_mscn)
    >>> img_normalized.shape
    (200, 200)
    """

    # getting min and max value from 2D array
    max_value = arr.max(axis=1).max()
    min_value = arr.min(axis=1).min()

    # lambda computation to normalize
    g = lambda x : (x - min_value) / (max_value - min_value)
    f = np.vectorize(g)

    return f(arr)


def rgb_to_mscn(image):
    """
    @brief Convert RGB Image into Mean Subtracted Contrast Normalized (MSCN)
    @param 3D RGB image numpy array or PIL RGB image

    Usage :

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
    im = np.array(color.rgb2gray(img_arr)*255, 'uint8')

    return metrics.gray_to_mscn(im)


def rgb_to_grey_low_bits(image, nb_bits=4):
    """
    @brief Convert RGB Image into grey image using only 4 low bits values
    @param 3D RGB image numpy array or PIL RGB image
    @param nb_bits, optional parameter which indicates the number of bits to keep

    Usage :

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> low_bits_grey_img = processing.rgb_to_grey_low_bits(img, 5)
    >>> low_bits_grey_img.shape
    (200, 200)
    """

    img_arr = np.array(image)
    grey_block = np.array(color.rgb2gray(img_arr)*255, 'uint8')

    return metrics.get_low_bits_img(grey_block, nb_bits)


def rgb_to_LAB_L_low_bits(image, nb_bits=4):
    """
    @brief Convert RGB Image into Lab L channel image using only 4 low bits values
    @param 3D RGB image numpy array or PIL RGB image
    @param nb_bits, optional parameter which indicates the number of bits to keep

    Usage :

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
    """
    @brief Returns only bits from LAB L canal specified into the interval
    @param image to convert using this interval of bits value to keep
    @param interval (begin, end) of bits values
    @return Numpy array with reduced values

    >>> from PIL import Image
    >>> from ipfml import processing
    >>> img = Image.open('./images/test_img.png')
    >>> bits_Lab_l_img = processing.rgb_to_LAB_L_bits(img, (2, 6))
    >>> bits_Lab_l_img.shape
    (200, 200)
    """

    L_block = np.asarray(metrics.get_LAB_L(image), 'uint8')

    return metrics.get_bits_img(L_block, interval)