from PIL import Image
from matplotlib import cm

import numpy as np
import ipfml.metrics as metrics


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
    
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGB format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library (PIL) image : default size (480,640,3)
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())

def get_LAB_L_SVD(image):
    """
    @brief Returns Singular values from LAB L Image information
    @param fig a matplotlib figure
    @return a Python Imaging Library (PIL) image : default size (480,640,3)

    Usage :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('./images/test_img.png')
    >>> U, s, V = image_processing.get_LAB_L_SVD(img)
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
    @param PIL Image
    @return vector of singular values

    Usage :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('./images/test_img.png')
    >>> s = image_processing.get_LAB_L_SVD_s(img)
    >>> len(s)
    200
    """
    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_s(L)

def get_LAB_L_SVD_U(image):
    """
    @brief Returns U SVD from L of LAB Image information
    @param PIL Image
    @return vector of singular values

    Usage :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('./images/test_img.png')
    >>> U = image_processing.get_LAB_L_SVD_U(img)
    >>> U.shape
    (200, 200)
    """
    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_U(L)

def get_LAB_L_SVD_V(image):
    """
    @brief Returns V SVD from L of LAB Image information
    @param PIL Image
    @return vector of singular values

    Usage :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('./images/test_img.png')
    >>> V = image_processing.get_LAB_L_SVD_V(img)
    >>> V.shape
    (200, 200)
    """

    L = metrics.get_LAB_L(image)
    return metrics.get_SVD_V(L)

def divide_in_blocks(image, block_size):
    '''
    @brief Divide image into equal size blocks
    @param img - PIL Image or numpy array
    @param block - tuple (width, height) representing the size of each dimension of the block
    @return list containing all PIL Image block (in RGB)

    Usage :

    >>> import numpy as np
    >>> from PIL import Image
    >>> from ipfml import image_processing
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

    # check input type (PIL Image or numpy array) and convert it if necessary
    if hasattr(image, 'filename'):
        image_array = np.array(image)
    else:
        image_array = image

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
            blocks.append(Image.fromarray(current_block.astype('uint8'), mode))

    return blocks

    
def normalize_arr(arr):
    '''
    @brief Normalize data of 1D array shape
    @param array - array data of 1D shape

    Usage :

    >>> from ipfml import image_processing
    >>> import numpy as np
    >>> arr = np.arange(11)
    >>> arr_normalized = image_processing.normalize_arr(arr)
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
    @param array - array data of 1D shape

    Usage :

    >>> from ipfml import image_processing
    >>> import numpy as np
    >>> arr = np.arange(11)
    >>> arr_normalized = image_processing.normalize_arr_with_range(arr, 0, 20)
    >>> arr_normalized[1]
    0.05
    '''
    
    output_arr = []

    for v in arr:
        output_arr.append((v - min) / (max - min))
    
    return output_arr
