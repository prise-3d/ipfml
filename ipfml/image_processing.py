from PIL import Image
from matplotlib import cm
import random

from skimage import color
import numpy as np
import ipfml.metrics as metrics
import cv2

from scipy import signal

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

def divide_in_blocks(image, block_size, pil=True):
    '''
    @brief Divide image into equal size blocks
    @param img - PIL Image or numpy array
    @param block - tuple (width, height) representing the size of each dimension of the block
    @param pil - kind block type (PIL by default or Numpy array)
    @return list containing all 2D numpy blocks (in RGB or not)

    Usage :

    >>> import numpy as np
    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> from ipfml import metrics
    >>> image_values = np.random.randint(255, size=(800, 800, 3))
    >>> blocks = divide_img_in_blocks(image_values, (20, 20))
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

            if pil:
                blocks.append(Image.fromarray(current_block.astype('uint8'), mode))
            else:
                blocks.append(current_block)

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


# TODO : add test to this method
def rgb_to_mscn(image):
    """
    @brief Convert RGB Image into Mean Subtracted Contrast Normalized (MSCN)
    @param 3D RGB image numpy array or PIL RGB image 
    """

    # check if PIL image or not
    img_arr = np.array(image)

    # convert rgb image to gray
    im = np.array(color.rgb2gray(img_arr)*255, 'uint8')

    s = 7/6
    blurred = cv2.GaussianBlur(im, (7, 7), s) # apply gaussian blur to the image
    blurred_sq = blurred * blurred 
    sigma = cv2.GaussianBlur(im * im, (7, 7), s)  # switch to -3, 3 (7, 7) before..
    sigma = abs(sigma - blurred_sq) ** 0.5
    sigma = sigma + 1.0/255 # to make sure the denominator doesn't give DivideByZero Exception
    structdis = (im - blurred)/sigma # final MSCN(i, j) image

    return structdis

# TODO : Check this method too...
def get_random_active_block(blocks, threshold = 0.1):
    """
    @brief Find an active block from blocks and return it (randomly way)
    @param 2D numpy array
    @param threshold 0.1 by default
    """

    active_blocks = []

    for id, block in enumerate(blocks):

        arr = np.asarray(block)
        variance = np.var(arr.flatten())

        if variance >= threshold:
            active_blocks.append(id)

    r_id = random.choice(active_blocks)

    return np.asarray(blocks[r_id])


# TODO : check this method and check how to use active block
def segment_relation_in_block(block, active_block):   
    """
    @brief Return bêta value to quantity relation between central segment and surrouding regions into block
    @param 2D numpy array
    """

    if block.ndim != 2:
        raise "Numpy array dimension is incorrect, expected 2."


    # getting middle information of numpy array
    x, y = block.shape

    if y < 4:
        raise "Block size too small needed at least (x, 4) shape"

    middle = int(y / 2)

    # get central segments
    central_segments = block[:, middle-1:middle+1]

    # getting surrouding parts
    left_part = block[:, 0:middle-1]
    right_part = block[:, middle+1:]
    surrounding_parts = np.concatenate([left_part, right_part])

    std_sur = np.std(surrounding_parts.flatten())
    std_cen = np.std(central_segments.flatten())
    std_block = np.std(block.flatten())

    print("CEN " + str(std_cen))
    print("SUR " + str(std_sur))
    print("BLOCK " + str(std_block))

    std_q = std_cen / std_sur

    # from article, it says that block if affected with noise if (std_block > 2 * beta)
    beta = abs(std_q - std_block) / max(std_q, std_block)

    return beta

def normalize_2D_arr(arr):
    """
    @brief Return array normalize from its min and max values
    @param 2D numpy array
    """

    # getting min and max value from 2D array
    max_value = arr.max(axis=1).max()
    min_value = arr.min(axis=1).min()
    
    # lambda computation to normalize
    g = lambda x : (x - min_value) / (max_value - min_value)
    f = np.vectorize(g)
    
    return f(arr)


### other way to compute MSCN :
# TODO : Temp code, check to remove or use it

def normalize_kernel(kernel):
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
    return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')

def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))

def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):

    # check if PIL image or not
    img_arr = np.array(image)

    #im = np.array(color.rgb2gray(img_arr)*255, 'uint8')
    #im = np.asarray(cv2.imread(image.filename, 0)) # read as gray scale
    print(img_arr.shape)

    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(img_arr, kernel, 'same')
    local_var = local_deviation(img_arr, local_mean, kernel)
    
    return (img_arr - local_mean) / (local_var + C)

