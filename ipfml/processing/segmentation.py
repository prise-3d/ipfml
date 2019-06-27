"""
All segmentation methods applied on images
"""

# main imports
import numpy as np

# image processing imports
from PIL import Image


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
    >>> from ipfml.processing import transform, segmentation
    >>> image_values = np.random.randint(255, size=(800, 800, 3))
    >>> blocks = segmentation.divide_in_blocks(image_values, (20, 20))
    >>> len(blocks)
    1600
    >>> blocks[0].width
    20
    >>> blocks[0].height
    20
    >>> img_l = Image.open('./images/test_img.png')
    >>> L = transform.get_LAB_L(img_l)
    >>> blocks_L = segmentation.divide_in_blocks(L, (100, 100))
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
