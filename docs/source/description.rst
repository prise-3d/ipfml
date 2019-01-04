Description
============


Installation
------------

Just install package using pip 

   >>> pip install ipfml


How to use ?
------------

To use, simply do :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('path/to/image.png')
    >>> s = image_processing.get_LAB_L_SVD_s(img)


