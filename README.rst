IPFML
=====

Image Processing For Machine Learning python package.

This is a package developed during a thesis project.

How to use ?
------------

To use, simply do :

    >>> from PIL import Image
    >>> from ipfml import image_processing
    >>> img = Image.open('path/to/image.png')
    >>> s = image_processing.get_LAB_L_SVD_s(img)


Modules
-------

This project contains modules.

- **processing** : *Image processing of images*
- **metrics** : *Metrics computation of PIL or 2D, 3D numpy images*
- **filters** : *Filters implemented such as noise filters*

All these modules will be enhanced during development of the package.

Documentation
-------------

For more information about package, documentation_ is available. 

.. _documentation: https://jbuisine.github.io/IPFML/  
