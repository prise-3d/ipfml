Image Processing For Machine Learning
=====================================

This is a package developed during a thesis project.

Installation
------------

.. code:: bash

   pip install ipfml

How to use ?
------------

To use, simply do :

.. code:: python

   from PIL import Image
   from ipfml import processing
   img = Image.open('path/to/image.png')
   s = processing.get_LAB_L_SVD_s(img)


Modules
-------

This project contains modules.

- **processing** : *Image processing of images*
- **metrics** : *Metrics computation of PIL or 2D, 3D numpy images*
- **filters** : *Filters implemented such as noise filters*
- **utils** : *All utils functions developed for the package*
- **exceptions** : *All customized exceptions*

All these modules will be enhanced during development of the package.

Documentation
-------------

For more information about package, documentation_ is available.

.. _documentation: https://jbuisine.github.io/IPFML/

Contribution
------------

Please refer to the guidelines_ file if you want to contribute!

.. _guidelines: https://github.com/jbuisine/IPFML/blob/master/CONTRIBUTION.md 
