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
   from ipfml.processing import transform
   img = Image.open('path/to/image.png')
   s = transform.get_LAB_L_SVD_s(img)


Modules
-------

This project contains modules.

- **metrics** : *Metrics computation for model performance*
- **utils** : *All utils functions developed for the package*
- **exceptions** : *All customized exceptions*
- **filters** : *Image filter module*
- **iqa** : *Image quality assessments*
- **processing** : *Image processing module*

All these modules will be enhanced during development of the package.

Documentation
-------------

For more information about package, documentation_ is available.

.. _documentation: https://prise-3d.github.io/ipfml/

Contributing
------------

Please refer to the guidelines_ file if you want to contribute!

.. _guidelines: https://github.com/prise-3d/ipfml/blob/master/CONTRIBUTING.md 
