Description
=====================================

.. image:: _static/ipfml_logo.png
   :width: 400 px
   :align: center

Installation
------------

Just install package using pip 

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
