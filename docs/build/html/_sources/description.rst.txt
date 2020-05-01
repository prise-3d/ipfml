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
   from ipfml.processing import transform
   img = Image.open('path/to/image.png')
   s = transform.get_LAB_L_SVD_s(img)
