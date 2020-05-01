Examples
=====================================

Some examples are already available into documentation. You can find here some others and results of use of `ipfml` package.

All examples below will use this picture.

.. image:: _static/nature.jpg

Processing example
--------------------

.. code:: python
   
   from PIL import Image
   from ipfml.processing import transform

   img_path = 'path/to/image_nature.jpg'
   img = Image.open(img_path)
   
   low_bits_img = transform.rgb_to_grey_low_bits(img, 6)
   
   output = Image.fromarray(low_bits_img)
   output.show()

Now we have picture information with only the 6 low bits values.

.. image:: _static/nature_low_bits_6.png

Noise filter example
---------------------

.. code:: python
   
   from PIL import Image
   from ipfml.filters import noise as nf

   img_path = 'path/to/image_nature.jpg'
   img = Image.open(img_path)

   # set noise impact to 400
   # set same noise for each chanel
   noisy_image = nf.gaussian_noise(img, n=400, identical=True)

   output = Image.fromarray(noisy_image)
   output.show()
   
Image result after applying gaussian noise on nature image.

.. image:: _static/nature_gaussian_noise.png

