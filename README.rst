IPFML
=====

Image Processing For Machine Learning package.

How to use ?
------------

To use, simply do::

    >>> from PIL import Image
    >>> import ipfml as iml
    >>> img = Image.open('path/to/image.png')
    >>> s = iml.metrics.get_SVD_s(img)


Modules
-------

This project contains modules.

- **img_processing** : *PIL image processing part*
    - fig2data(fig): *Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it*
    - fig2img(fig): *Convert a Matplotlib figure to a PIL Image in RGB format and return it*

- **metrics** : *Metrics computation of PIL image*
    - get_SVD(image): *Transforms PIL Image into SVD*
    - get_SVD_s(image): *Transforms PIL Image into SVD and returns only 's' part*
    - get_SVD_U(image): *Transforms PIL Image into SVD and returns only 'U' part*
    - get_SVD_V(image): *Transforms PIL Image into SVD and returns only 'V' part*

- **ts_model_helper** : *contains helpful function to save or display model information and performance of tensorflow model*
    - save(history, filename): *Function which saves data from neural network model*
    - show(history, filename): *Function which shows data from neural network model*

All these modules will be enhanced during development of the project

How to contribute
-----------------

This git project uses git-flow_ implementation. You are free to contribute to it.

.. _git-flow : https://danielkummer.github.io/git-flow-cheatsheet/