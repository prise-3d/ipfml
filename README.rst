IPFML
=====

Image Processing For Machine Learning package.

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

- **img_processing** : *PIL image processing part*
    - fig2data(fig): *Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it*
    - fig2img(fig): *Convert a Matplotlib figure to a PIL Image in RGB format and return it*
    - get_LAB_L_SVD_U(image): *Returns U SVD from L of LAB Image information*
    - get_LAB_L_SVD_s(image): *Returns s (Singular values) SVD from L of LAB Image information*
    - get_LAB_L_SVD_V(image): *Returns V SVD from L of LAB Image information*
    - divide_in_blocks(image, block_size): Divide image into equal size blocks
    - normalize_arr(arr): *Normalize array values*
    - normalize_arr_with_range(arr, min, max): *Normalize array values with specific min and max values*
    - rgb_to_mscn(image): *Convert RGB Image into Mean Subtracted Contrast Normalized (MSCN) using only gray level*

- **metrics** : *Metrics computation of PIL image*
    - get_SVD(image): *Transforms PIL Image into SVD*
    - get_SVD_U(image): *Transforms PIL Image into SVD and returns only 'U' part*
    - get_SVD_s(image): *Transforms PIL Image into SVD and returns only 's' part*
    - get_SVD_V(image): *Transforms PIL Image into SVD and returns only 'V' part*
    - get_LAB(image): *Transforms PIL Image into LAB*
    - get_LAB_L(image): *Transforms PIL Image into LAB and returns only 'L' part*
    - get_LAB_A(image): *Transforms PIL Image into LAB and returns only 'A' part*
    - get_LAB_B(image): *Transforms PIL Image into LAB and returns only 'B' part*
    - get_XYZ(image): *Transforms PIL Image into XYZ*
    - get_XYZ_X(image): *Transforms PIL Image into XYZ and returns only 'X' part*
    - get_XYZ_Y(image): *Transforms PIL Image into XYZ and returns only 'Y' part*
    - get_XYZ_Z(image): *Transforms PIL Image into XYZ and returns only 'Z' part*

- **ts_model_helper** : *contains helpful function to save or display model information and performance of tensorflow model*
    - save(history, filename): *Function which saves data from neural network model*
    - show(history, filename): *Function which shows data from neural network model*

All these modules will be enhanced during development of the project

How to contribute
-----------------

This git project uses git-flow_ implementation. You are free to contribute to it.

.. _git-flow : https://danielkummer.github.io/git-flow-cheatsheet/