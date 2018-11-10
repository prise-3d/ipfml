IPFML
=====

Image Processing For Machine Learning package.

How to use ?
------------

To use, simply do :

```python
from PIL import Image
from ipfml import image_processing
img = Image.open('path/to/image.png')
s = image_processing.get_LAB_L_SVD_s(img)
```

Modules
-------

This project contains modules.

- **image_processing** : *Image processing module*
    - get_LAB_L_SVD_U(image): *Returns U SVD from L of LAB Image information*
    - get_LAB_L_SVD_s(image): *Returns s (Singular values) SVD from L of LAB Image information*
    - get_LAB_L_SVD_V(image): *Returns V SVD from L of LAB Image information*
    - divide_in_blocks(image, block_size): *Divide image into equal size blocks*
    - rgb_to_mscn(image): *Convert RGB Image into Mean Subtracted Contrast Normalized (MSCN) using only gray level*
    - rgb_to_grey_low_bits(image, bind=15): *Convert RGB Image into grey image using only 4 low bits values by default*
    - rgb_to_LAB_L_low_bits(image, bind=15): *Convert RGB Image into LAB L channel image using only 4 low bits values by default*
    - normalize_arr(arr): *Normalize array values*
    - normalize_arr_with_range(arr, min, max): *Normalize array values with specific min and max values*
    - normalize_2D_arr(arr): *Return 2D array normalize from its min and max values*

- **metrics** : *Metrics computation of PIL or 2D numpy image*
    - get_SVD(image): *Transforms Image into SVD*
    - get_SVD_U(image): *Transforms Image into SVD and returns only 'U' part*
    - get_SVD_s(image): *Transforms Image into SVD and returns only 's' part*
    - get_SVD_V(image): *Transforms Image into SVD and returns only 'V' part*
    - get_LAB(image): *Transforms Image into LAB*
    - get_LAB_L(image): *Transforms Image into LAB and returns only 'L' part*
    - get_LAB_A(image): *Transforms Image into LAB and returns only 'A' part*
    - get_LAB_B(image): *Transforms Image into LAB and returns only 'B' part*
    - get_XYZ(image): *Transforms Image into XYZ*
    - get_XYZ_X(image): *Transforms Image into XYZ and returns only 'X' part*
    - get_XYZ_Y(image): *Transforms Image into XYZ and returns only 'Y' part*
    - get_XYZ_Z(image): *Transforms Image into XYZ and returns only 'Z' part*
    - get_low_bits_img(image, bind=15): *Returns Image or Numpy array with data information reduced using only low bits (by default*

All these modules will be enhanced during development of the project

How to contribute
-----------------

This git project uses [git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) implementation. You are free to contribute to it.
