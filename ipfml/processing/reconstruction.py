"""
Functions for reconstruction process of image using reduction/compression methods
"""

# main imports
import numpy as np

# image processing imports
from numpy.linalg import svd as np_svd
from sklearn.decomposition import FastICA, IncrementalPCA

# ipfml imports
from ipfml.processing import transform


def svd(image, interval):
    """Reconstruct an image from SVD compression using specific interval of Singular Values

    Args:
        image: PIL Image, Numpy array or path of 3D image
        interval: Interval used for reconstruction

    Returns:
        Reconstructed image

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml.processing import reconstruction
    >>> image_values = Image.open('./images/test_img.png')
    >>> reconstructed_image = reconstruction.svd(image_values, (100, 200))
    >>> reconstructed_image.shape
    (200, 200)
    """

    begin, end = interval
    lab_img = transform.get_LAB_L(image)
    lab_img = np.array(lab_img, 'uint8')

    U, s, V = np_svd(lab_img, full_matrices=True)

    # reconstruction using specific interval
    smat = np.zeros((end - begin, end - begin), dtype=complex)
    smat[:, :] = np.diag(s[begin:end])
    output_img = np.dot(U[:, begin:end], np.dot(smat, V[begin:end, :]))

    return output_img


def fast_ica(image, components):
    """Reconstruct an image from Fast ICA compression using specific number of components to use

    Args:
        image: PIL Image, Numpy array or path of 3D image
        components: Number of components used for reconstruction

    Returns:
        Reconstructed image

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml.processing import reconstruction
    >>> image_values = Image.open('./images/test_img.png')
    >>> reconstructed_image = reconstruction.fast_ica(image_values, 25)
    >>> reconstructed_image.shape
    (200, 200)
    """

    lab_img = transform.get_LAB_L(image)
    lab_img = np.array(lab_img, 'uint8')

    ica = FastICA(n_components=50)
    # run ICA on image
    ica.fit(lab_img)
    # reconstruct image with independent components
    image_ica = ica.fit_transform(lab_img)
    restored_image = ica.inverse_transform(image_ica)

    return restored_image


def ipca(image, components, _batch_size=25):
    """Reconstruct an image from IPCA compression using specific number of components to use and batch size

    Args:
        image: PIL Image, Numpy array or path of 3D image
        components: Number of components used for reconstruction
        batch_size: Batch size used for learn (default 25)

    Returns:
        Reconstructed image

    Example:

    >>> from PIL import Image
    >>> import numpy as np
    >>> from ipfml.processing import reconstruction
    >>> image_values = Image.open('./images/test_img.png')
    >>> reconstructed_image = reconstruction.ipca(image_values, 20)
    >>> reconstructed_image.shape
    (200, 200)
    """
    lab_img = transform.get_LAB_L(image)
    lab_img = np.array(lab_img, 'uint8')

    transformer = IncrementalPCA(
        n_components=components, batch_size=_batch_size)

    transformed_image = transformer.fit_transform(lab_img)
    restored_image = transformer.inverse_transform(transformed_image)

    return restored_image
