import numpy as np
from ipfml import processing

def __global_noise_filter(image, n, random_function, identical=False, distribution_interval=(-0.5, 0.5), k=0.2):
    """
    @brief White noise filter to apply on image
    @param image - image used as input (2D or 3D image representation)
    @param n - used to set importance of noise [1, 999]
    @param random_function - random function we want to use to generate random numpy array
    @param distribution_interval - set the distribution interval of uniform distribution
    @param k - variable that specifies the amount of noise to be taken into account in the output image
    @return Image with specified noise applied

    Usage :

    >>> from ipfml.filters.noise import global_noise_filter
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = global_noise_filter(image, 10, np.random.uniform)
    >>> noisy_image.shape
    (100, 100)
    """

    image_array = np.asarray(image)
    nb_chanel = 1

    if image_array.ndim != 3:
        width, height = image_array.shape
    else:
        width, height, nb_chanel = image_array.shape

    a, b = distribution_interval
    nb_pixels = width * height

    if identical:
        noise_filter = random_function(a, b, nb_pixels)

    # final output numpy array
    output_array = []

    for chanel in range(0, nb_chanel):

        # getting flatten information from image and noise
        if nb_chanel == 3:
            image_array_flatten = image_array[:, :, chanel].reshape(nb_pixels)
        else:
            image_array_flatten = image_array.reshape(nb_pixels)

        # redefine noise if necessary
        if not identical:
            noise_filter = random_function(a, b, nb_pixels)

        # compute new pixel value
        # n * k * white_noise_filter[i]
        noisy_image = np.asarray([image_array_flatten[i] + n * k * noise_filter[i] for i in range(0, nb_pixels)])

        # reshape and normalize new value
        noisy_image = noisy_image.reshape((width, height))

        noisy_image = np.asarray(noisy_image, 'uint8')

        # in order to concatenae output array
        if nb_chanel == 3:
            noisy_image = noisy_image[:, :, np.newaxis]

        # append new chanel
        output_array.append(noisy_image)

    # concatenate RGB image
    if nb_chanel == 3:
        output_array = np.concatenate(output_array, axis=2)

    return np.asarray(output_array)


def white_noise(image, n, identical=False, distribution_interval=(-0.5, 0.5), k=0.2):
    """
    @brief White noise filter to apply on image
    @param image - image used as input (2D or 3D image representation)
    @param n - used to set importance of noise [1, 999]
    @param distribution_interval - set the distribution interval of normal law distribution
    @param k - variable that specifies the amount of noise to be taken into account in the output image
    @return Image with white noise applied

    Usage :

    >>> from ipfml.filters.noise import white_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = white_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    return __global_noise_filter(image, n, np.random.uniform, identical, distribution_interval, k)


def gaussian_noise(image, n, identical=False, distribution_interval=(0, 1), k=0.1):
    """
    @brief Gaussian noise filter to apply on image
    @param image - image used as input (2D or 3D image representation)
    @param n - used to set importance of noise [1, 999]
    @param distribution_interval - set the distribution interval of normal law distribution
    @param k - variable that specifies the amount of noise to be taken into account in the output image
    @return Image with gaussian noise applied

    Usage :

    >>> from ipfml.filters.noise import gaussian_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = gaussian_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    return __global_noise_filter(image, n, np.random.normal, identical, distribution_interval, k)


def laplace_noise(image, n, identical=False, distribution_interval=(0, 1), k=0.1):
    """
    @brief Laplace noise filter to apply on image
    @param image - image used as input (2D or 3D image representation)
    @param n - used to set importance of noise [1, 999]
    @param distribution_interval - set the distribution interval of normal law distribution
    @param k - variable that specifies the amount of noise to be taken into account in the output image
    @return Image with Laplace noise applied

    Usage :

    >>> from ipfml.filters.noise import gaussian_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = gaussian_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    return __global_noise_filter(image, n, np.random.laplace, identical, distribution_interval, k)





