"""
Noise filters to apply on images
"""

import numpy as np
import random

from ipfml import processing


def _normalise(x):

    if isinstance(x, np.ndarray):
        return np.array(list(map(_normalise, x)))

    if x > 255:
        return 255
    if x < 0:
        return 0

    return x


def _global_noise_filter(image, generator, updator, identical=False):
    """White noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        generator: lambda function used to generate random numpy array with specific distribution
        updator: lambda function used to update pixel value
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)

    Returns:
        2D Numpy array with specified noise applied

    Example:

    >>> from ipfml.filters.noise import _global_noise_filter as gf
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> generator = lambda h, w: np.random.uniform(-0.5, 0.5, (h, w))
    >>> n = 10
    >>> k = 0.2
    >>> updator = lambda x, noise: x + n * k * noise
    >>> noisy_image = gf(image, generator, updator)
    >>> noisy_image.shape
    (100, 100)
    """

    image_array = np.asarray(image)
    nb_chanel = 1

    if image_array.ndim != 3:
        height, width = image_array.shape
    else:
        height, width, nb_chanel = image_array.shape

    if nb_chanel == 1 or identical:
        noise_filter = generator(height, width)

    # final output numpy array
    output_array = []

    # check number of chanel
    if nb_chanel == 1:

        noisy_image = np.array(list(map(updator, image_array, noise_filter)))

        # normalise values
        return np.array(list(map(_normalise, noisy_image)), 'uint8')

    else:
        # final output numpy array
        output_array = []

        for chanel in range(0, nb_chanel):

            # getting flatten information from image and noise
            image_array_chanel = image_array[:, :, chanel]

            # redefine noise if necessary
            if not identical:
                noise_filter = generator(height, width)

            # compute new pixel value
            # x + n * k * white_noise_filter[i] as example
            noisy_image = np.array(
                list(map(updator, image_array_chanel, noise_filter)))

            # normalise values
            noisy_image = np.array(list(map(_normalise, noisy_image)), 'uint8')

            # in order to concatenate output array
            noisy_image = noisy_image[:, :, np.newaxis]

            # append new chanel
            output_array.append(noisy_image)

        # concatenate RGB image
        output_array = np.concatenate(output_array, axis=2)

        return output_array


def white_noise(image,
                n,
                identical=False,
                distribution_interval=(-0.5, 0.5),
                k=0.2):
    """White noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (-0.5, 0.5))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.2)

    Returns:
        2D Numpy array with white noise applied

    Example:

    >>> from ipfml.filters.noise import white_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = white_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda h, w: np.random.uniform(a, b, (h, w))

    updator = lambda x, noise: x + n * k * noise

    return _global_noise_filter(image, generator, updator, identical)


def gaussian_noise(image,
                   n,
                   identical=False,
                   distribution_interval=(0, 1),
                   k=0.1):
    """Gaussian noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (0, 1))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.1)

    Returns:
        2D Numpy array with gaussian noise applied

    Example:

    >>> from ipfml.filters.noise import gaussian_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = gaussian_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda h, w: np.random.normal(a, b, (h, w))

    updator = lambda x, noise: x + n * k * noise

    return _global_noise_filter(image, generator, updator, identical)


def laplace_noise(image,
                  n,
                  identical=False,
                  distribution_interval=(0, 1),
                  k=0.1):
    """Laplace noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (0, 1))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.1)

    Returns:
        2D Numpay array with Laplace noise applied

    Example:

    >>> from ipfml.filters.noise import laplace_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = laplace_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda h, w: np.random.laplace(a, b, (h, w))

    updator = lambda x, noise: x + n * k * noise

    return _global_noise_filter(image, generator, updator, identical)


def cauchy_noise(image,
                 n,
                 identical=False,
                 distribution_interval=(0, 1),
                 k=0.0002):
    """Cauchy noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (0, 1))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.0002)

    Returns:
        2D Numpy array with Cauchy noise applied

    Example:

    >>> from ipfml.filters.noise import cauchy_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = cauchy_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda h, w: np.random.standard_cauchy((h, w))

    updator = lambda x, noise: x + n * k * noise

    return _global_noise_filter(image, generator, updator, identical)


def log_normal_noise(image,
                     n,
                     identical=False,
                     distribution_interval=(0, 1),
                     k=0.05):
    """Log-normal noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (0, 1))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.05)

    Returns:
        2D Numpy array with Log-normal noise applied

    Example:

    >>> from ipfml.filters.noise import log_normal_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = log_normal_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda h, w: np.random.lognormal(a, b, (h, w))

    updator = lambda x, noise: x + n * k * noise

    return _global_noise_filter(image, generator, updator, identical)


def mut_white_noise(image,
                    n,
                    identical=False,
                    distribution_interval=(0, 1),
                    k=0.002):
    """Multiplied White noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (0, 1))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.002)

    Returns:
        2D Numpy array with multiplied white noise applied

    Example:

    >>> from ipfml.filters.noise import mut_white_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = mut_white_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    min_value = 1 - (k / 2)
    max_value = 1 + (k / 2)

    a, b = distribution_interval
    generator = lambda h, w: min_value + (np.random.uniform(a, b, (h, w)) * (max_value - min_value))

    updator = lambda x, noise: x * pow(noise, n)

    return _global_noise_filter(image, generator, updator, identical)


def salt_pepper_noise(image, n, identical=False, p=0.1, k=0.5):
    """Pepper salt noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        p: probability to increase pixel value otherwise decrease it
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.5)

    Returns:
        2D Numpy array with salt and pepper noise applied

    Example:

    >>> from ipfml.filters.noise import salt_pepper_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = salt_pepper_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    def _generator(h, w):

        x = w * h
        nb_elem = int(p * x)

        elements = np.full(x, 0)
        elements[0:nb_elem] = 1
        np.random.shuffle(elements)

        return elements.reshape(h, w)

    image_array = np.array(image)
    if image_array.ndim == 3:
        height, width, _ = image_array.shape
    else:
        height, width = image_array.shape

    # need same random variable for each pixel value if identical
    if identical:
        gen = np.random.uniform(0, 1, height * width)
        for i in range(2):
            gen = np.insert(gen, 0, gen)
        gen = iter(gen)

    # here noise variable is boolean to update or not pixel value
    def _updator(x, noise):

        # apply specific changes to each value of 1D array
        if isinstance(x, np.ndarray):
            return np.array(list(map(_updator, x, noise)))

        # probabilty to increase or decrease pixel value
        if identical:
            rand = next(gen)
        else:
            rand = random.uniform(0, 1)

        if noise:
            if rand > 0.5:
                return x + n * k
            else:
                return x - n * k
        else:
            return x

    return _global_noise_filter(image, _generator, _updator, identical)
