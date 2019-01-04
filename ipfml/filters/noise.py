import numpy as np
from ipfml import processing


def _global_noise_filter(image,
                         n,
                         generator,
                         updator,
                         identical=False,
                         distribution_interval=(-0.5, 0.5),
                         k=0.2):
    """White noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        generator: lambda function used to generate random numpy array with specific distribution
        updator: lambda function used to update pixel value
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: tuple which set the distribution interval of uniform distribution (default (-0.5, 0.5))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.2)

    Returns:
        2D Numpy array with specified noise applied

    Usage:

    >>> from ipfml.filters.noise import _global_noise_filter as gf
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> generator = lambda x: np.random.uniform(-0.5, 0.5, x)
    >>> updator = lambda x, n, k, noise: x + n * k * noise
    >>> noisy_image = gf(image, 10, generator, updator)
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
        noise_filter = generator(nb_pixels)

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
            noise_filter = generator(nb_pixels)

        # compute new pixel value
        # n * k * white_noise_filter[i]
        noisy_image = np.asarray([
            updator(image_array_flatten[i], n, k, noise_filter[i])
            for i in range(0, nb_pixels)
        ])

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
    else:
        output_array = np.asarray(output_array).reshape(width, height)

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

    Usage:

    >>> from ipfml.filters.noise import white_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = white_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda x: np.random.uniform(a, b, x)

    updator = lambda x, n, k, noise: x + n * k * noise

    return _global_noise_filter(image, n, generator, updator, identical,
                                distribution_interval, k)


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

    Usage:

    >>> from ipfml.filters.noise import gaussian_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = gaussian_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda x: np.random.normal(a, b, x)

    updator = lambda x, n, k, noise: x + n * k * noise

    return _global_noise_filter(image, n, generator, updator, identical,
                                distribution_interval, k)


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

    Usage:

    >>> from ipfml.filters.noise import laplace_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = laplace_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda x: np.random.laplace(a, b, x)

    updator = lambda x, n, k, noise: x + n * k * noise

    return _global_noise_filter(image, n, generator, updator, identical,
                                distribution_interval, k)


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

    Usage:

    >>> from ipfml.filters.noise import cauchy_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = cauchy_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda x: np.random.standard_cauchy(x)

    updator = lambda x, n, k, noise: x + n * k * noise

    return _global_noise_filter(image, n, generator, updator, identical,
                                distribution_interval, k)


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

    Usage:

    >>> from ipfml.filters.noise import log_normal_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = log_normal_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda x: np.random.lognormal(a, b, x)

    updator = lambda x, n, k, noise: x + n * k * noise

    return _global_noise_filter(image, n, generator, updator, identical,
                                distribution_interval, k)


def mut_white_noise(image,
                    n,
                    identical=False,
                    distribution_interval=(-0.5, 0.5),
                    k=0.2):
    """Multiplied White noise filter to apply on image

    Args:
        image: image used as input (2D or 3D image representation)
        n: used to set importance of noise [1, 999]
        identical: keep or not identical noise distribution for each canal if RGB Image (default False)
        distribution_interval: set the distribution interval of normal law distribution (default (-0.5, 0.5))
        k: variable that specifies the amount of noise to be taken into account in the output image (default 0.2)

    Returns:
        2D Numpy array with multiplied white noise applied

    Usage:

    >>> from ipfml.filters.noise import mut_white_noise
    >>> import numpy as np
    >>> image = np.random.uniform(0, 255, 10000).reshape((100, 100))
    >>> noisy_image = mut_white_noise(image, 10)
    >>> noisy_image.shape
    (100, 100)
    """

    a, b = distribution_interval
    generator = lambda x: np.random.uniform(a, b, x)

    updator = lambda x, n, k, noise: x * n * k * noise

    return _global_noise_filter(image, n, generator, updator, identical,
                                distribution_interval, k)
