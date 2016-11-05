import numpy as np

weights = np.array([0.299, 0.587, 0.114])


def vector_method(image):
    """
     Vectorized implementation of function, which
     return gray image with weights = [0.299, 0.587, 0.114]

     Input parameter:
     image: 3D-numpy.array, input image with three channels R, G, B

     Output parameter:
     gray_image: 2D-numpy.array, result gray image
    """
    gray_image = (image * weights).sum(axis=2)
    return gray_image


def cycle_method(image):
    """
     Implementation with loops of function, which
     return gray image with weights = [0.299, 0.587, 0.114]

     Input parameter:
        image: 3D-numpy.array, input image with three channels R, G, B

     Output parameter:
        gray_image: 2D-numpy.array, result gray image
    """
    height, width, num_channels = image.shape
    gray_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            for k in range(numChannels):
                gray_image[i, j] += image[i, j, k] * weights[k]

    return gray_image


def other_method(image):
    """
     The third implementation of function, which
     return gray image with weights = [0.299, 0.587, 0.114]

     Input parameter:
        image: 3D-numpy.array, input image with three channels R, G, B

     Output parameter:
        gray_image: 2D-numpy.array, result gray image
    """
    height, width, num_channels = image.shape
    gray_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            gray_image[i, j] = (image[i, j] * weights).sum()

    return gray_image
