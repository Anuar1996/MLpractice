import numpy as np


def vect_max_be4_zero(x):
    """
     Vectorized implementation of function, which
     find maximum in input array x before 0.

     Input parameter:
     x: 1d-numpy.array, input array

     Output parameter:
     _max: scalar, result of function
    """
    zero_ind = np.where(x == 0)[0]
    x_size = x.shape[0]
    zero_ind_size = zero_ind.shape[0]

    if zero_ind[zero_ind_size - 1] == x_size - 1:
        zero_ind = np.delete(zero_ind, zero_ind_size - 1)
    _max = np.max(x[zero_ind + 1])

    return _max


def max_be4_zero(x):
    """
     Implementation with loops of function, which
     find maximum in input array x before 0.

     Input parameter:
        x: 1d-numpy.array, input array

     Output parameter:
        _max: scalar, result of function
    """
    zero_ind = []
    x_size = x.shape[0]

    for i in range(x_size):
        if (x[i] == 0) & (i + 1 < x_size):
            zero_ind.append(x[i + 1])
    zero_ind = np.array(zero_ind)
    zero_ind_size = zero_ind.shape[0]

    _max = zero_ind[0]
    for i in range(zero_ind_size):
        if zero_ind[i] >= _max:
            _max = zero_ind[i]

    return _max


def other_max_be4_zero(x):
    """
     The third implementation of function, which
     find maximum in input array x before 0.

     Input parameter:
        x: 1d-numpy.array, input array

     Output parameter:
        _max: scalar, result of function
    """
    zero_ind = []
    x_size = x.shape[0]

    for i in range(x_size):
        if (x[i] == 0) & (i + 1 < x_size):
            zero_ind.append(x[i + 1])
    _max = np.max(zero_ind)

    return _max
