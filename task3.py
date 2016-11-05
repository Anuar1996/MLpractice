import numpy as np


def vect_is_equal_multisets(x, y):
    """
     Vectorized implementation of function, which returns True
     if two vectors x and y equal as multiset

     Input parameters:
        x: 1d numpy.array, the first vector
        y: 1d numpy.array, the second vector

     Output parameter:
        res: boolean, result of function
    """
    if x.shape[0] != y.shape[0]:
        return False

    multiset_x = np.sort(x)
    multiset_y = np.sort(y)
    res = np.all(multiset_x == multiset_y)

    return res


def is_equal_multisets(x, y):
    """
     Implementation with loops of function, which returns True
     if two vectors x and y equal as multiset

     Input parameters:
        x: 1d numpy.array, the first vector
        y: 1d numpy.array, the second vector

     Output parameter:
        res: boolean, result of function
    """
    multiset_cnt = x.shape[0]

    for i in range(multiset_cnt):
        if np.sum(x[i] == x) != np.sum(x[i] == y):
            return False

    return True


def other_is_equal_multisets(x, y):
    """
     The third implementation of function, which returns True
     if two vectors x and y equal as multiset

     Input parameters:
        x: 1d numpy.array, the first vector
        y: 1d numpy.array, the second vector

     Output parameter:
        res: boolean, result of function
    """

    x_nums, x_counts = np.unique(x, return_counts=True)
    y_nums, y_counts = np.unique(y, return_counts=True)
    return np.all(x_counts == y_counts) and np.all(x_nums == y_nums)
