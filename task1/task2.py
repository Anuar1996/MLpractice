import numpy as np


def vect_get_array_by_inds(matrix, row_inds, col_inds):
    """
     Vectorized implementation of function,
     which returns vector of a matrix' elements,
     given two vectors of indices.

     Input parameters:
        matrix: 2d-numpy.array, input matrix
        row_inds: 1xN numpy.array, row indexes for result array
        col_inds: 1xN numpy.array, col indexes for result array

     Output parameter:
        res_array: 1xN numpy.array, result array
    """
    res_array = matrix[row_inds, col_inds]

    return res_array


def get_array_by_inds(matrix, row_inds, col_inds):
    """
     Implementation with loops of function,
     which returns vector of a matrix' elements,
     given two vectors of indices.

     Input parameters:
        matrix: 2d-numpy.array, input matrix
        row_inds: 1xN numpy.array, row indexes for result array
        col_inds: 1xN numpy.array, col indexes for result array

     Output parameter:
        res_array: 1xN numpy.array, result array
    """
    res_array = []
    res_len = row_inds.shape[0]

    for row in range(res_len):
        for col in range(res_len):
            if row == col:
                res_array.append(matrix[row_inds[row], col_inds[col]])
    res_array = np.array(res_array)

    return res_array


def other_get_array_by_inds(matrix, row_inds, col_inds):
    """
     The third implementation of function,
     which returns vector of a matrix' elements,
     given two vectors of indices.

     Input parameters:
        matrix: 2d-numpy.array, input matrix
        row_inds: 1xN numpy.array, row indexes for result array
        col_inds: 1xN numpy.array, col indexes for result array

     Output parameter:
        res_array: 1xN numpy.array, result array
    """
    res_array = []
    res_len = row_inds.shape[0]

    for i in range(res_len):
        res_array.append(matrix[row_inds[i], col_inds[i]])
    res_array = np.array(res_array)

    return res_array
