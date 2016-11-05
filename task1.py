import numpy as np


def vect_diag_prod(matrix):
    """
     Vectorized implementation of the function,
     which calculating product of nonzero elements
     on the matrix' diagonal

     Input parameter:
        matrix: 2d-numpy.array, input matrix

     Output parameter:
        res_prod: scalar, product of nonzero elements
    """

    matrix_diag = np.diag(matrix)
    res_prod = np.prod(matrix_diag[matrix_diag != 0])

    return res_prod


def diag_prod(matrix):
    """
     Implementation with loops of the function,
     which calculating product of nonzero elements
     on the matrix' diagonal

     Input parameter:
        matrix: 2d-numpy.array, input matrix

     Output parameter:
        res_prod: scalar, product of nonzero elements
    """

    res_prod = 1
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            if (matrix[i, j] != 0) & (i == j):
                res_prod *= int(matrix[i, j])

    return res_prod


def other_diag_prod(matrix):
    """
     The third implementation of the function,
     which calculating product of nonzero elements
     on the matrix' diagonal

     Input parameter:
        matrix: 2d-numpy.array, input matrix

     Output parameter:
        res_prod: scalar, product of nonzero elements
    """

    res_prod = 1
    rows, cols = matrix.shape

    for i in range(min(rows, cols)):
        if matrix[i, i] != 0:
            res_prod *= int(matrix[i, i])

    return res_prod
