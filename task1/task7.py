import numpy as np


def vect_distance(matrix1, matrix2):
    """
     Vectorized implementation of function,
     which return euclidean distance between two
     input matrix
     Input parameters:
     matrix1: 2d-numpy.array with shape (N, K), input matrix
     matrix2: 2d-numpy.array with shape (M, K), input matrix
     Output parameter:
     res_distance: 2d-numpy.array with shape (N, M), distance
     between two matrix
    """
    res_distance = np.square(matrix2 - matrix1[:, np.newaxis, :])
    res_distance = np.sqrt(res_distance.sum(axis=2))

    return res_distance


def distance(matrix1, matrix2):
    """
     Implementation with loops of function,
     which return euclidean distance between two
     input matrix

     Input parameters:
        matrix1: 2d-numpy.array with shape (N, K), input matrix
        matrix2: 2d-numpy.array with shape (M, K), input matrix

     Output parameter:
        res_distance: 2d-numpy.array with shape (N, M), distance
        between two matrix
    """
    matr1_rows, cols = matrix1.shape
    matr2_rows, cols = matrix2.shape
    dist_shape = (matr1_rows, matr2_rows)

    res_distance = np.zeros(dist_shape)
    for i in range(matr1_rows):
        for j in range(matr2_rows):
            for k in range(cols):
                res_distance[i, j] += np.square(matrix1[i, k] - matrix2[j, k])
            res_distance[i, j] = np.sqrt(res_distance[i, j])

    return res_distance


def other_distance(matrix1, matrix2):
    """
     The third implementation of function,
     which return euclidean distance between two
     input matrix

     Input parameters:
        matrix1: 2d-numpy.array with shape (N, K), input matrix
        matrix2: 2d-numpy.array with shape (M, K), input matrix

     Output parameter:
        res_distance: 2d-numpy.array with shape (N, M), distance
        between two matrix
    """
    matr1_rows, cols = matrix1.shape
    matr2_rows, cols = matrix2.shape
    dist_shape = (matr1_rows, matr2_rows)

    res_distance = np.zeros(dist_shape)

    for i in range(X.shape[0]):
        res_distance[i] = np.sqrt(np.square(matrix1[i] - matrix2).sum(axis=1))

    return res_distance
