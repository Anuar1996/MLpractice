import numpy as np
import scipy as sc
from math import pi, log


def vect_log_normal(X, mu, sigma):
    """
     Vectorized implementation of function, which
     return logarithm of multivariate normal distribution
     with parameters mu and sigma

     Input parameters:
        X: 2d-numpy.array with shape (n, d), input points
        mu: 1d-numpy.array with shape d, mean of normal distribution
        sigma: 2d-numpy.array with shape (d, d), covariance matrix of normal distribution

     Output parameter:
        res_log: 1d-numpy.array with shape n, logarithm of normal distribution for
        input points
    """
    d = mu.shape[0]
    sum1 = -(d / 2) * log(2 * pi)
    cholesky_sigma = np.linalg.cholesky(sigma)
    sum2 = -np.log(np.diag(cholesky_sigma)).sum()
    sigma_inv = np.linalg.inv(sigma)
    sum3 = (-0.5) * np.diag(((X - mu).dot(sigma_inv)).dot((X - mu).T))
    res_log = sum1 + sum2 + sum3

    return res_log


def log_normal(X, mu, sigma):
    """
     Implementation with loops of function, which
     return logarithm of multivariate normal distribution
     with parameters mu and sigma

     Input parameters:
        X: 2d-numpy.array with shape (n, d), input points
        mu: 1d-numpy.array with shape d, mean of normal distribution
        sigma: 2d-numpy.array with shape (d, d), covariance matrix of normal distribution

     Output parameter:
        res_log: 1d-numpy.array with shape n, logarithm of normal distribution for
        input points
    """
    n, d = X.shape
    res_log = np.zeros(n)
    sigma_inv = np.linalg.inv(sigma)

    sum1 = -(d / 2) * log(2 * pi)

    cholesky_sigma = np.linalg.cholesky(sigma)
    sum2 = -np.log(np.diag(cholesky_sigma)).sum()

    for i in range(n):
        tmp_prod = []
        x_diff_mu = []
        for j in range(d):
            x_diff_mu.append(X[i][j] - mu[j])
        x_diff_mu = np.array(x_diff_mu)
        for j in range(d):
            sum_3 = 0
            for k in range(d):
                sum_3 += x_diff_mu[k] * sigma_inv[k, j]
            tmp_prod.append(sum_3)
        x_diff_mu = x_diff_mu.reshape(d, 1)
        for j in range(d):
            sum_3 += tmp_prod[j] * x_diff_mu[j, 0]
        sum_3 = -0.5 * sum_3
        res_log[i] = sum1 + sum2 + sum_3

    return res_log


def other_log_normal(X, mu, sigma):
    """
     The third implementation of function, which
     return logarithm of multivariate normal distribution
     with parameters mu and sigma

     Input parameters:
        X: 2d-numpy.array with shape (n, d), input points
        mu: 1d-numpy.array with shape d, mean of normal distribution
        sigma: 2d-numpy.array with shape (d, d), covariance matrix of normal distribution

     Output parameter:
        res_log: 1d-numpy.array with shape n, logarithm of normal distribution for
        input points
    """
    n, d = X.shape
    res_log = np.zeros(n)
    sigma_inv = np.linalg.inv(C)

    sum1 = -(d / 2) * log(2 * pi)

    cholesky_sigma = np.linalg.cholesky(sigma)
    sum2 = -np.log(np.diag(cholesky_sigma)).sum()

    for i in range(N):
        x_diff_mu = []
        for j in range(D):
            x_diff_mu.append(X[i][j] - mu[j])
        x_diff_mu = np.array(x_diff_mu)
        sum_3 = x_diff_mu.dot(sigma_inv)
        x_diff_mu = x_diff_mu.reshape(d, 1)
        sum_3 = sum_3.dot(x_diff_mu)
        sum_3 = -0.5 * sum_3
        res_log[i] = sum1 + sum2 + sum_3[0]

    return res_log
