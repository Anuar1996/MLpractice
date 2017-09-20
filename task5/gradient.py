import numpy as np

def grad_simple_func(x):
    func = x[0] ** 2 + x[0] * x[1] + x[1] ** 2 + x[2] ** 2
    grad = np.zeros(3)
    
    grad[0] = 2 * x[0] + x[1]
    grad[1] = x[0] + 2 * x[1]
    grad[2] = 2 * x[2]
    
    return (func, grad)

def compute_gradient(J, theta):
    n = theta.shape[0]
    grad = np.zeros(n)
    eps = 1e-4
    
    for i in range(n):
        step = np.zeros(n)
        step[i] = 1
        grad[i] = (J(theta + eps * step)[0] - J(theta - eps * step)[0]) / (2 * eps)
    
    return grad

def check_gradient():
    
    x = np.array([1, 2, 3])
    func_value, grad_1 = grad_simple_func(x)
    grad_2 = compute_gradient(grad_simple_func, x)
    
    error = np.linalg.norm(grad_1 - grad_2)
    print('Difference between gradients: ', error)
