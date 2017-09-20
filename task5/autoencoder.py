import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def KL_div(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def initialize(hidden_size, visible_size):
    L = hidden_size.shape[0]
    b = np.zeros(np.sum(hidden_size))
    b = np.concatenate((b, np.zeros(visible_size)))
    
    theta = np.array([])
    bound = np.sqrt(6 / (visible_size + hidden_size[0] + 1))
    W = np.random.random(visible_size * hidden_size[0]) * 2 * bound - bound
    theta = np.concatenate((theta, W))
    for i in range(L - 1):
        bound = np.sqrt(6 / (hidden_size[i] + hidden_size[i + 1] + 1))
        W = np.random.random(hidden_size[i] * hidden_size[i + 1]) * 2 * bound - bound
        theta = np.concatenate((theta, W))
        
    bound = np.sqrt(6 / (hidden_size[L - 1] + visible_size + 1))
    W = np.random.random(hidden_size[L - 1] * visible_size) * 2 * bound - bound
    theta = np.concatenate((theta, W))
    theta = np.concatenate((theta, b))
    return theta

def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    
    ##Считывание данных с theta
    L = hidden_size.shape[0]
    l_bound = 0
    W = list()
    W_l = theta[l_bound: l_bound + visible_size * hidden_size[0]]
    W_l = W_l.reshape((hidden_size[0], visible_size))
    W.append(W_l)
    l_bound += visible_size * hidden_size[0]
    for i in range(L - 1):
        W_l = theta[l_bound: l_bound + hidden_size[i] * hidden_size[i + 1]]
        W_l = W_l.reshape((hidden_size[i + 1], hidden_size[i]))
        W.append(W_l)
        l_bound += hidden_size[i] * hidden_size[i + 1]
    W_l = theta[l_bound: l_bound + visible_size * hidden_size[L - 1]]
    W_l = W_l.reshape((visible_size, hidden_size[L - 1]))
    W.append(W_l)
    l_bound += visible_size * hidden_size[L - 1]
    b = list()
    for i in range(L):
        b_l = theta[l_bound: l_bound + hidden_size[i]]
        b.append(b_l)
        l_bound += hidden_size[i]
    b_l = theta[l_bound: l_bound + visible_size]
    b.append(b_l)
    
    ##Прямой проход
    num_patches = data.shape[0]
    z = list()
    a = list()
    a.append(data)
    z_l = W[0].dot(data.T) + np.tile(b[0], (num_patches, 1)).T #входы в первый скрытый слой
    z.append(z_l)
    a_l = sigmoid(z_l) #выходы с первого скрытого слоя
    a.append(a_l)
    for i in range(1, L):
        z_l = W[i].dot(a_l) + np.tile(b[i], (num_patches, 1)).T #входы в скрытые слоя
        z.append(z_l)
        a_l = sigmoid(z_l) #выходы со скрытых слоёв
        a.append(a_l)
    z_l = W[L].dot(a_l) + np.tile(b[L], (num_patches, 1)).T #входы в последний/видимый слой
    z.append(z_l)
    h_W_b = sigmoid(z_l)
    
    ##Средний скрытый слой - разрежен
    center = int(L / 2)
    rho_param = np.sum(a[center + 1], axis=1) / num_patches
    rho = np.tile(sparsity_param, hidden_size[center])
    
    ##Значение целевой функции
    regul = 0
    for W_l in W:
        regul += np.sum(W_l ** 2)
    sparse = np.sum(KL_div(rho, rho_param))
    
    loss_func = np.sum((h_W_b - data.T) ** 2) / (2 * num_patches) + \
               (lambda_ / 2) * regul + beta * sparse
        
    ##Обратное распространение ошибки
    delta = []
    delta_l = -(data.T - h_W_b) * sigmoid_grad(z[-1])
    delta.append(delta_l)
    for i in range(L - 1, -1, -1):
        if (i == center):
            sparse = (-rho / rho_param + (1 - rho) / (1 - rho_param))
            sparse = np.tile(sparse, (num_patches, 1)).T
            delta_l = ((W[i + 1].T).dot(delta_l) + beta * sparse) * sigmoid_grad(z[i])
            delta.append(delta_l)
        else:
            delta_l = (W[i + 1].T).dot(delta_l) * sigmoid_grad(z[i])
            delta.append(delta_l)
    z.clear()
    b.clear()
    grad_func = np.array([])
    b_grad = []
    W_l_grad = delta[-1].dot(a[0]) / num_patches + lambda_ * W[0]
    W_l_grad = W_l_grad.reshape((hidden_size[0] * visible_size))
    grad_func = np.concatenate((grad_func, W_l_grad))
    b_l_grad = np.sum(delta[-1], axis=1) / num_patches
    b_grad.append(b_l_grad)
    
    for i in range(1, L):
        W_l_grad = delta[-i - 1].dot(a[i].T) / num_patches + lambda_ * W[i]
        W_l_grad = W_l_grad.reshape((hidden_size[i] * hidden_size[i - 1]))
        grad_func = np.concatenate((grad_func, W_l_grad))
        b_l_grad = np.sum(delta[-i - 1], axis=1) / num_patches
        b_grad.append(b_l_grad)
    W_l_grad = delta[0].dot(a[L].T) / num_patches + lambda_ * W[L]
    W_l_grad = W_l_grad.reshape((visible_size * hidden_size[L - 1]))
    grad_func = np.concatenate((grad_func, W_l_grad))
    b_l_grad = np.sum(delta[0], axis=1) / num_patches
    b_grad.append(b_l_grad)
    
    for b_l_grad in b_grad:
        grad_func = np.concatenate((grad_func, b_l_grad))
    
    return (loss_func, grad_func)

def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
       
    #считывание данных с theta
    L = hidden_size.shape[0]
    l_bound = 0
    W = list()
    W_l = theta[l_bound: l_bound + visible_size * hidden_size[0]]
    W_l = W_l.reshape((hidden_size[0], visible_size))
    W.append(W_l)
    l_bound += visible_size * hidden_size[0]
    for i in range(L - 1):
        W_l = theta[l_bound: l_bound + hidden_size[i] * hidden_size[i + 1]]
        W_l = W_l.reshape((hidden_size[i + 1], hidden_size[i]))
        W.append(W_l)
        l_bound += hidden_size[i] * hidden_size[i + 1]
    W_l = theta[l_bound: l_bound + visible_size * hidden_size[L - 1]]
    W_l = W_l.reshape((visible_size, hidden_size[L - 1]))
    W.append(W_l)
    l_bound += visible_size * hidden_size[L - 1]
    b = list()
    for i in range(L):
        b_l = theta[l_bound: l_bound + hidden_size[i]]
        b.append(b_l)
        l_bound += hidden_size[i]
    b_l = theta[l_bound: l_bound + visible_size]
    b.append(b_l)
    
    
    num_patches = data.shape[0]
    z = list()
    a = list()
    a.append(data)
    z_l = W[0].dot(data.T) + np.tile(b[0], (num_patches, 1)).T #входы в первый скрытый слой
    z.append(z_l)
    a_l = sigmoid(z_l) #выходы с первого скрытого слоя
    a.append(a_l)
    for i in range(1, L):
        z_l = W[i].dot(a_l) + np.tile(b[i], (num_patches, 1)).T #входы в скрытые слоя
        z.append(z_l)
        a_l = sigmoid(z_l) #выходы со скрытых слоёв
        a.append(a_l)
    z_l = W[L].dot(a_l) + np.tile(b[L], (num_patches, 1)).T #входы в последний/видимый слой
    z.append(z_l)
    a_l = sigmoid(z_l)
    a.append(a_l)
    return (a[layer_number - 1].T, W[layer_number - 1])