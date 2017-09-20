import numpy as np
from copy import deepcopy

def gen_pow_matrix(primpoly):
    
    poly_repr = np.binary_repr(primpoly)
    res = []
    alpha_ = 2
    res.append(alpha_)
    for i in range(1, 2 ** (len(poly_repr) - 1) - 1):
        alpha_ = alpha_ << 1
        if (np.log2(alpha_) >= len(poly_repr) - 1):
            alpha_ = np.binary_repr(alpha_, width=len(poly_repr))
            alpha_ = int(alpha_, 2) ^ int(poly_repr, 2)
            res.append(alpha_)
        else:
            res.append(alpha_)
    res_2 = np.array(res)
    res_1 = np.argsort(res_2) + 1
    
    res = np.zeros((res_1.shape[0], 2)).astype('int')
    
    res[:, 0] = res_1
    res[:, 1] = res_2
    
    return res

def add(X, Y):
    return X ^ Y

def sum(X, axis=0):
    
    shape = X.shape
    new_shape = tuple(shape[i] for i in range(len(shape))
                     if i != axis)
    res = np.zeros(new_shape).astype('int')
    for i in range(shape[axis]):
        tmp = np.take(X, i, axis=axis)
        res = res ^ tmp
    return res

def prod(X, Y, pm):
    
    res = np.zeros(X.shape).astype('int')
    non_zero_XY = np.where(np.logical_and(X != 0, Y != 0))
    
    X_nonzero = X[non_zero_XY]
    Y_nonzero = Y[non_zero_XY]
    alpha_m = np.zeros(X.shape).astype('int')
    alpha_n = np.zeros(Y.shape).astype('int')
    alpha = np.zeros(Y.shape).astype('int')
    alpha_m[non_zero_XY] = pm[X_nonzero - 1, 0]
    alpha_n[non_zero_XY] = pm[Y_nonzero - 1, 0]
    alpha = (alpha_m + alpha_n) % pm.shape[0]
    alpha[non_zero_XY] = pm[alpha[non_zero_XY] - 1, 1]
    res[non_zero_XY] = alpha[non_zero_XY]
        
    return res

def divide(X, Y, pm):
    
    if (np.where(Y == 0)[0].shape[0] != 0):
        raise ValueError('Division by zero')
    res = np.zeros(X.shape).astype('int')
    alpha_m = np.zeros(X.shape).astype('int')
    alpha_n = np.zeros(Y.shape).astype('int')
    alpha = np.zeros(Y.shape).astype('int')
    non_zero_XY = np.where(np.logical_and(X != 0, Y != 0))
    
    X_nonzero = X[non_zero_XY]
    Y_nonzero = Y[non_zero_XY]
    
    alpha_m[non_zero_XY] = pm[X_nonzero - 1, 0]
    alpha_n[non_zero_XY] = pm[Y_nonzero - 1, 0]
    alpha = (alpha_m - alpha_n) % pm.shape[0]
    alpha[non_zero_XY] = pm[alpha[non_zero_XY] - 1, 1]
    res[non_zero_XY] = alpha[non_zero_XY]
        
    return res

def linsolve(A, b, pm):
    N, _ = A.shape
    alpha_A = pm[A - 1, 0]
    res = np.zeros(N).astype('int')
    Ab = np.hstack((A, b[:, np.newaxis]))
    
    
    #forward
    for i in range(N - 1):
        if (Ab[i][i] == 0):
            tmpAb = Ab[i + 1: ,:]
            j = np.argmax(tmpAb[:, i])
            j += (i + 1)
            Ab[[i, j], :] = Ab[[j, i], :]
        X = Ab[i, :]
        Y = np.tile(Ab[i][i], X.shape[0])
        try:
            Ab[i, :] = divide(X, Y, pm)
        except:
            return np.nan
            
        for j in range(i + 1, N):
            X = Ab[i, :]
            Y = np.tile(Ab[j, i], X.shape[0])
            tmp = prod(X, Y, pm)
            Ab[j, :] = Ab[j, :] ^ tmp
    
    diag = np.diag(Ab)
    if (np.where(diag == 0)[0].shape[0] != 0):
        return np.nan
    
    #back
    for i in range(N - 1, -1, -1):
        X = Ab[i, :]
        Y = np.tile(Ab[i][i], X.shape[0])
        Ab[i, :] = divide(X, Y, pm)
        
        for j in range(i - 1, -1, -1):
            X = Ab[i, :]
            Y = np.tile(Ab[j, i], X.shape[0])
            tmp = prod(X, Y, pm)
            Ab[j, :] = Ab[j, :] ^ tmp
    diag = np.diag(Ab)

    res = divide(Ab[:, N], diag, pm)
    
    return res

def minpoly(x, pm):
    
    alpha_m = pm[x - 1, 0]
    n = pm.shape[0]
    class_ = set()
    for i in range(x.shape[0]):
        cyclomatic_class = []
        cyclomatic_class.append(pm[alpha_m[i] - 1, 1])
        tmp = alpha_m[i]
        for j in range(1, n + 1):
            tmp = (tmp * 2) % n
            if (pm[tmp - 1, 1] in cyclomatic_class):
                class_ |= set(cyclomatic_class)
                break
            else:
                cyclomatic_class.append(pm[tmp - 1, 1])
    class_ = np.array(list(class_))
    
    pols = np.zeros((class_.shape[0], 2)).astype('int')
    pols[:, 0] = np.ones(class_.shape[0]).astype('int')
    pols[:, 1] = class_
    
    res = polyprod(np.array([1]), pols[0, :], pm)
    for i in range(pols.shape[0] - 1):
        res = polyprod(res, pols[i + 1, :], pm)
    res = normal_poly(res)
    
    return (res, class_)

def polyval(p, x, pm):
    
    deg = p.shape[0] - 1
    n = pm.shape[0]
     #degrees for x by alpha
    pol_degree = np.arange(deg + 1).astype('int')[::-1]
    res = np.zeros(x.shape[0]).astype('int')
    
    for i in range(x.shape[0]):
        if (x[i] == 0):
            res[i] = p[-1]
        else:
            alpha_x = pm[x[i] - 1, 0]
            alpha_ = (alpha_x * pol_degree) % n
            val_by_deg = pm[alpha_ - 1, 1]
            res[i] = sum(prod(val_by_deg, p, pm))
        
    return res

def polyprod(p1, p2, pm):
    if (p1.shape[0] == 1) and (p1[0] == 0):
        return np.array([0])
    if (p2.shape[0] == 1) and (p2[0] == 0):
        return np.array([0])
    deg_p1 = p1.shape[0] - 1
    deg_p2 = p2.shape[0] - 1
    deg_res = deg_p1 + deg_p2
    p1_full = np.zeros(deg_res + 1).astype('int')
    p2_full = np.zeros(deg_res + 1).astype('int')
    p1_full[deg_res - deg_p1: ] = p1
    p2_full[deg_res - deg_p2: ] = p2
    res = np.zeros(deg_res + 1).astype('int')
    
    for i in range(deg_res + 1):
        a = p1_full[i: ]
        b = p2_full[i: ][::-1]

        res[i] = sum(prod(a, b, pm))
    res = normal_poly(res)
    return res

def polydivmod(p1, p2, pm):
    deg_p1 = p1.shape[0] - 1
    deg_p2 = p2.shape[0] - 1
    
    
    if (p2.shape[0] == 1) and (p2[0] == 0):
        raise ValueError('Division by zero')
    if (p1.shape[0] == 1) and (p1[0] == 0):
        return (np.array([0]), np.array([0]))
    
    if (p2.shape[0] == 1):
        mod = np.array([0])
        div = divide(p1, np.tile(p2, p1.shape[0]), pm)
        return (div, mod)
    
    if (deg_p1 < deg_p2):
        mod = p1
        div = np.array([0])
        return (div, mod)
    dividend = p1.copy()
    div = np.zeros(deg_p1 - deg_p2 + 1).astype('int')
    while True:
        i = dividend.shape[0] - p2.shape[0]
        div[-i - 1] = divide(np.array([dividend[0]]), np.array([p2[0]]), pm)
        sub = np.zeros(dividend.shape[0])
        tmp = np.zeros(div.shape[0]).astype('int')
        tmp[-i - 1] = div[-i - 1]
        sub = polyprod(tmp, p2, pm)
        
        dividend = polyadd(dividend, sub)
        if dividend.shape[0] < p2.shape[0]:
            break

    div = div
    mod = dividend
    return (div, mod)

def polyadd(p1, p2):
    max_deg = max(p1.shape[0], p2.shape[0]) - 1
    
    p = np.zeros(max_deg + 1).astype('int')
    q = np.zeros(max_deg + 1).astype('int')
    
    p[-p1.shape[0]:] = p1
    q[-p2.shape[0]:] = p2
    
    res = p ^ q
    return normal_poly(res)

def normal_poly(p):
    
    if (np.all(p == 0)):
        p = np.array([0])
    else:
        ind = np.where(p != 0)[0][0]
        p = p[ind: ]
    
    return p

def euclid(p1, p2, pm, max_deg=0):
    x0, y0, x1, y1 = np.array([1]), np.array([0]), np.array([0]), np.array([1])
    
    p, q = p1.copy(), p2.copy()
    deg = q.shape[0] - 1
    mod = q.copy()
    
    while ((mod.shape[0] != 1) and (mod[0] != 0)) and (deg > max_deg):
        div, mod = polydivmod(p, q, pm)
        deg = mod.shape[0] - 1
        tmp = deepcopy(x1)
        x1 = polyadd(x0, polyprod(x1, div, pm))
        x0 = tmp
        tmp = deepcopy(y1)
        y1 = polyadd(y0, polyprod(y1, div, pm))
        y0 = tmp
        x1 = normal_poly(x1)
        y1 = normal_poly(y1)
            
        p = q
        q = mod
        
    
    return mod, x1, y1