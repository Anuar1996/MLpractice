import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageFilter

def display_layer(X, filename='layer.png'):
    N, D = X.shape
    num_pixels = int(D / 3)
    d = int(np.sqrt(num_pixels)) #пикселей по строком и столбцам
    
    m = int(np.round(np.sqrt(N))) #кол-во строк
    n = int(np.ceil(N / m)) #кол-во столбцов
    
    if np.min(X) >= 0:
        X = X - np.mean(X)
    
    R = np.zeros((N, num_pixels))
    G = np.zeros((N, num_pixels))
    B = np.zeros((N, num_pixels))
    
    for i in range(num_pixels):
        R[:, i] = X[:, 3 * i]
        G[:, i] = X[:, 3 * i + 1]
        B[:, i] = X[:, 3 * i + 2]
    
    #в отрезок [-1, 1]
    R /= np.max(np.abs(R))
    G /= np.max(np.abs(G))
    B /= np.max(np.abs(B))
    
    d_w = d + 1
    
    image = np.ones(((n - 1) * d_w + d, (m - 1) * d_w + d, 3))
    
    
    for i in range(n):
        for j in range(m):
            if (i * m + j == N):
                break
            image[i * d_w: i * d_w + d, j * d_w: j * d_w + d, 0] = R[i * m + j, :].reshape(d, d)
            image[i * d_w: i * d_w + d, j * d_w: j * d_w + d, 1] = G[i * m + j, :].reshape(d, d)
            image[i * d_w: i * d_w + d, j * d_w: j * d_w + d, 2] = B[i * m + j, :].reshape(d, d)
    
    #в отрезок [0, 1]
    image = (image + 1) / 2
    
    img = Image.fromarray(np.uint8(image * 255), 'RGB')
    img = img.resize((512, 512))
    img.save(filename)