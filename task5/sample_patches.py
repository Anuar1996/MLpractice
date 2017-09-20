import numpy as np

def normalize_data(images):
    
    #нормализуем, отбрасывая значения, выходящие за три стандартных отклонения
    mu = images.mean(axis=0)
    images -= mu
    sigma = 3 * images.std(axis=0)
    images = np.maximum(np.minimum(images, sigma), -sigma)
    
    #преобразуем в отрезок [-1, 1]
    images /= sigma
    
    #преобразуем в отрезок [0.1, 0.9]
    images = (images + 1) * 0.4 + 0.1
    
    return images
    
def sample_patches_raw(images, num_patches=10000, patch_size=8):
    N, D = images.shape
    im_size = int(np.sqrt(D / 3))
    
    x = np.random.choice(im_size - patch_size, num_patches)
    y = np.random.choice(im_size - patch_size, num_patches)
    images_id = np.random.choice(np.arange(N), num_patches)
    patches = np.zeros((num_patches, patch_size * patch_size * 3))
    im_matrix = images.reshape((N, im_size, im_size, 3))
    for i in range(num_patches):
        patche = im_matrix[images_id[i], x[i]:x[i] + patch_size, y[i]:y[i] + patch_size, :]
        patches[i:,:] = patche.reshape(patch_size * patch_size * 3)
    
    return patches

def sample_patches(images, num_patches=10000, patch_size=8):
    N, D = images.shape
    im_size = int(np.sqrt(D / 3))
    
    x = np.random.choice(im_size - patch_size, num_patches)
    y = np.random.choice(im_size - patch_size, num_patches)
    images_id = np.random.choice(np.arange(N), num_patches)
    patches = np.zeros((num_patches, patch_size * patch_size * 3))
    im_matrix = images.reshape((N, im_size, im_size, 3))
    for i in range(num_patches):
        patche = im_matrix[images_id[i], x[i]:x[i] + patch_size, y[i]:y[i] + patch_size, :]
        patches[i:,:] = patche.reshape(patch_size * patch_size * 3)
    
    return normalize_data(patches)