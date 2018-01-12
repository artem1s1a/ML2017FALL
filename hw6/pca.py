from skimage import io
import numpy as np
import os
import sys


def trans(Y, X_M):
    Y += X_M
    Y -= np.min(Y)
    Y /= np.max(Y)
    Y = (Y * 255).astype(np.uint8)
    Y = np.reshape(Y, (600, 600, 3))
    return Y


dir = sys.argv[1]
img_list = os.listdir(dir)

X = np.zeros((len(img_list), 1080000))

i = 0
for path in img_list:
    img = io.imread(dir + '/' + path)
    img = img.flatten()
    X[i] = img
    i = i+1

X_mean = X.mean(axis=0)
X = (X - X_mean).T

U, s, V = np.linalg.svd(X, full_matrices=False)

k = 4

img = io.imread(dir + '/' + sys.argv[2])
img = img.flatten()
Y = (img - X_mean).T
col = U[:, :k]
col = col.T
weight = np.dot(col, Y)

Y_re = np.zeros((1, 1080000))
for i in range(k):
    Y_re += weight[i] * col[i]
Y_re = trans(Y_re, X_mean)
io.imsave('reconstruction.jpg', Y_re)
