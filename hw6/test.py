import pandas as pd
import numpy as np
import sys

# from PIL import Image

# image = np.load('data/image.npy')
# image = np.reshape(image, (-1, 28, 28))
# image = 255 - image
data = pd.read_csv(sys.argv[2]).values
data = data[:, 1:]
y = np.zeros(data.shape[0])

# x = [1, 2, 4, 6]   for label_82.npy
x = [1, 2, 3, 6, 7, 10, 11, 12, 13]  # for label_94
label = np.load('data/label_94.npy')

for i in range(data.shape[0]):
    if (label[data[i, 0]] in x and label[data[i, 1]] in x) or (
            label[data[i, 0]] not in x and label[data[i, 1]] not in x):
        y[i] = 1
        # im = Image.fromarray(image[data[i, 0]])
        # im.show()
        # im = Image.fromarray(image[data[i, 1]])
        # im.show()

y = y.astype(int)
result = pd.DataFrame(y, columns=['Ans'])
result.to_csv(sys.argv[3], index_label='ID')
