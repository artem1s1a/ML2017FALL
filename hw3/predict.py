import pandas as pd
import numpy as np
from keras.models import load_model
import sys

data = pd.read_csv(sys.argv[1])
X = data.iloc[:, 1].values

X_test = np.zeros((X.shape[0], 48, 48, 1))
for i in range(X.shape[0]):
    X_test[i] = np.fromstring(X[i], dtype=np.float64, sep=' ').reshape((1, 48, 48, 1))

# np.save('data/X_test.npy', X_test)
# X_test = np.load('data/X_test.npy')

X_test /= 255

# 0.64 = 1 2 3 5
# 0.66 = 1 6 7 8
# 0.66 = 1 6 7 9
# 0.67 = 7 8 9 A B

model = load_model('modelB.h5')
model.load_weights('wB.hdf5')
y1 = model.predict(X_test, batch_size=512)

model = load_model('modelA.h5')
model.load_weights('wA.hdf5')
y2 = model.predict(X_test, batch_size=512)

model = load_model('model9.h5')
model.load_weights('w9.hdf5')
y3 = model.predict(X_test, batch_size=512)

model = load_model('model7.h5')
model.load_weights('w7.hdf5')
y4 = model.predict(X_test, batch_size=512)

model = load_model('model8.h5')
model.load_weights('w8.hdf5')
y5 = model.predict(X_test, batch_size=512)

y = y1 + y2 + y3 + y4 + y5

y_test = y.argmax(axis=-1)
'''
y_test = np.zeros((y.shape[0]))
for i in range(y.shape[0]):
    max = 0
    for j in range(7):
        if y[i, j] > max:
            y_test[i] = j
            max = y[i, j]

y_test = y_test.reshape((-1, 1)).astype(int)
'''
result = pd.DataFrame(y_test, columns=['label'])
result.to_csv(sys.argv[2], index_label='id')
