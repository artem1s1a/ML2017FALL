import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
import sys

data = pd.read_csv(sys.argv[1])
X = data.iloc[:, 1].values
y = data.iloc[:, 0].values
X = X.reshape((-1, 1))
y = y.reshape((-1, 1))

X_train = np.zeros((X.shape[0], 48, 48, 1))
for i in range(X.shape[0]):
    X_train[i] = np.fromstring(X[i, 0], dtype=np.float64, sep=' ').reshape((1, 48, 48, 1))

y_train = np.zeros((y.shape[0], 7))
for i in range(y.shape[0]):
    y_train[i, y[i, 0]] = 1

# np.save('X_train.npy', X_train)
# np.save('y_train.npy', y_train)

X_train /= 255

'''
np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(48, 48, 1)))
model.add(Conv2D(128, (3, 3), activation="selu"))
model.add(Dropout(0.5))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(48, (4, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Dropout(0.5))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(48, (7, 4), activation='relu'))
model.add(Dropout(0.5))
model.add(AveragePooling2D((3, 3), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
'''
model = load_model('data/model9.h5')
# model.load_weights('data/model_w.h5')

adad = Adadelta(lr=1, rho=0.95, epsilon=1e-08)
model.compile(optimizer=adad, loss='categorical_crossentropy', metrics=['accuracy'])

# model.save('data/model.h5')
'''
filepath = 'data/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
callbacks = [
    EarlyStopping(monitor='val_acc', min_delta=0.005, patience=20, verbose=0, mode='auto'),
    ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto',
                    period=5)
]
'''
# history = model.fit(X_train, y_train, batch_size=256, epochs=5, validation_split=0.1, callbacks=callbacks)
model.fit(X_train, y_train, batch_size=256, epochs=100, validation_split=0)

'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('CNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('result/Train.png')
'''



