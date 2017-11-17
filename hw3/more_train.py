import pandas as pd
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adadelta

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

X_train /= 255

model = load_model('data/DNN_model.h5')
# model.load_weights('data/weights.04-0.25.hdf5')

adad = Adadelta(lr=1, rho=0.95, epsilon=1e-08)
# model.compile(optimizer=adad, loss='categorical_crossentropy', metrics=['accuracy'])
# filepath = 'data/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
# callbacks = [
#    EarlyStopping(monitor='val_acc', min_delta=0.005, patience=20, verbose=0, mode='auto'),
#    ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto',
#                    period=5)
# ]
model.fit(X_train, y_train, batch_size=256, epochs=100)

# model.save('data/model.h5')
model.save_weights('data/DNN_weights.hdf5')
