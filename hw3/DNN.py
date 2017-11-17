import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

X_train /= 255

model = Sequential()

model.add(Flatten(input_shape=(48, 48, 1)))
model.add(Dense(1028, activation='relu'))
model.add(Dropout(0.5))
for i in range(7):
    model.add(Dense(1028, activation='relu'))
    model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

adad = Adadelta(lr=1, rho=0.95, epsilon=1e-08)
model.compile(optimizer=adad, loss='categorical_crossentropy', metrics=['accuracy'])

model.save('data/DNN_model.h5')
model.summary()

history = model.fit(X_train, y_train, batch_size=256, epochs=100, validation_split=0.1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('result/Training.png')
