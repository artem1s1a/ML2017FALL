import numpy as np
import pickle
from util import *
import sys

users, movies, ratings, n_users, n_movies = readTrain('data/train.csv')
X = np.concatenate((users.reshape((-1, 1)), movies.reshape((-1, 1))), axis=1)
X = np.concatenate((X, ratings.reshape((-1, 1))), axis=1)
np.random.shuffle(X)
users = X[:, 0]
movies = X[:, 1]
ratings = X[:, 2]
del X
ratings, mean, std = normalize(ratings)
with open('data/para.pkl', 'wb') as f:
    pickle.dump([mean, std], f)

from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import Embedding, dot, Flatten, Input
from keras.callbacks import ModelCheckpoint

inputU = Input(shape=(1, ))
U = Embedding(n_users, 192, input_length=1)(inputU)
U = Dropout(0.3)(U)
U = Flatten()(U)
inputM = Input(shape=(1, ))
M = Embedding(n_movies, 192, input_length=1)(inputM)
M = Dropout(0.3)(M)
M = Flatten()(M)
R = dot([U, M], axes=1)

model = Model(inputs=[inputU, inputM], outputs=R)
model.compile(loss='mse', optimizer='adamax')
filepath = 'model/model.{epoch:02d}-{val_loss:.3f}.hdf5'
callbacks = [ModelCheckpoint(filepath, save_weights_only=False, mode='auto', period=1)]
model.fit([users, movies], ratings, batch_size=256, epochs=200, validation_split=0.1, callbacks=callbacks)
model.save('model/model.h5')


'''
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('RNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('result/RNN_Train.png')
'''
