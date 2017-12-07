import numpy as np
import pickle
from util import *
from gensim.models.word2vec import Word2Vec
import sys

X, y_train = readTrain(sys.argv[1], True)

lines = []
for line in X:
    lines.append(line.split())
del X

lines = padLines(lines, 40)

model = Word2Vec.load('data/vectors_1.pkl')
X_train = ByWord2Vector(lines, model, dim=256)
del model, lines

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM, GRU
from keras.callbacks import EarlyStopping

'''
model = load_model('data/model1-LSTMr.h5')
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_acc', min_delta=0.002, patience=5, mode='auto'),
    # ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, period=5)
]
# history = model.fit(X_train, y_train, batch_size=1024, epochs=10, validation_split=0.1)
model.fit(X_train, y_train, batch_size=1024, epochs=50, validation_split=0.1, callbacks=callbacks)

model.save('data/model1-LSTMr-1.h5')
'''


model = Sequential()
model.add(GRU(128, dropout=0.5, return_sequences=True, input_shape=(40, 256)))
model.add(GRU(128, dropout=0.4, return_sequences=True))
model.add(GRU(128, dropout=0.3, return_sequences=True))
model.add(GRU(128, dropout=0.2, return_sequences=False))
model.add(Dense(256, activation='selu'))
model.add(Dropout(0.7))
model.add(Dense(128, activation='selu'))
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.002, patience=5, mode='auto')]
model.fit(X_train, y_train, batch_size=1024, epochs=50, validation_split=0.1, callbacks=callbacks)
# model.save('data/fuckmydeletekey.h5')

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
