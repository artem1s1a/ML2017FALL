import numpy as np
import pandas as pd
import pickle
from util import *
from keras.models import load_model
import sys


users, movies = readTest(sys.argv[1])
with open('data/para_1.pkl', 'rb') as f:
    mean, std = pickle.load(f)
model = load_model('model/model1.hdf5')
y1 = model.predict([users, movies], batch_size=2048)
y1 = y1 * std + mean
y1 = cut(y1)
with open('data/para_2.pkl', 'rb') as f:
    mean, std = pickle.load(f)
model = load_model('model/model2.h5')
y2 = model.predict([users, movies], batch_size=2048)
y2 = y2 * std + mean
y2 = cut(y2)
with open('data/para_0.pkl', 'rb') as f:
    mean, std = pickle.load(f)
model = load_model('model/model3.hdf5')
y3 = model.predict([users, movies], batch_size=2048)
y3 = y3 * std + mean
y3 = cut(y3)
y = y1 + y2 + y3
y /= 3


result = pd.DataFrame(y, columns=['Rating'])
result.index += 1
result.to_csv(sys.argv[2], index_label='TestDataID')
