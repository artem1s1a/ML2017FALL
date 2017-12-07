import numpy as np
import pandas as pd
from util import *
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import sys
# import time

# start_time = time.time()

X = readTest(sys.argv[1])

lines = []
for line in X:
    lines.append(line.split())
del X

lines = padLines(lines, 40)
model = Word2Vec.load('vectors_1.pkl')
X_test = ByWord2Vector(lines, model, dim=256)
del model

model = load_model('data/model1-LSTMr-1.h5')
# model.summary()
y1 = model.predict(X_test, batch_size=1024)
model = load_model('data/model2-LSTMr-1.h5')
# model.summary()
y2 = model.predict(X_test, batch_size=1024)
model = load_model('data/model3-LSTMr-1.h5')
# model.summary()
y3 = model.predict(X_test, batch_size=1024)
model = load_model('data/model4-GRUs-1.h5')
# model.summary()
y4 = model.predict(X_test, batch_size=1024)
model = load_model('data/model5-GRUs-1.h5')
# model.summary()
y5 = model.predict(X_test, batch_size=1024)
model = load_model('data/fuckmydeletekey.h5')
# model.summary()
y6 = model.predict(X_test, batch_size=1024)

y = (y1 + y2 + y3 + y4 + y5 + y6) / 6  # <- current best
# y_5 = (y1 + y2 + y3 + y4 + y5) / 5
# y_4 = (y1 + y2 + y3 + y4) / 4
# y = (y4 + y5 + y6) / 3
y_test = (y > 0.5).astype(int)
result = pd.DataFrame(y_test, columns=['label'])
result.to_csv(sys.argv[2], index_label='id')

# runtime = time.time() - start_time
# print('{}min {}sec'.format(int(runtime / 60), (int(runtime)) % 60))
