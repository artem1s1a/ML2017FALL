import pandas as pd
import numpy as np
import sys
# from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(X):
    X = X.T
    first = True
    for col in X:
        mean = np.mean(col)
        std = np.std(col)
        if first:
            Y = ((col - mean) / (std if std != 0 else 1)).reshape((1, -1))
            first = False
        else:
            Y = np.concatenate((Y, ((col - mean) / (std if std != 0 else 1)).reshape((1, -1))), axis=0)
    return Y.T


X_train = pd.read_csv(sys.argv[3]).drop('fnlwgt', axis=1).values.astype(np.float64)
y_train = pd.read_csv(sys.argv[4]).values
X_test = pd.read_csv(sys.argv[5]).drop('fnlwgt', axis=1).values.astype(np.float64)

X_train = normalize(X_train)
'''
valid = False
v = 3
N = X_train.shape[0]
if valid:
    data_train = np.append(X_train, y_train, axis=1)
    np.random.shuffle(data_train)
    X_train = data_train[:, :-1]
    y_train = data_train[:, -1:]

    X_valid = X_train[:N // v, :]
    y_valid = y_train[:N // v, :]
    X_train = X_train[N // v:, :]
    y_train = y_train[N // v:, :]

# add bias
X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis=1)
X_train_T = X_train.T
l_rate = 0.000001
repeat = 8000
lambda_ = 0
w = np.zeros((X_train.shape[1], 1))

for i in range(repeat):
    f_x = sigmoid(np.dot(X_train, w))
    ans = np.floor(f_x + 0.5)
    loss = y_train - f_x
    gra = np.dot(X_train_T, loss)
    w = w + l_rate * gra - lambda_*w
    print('{} th acc : {}'.format(i, accuracy_score(y_train, (f_x > 0.5).astype(int))))

if valid:
    X_valid = np.append(X_valid, np.ones((X_valid.shape[0], 1)), axis=1)
    print('valid : {}'.format(accuracy_score(y_valid, (np.dot(X_valid, w) > 0.5).astype(int))))

np.save('data/logistic.npy', w)
'''
w = np.load('data/logistic.npy')

X_test = normalize(X_test)
X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)
y_test = (np.dot(X_test, w) > 0.5).astype(int)
y = pd.DataFrame([[i + 1, int(y_test[i])] for i in range(len(y_test))], columns=['id', 'label'])
y.to_csv(sys.argv[6], index=False)
