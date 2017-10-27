import pandas as pd
import numpy as np
import sys
# from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X_train = pd.read_csv(sys.argv[3]).drop('fnlwgt', axis=1).values.astype(np.float64)
y_train = pd.read_csv(sys.argv[4]).values
X_test = pd.read_csv(sys.argv[5]).drop('fnlwgt', axis=1).values.astype(np.float64)
'''
valid = False
v = 2
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

# 1 for > 50, 2 for < 50
X1 = False
X2 = False
for i in range(X_train.shape[0]):
    if y_train[i] == 1:
        if not X1:
            X_1 = X_train[i, :].reshape((1, -1))
            X1 = True
        else:
            X_1 = np.concatenate((X_1, X_train[i, :].reshape((1, -1))), axis=0)
    else:
        if not X2:
            X_2 = X_train[i, :].reshape((1, -1))
            X2 = True
        else:
            X_2 = np.concatenate((X_2, X_train[i, :].reshape((1, -1))), axis=0)
    if i % 1000 == 0:
        print(i)

mu_1 = np.mean(X_1, axis=0)
mu_2 = np.mean(X_2, axis=0)
sigma_1 = np.cov(X_1.T)
sigma_2 = np.cov(X_2.T)
N_1 = X_1.shape[0]
N_2 = X_2.shape[0]
sigma = N_1 / y_train.shape[0] * sigma_1 + N_2 / y_train.shape[0] * sigma_2
sigma_inv = np.linalg.pinv(sigma)
w = np.dot((mu_1 - mu_2), sigma_inv)
b = -np.dot(np.dot(mu_1, sigma_inv), mu_1.T) / 2 + np.dot(np.dot(mu_2, sigma_inv), mu_2.T) / 2 + np.log(N_1 / N_2)

z = np.dot(X_train, w) + b
ans = sigmoid(z)
print('Train : {}'.format(accuracy_score(y_train, (ans > 0.5).astype(int))))

if valid:
    print('valid : {}'.format(accuracy_score(y_valid, sigmoid((np.dot(X_valid, w) + b) > 0.5).astype(int))))

np.save('data/generative_1.npy', w)
np.save('data/generative_2.npy', b)
'''
w = np.load('data/generative_1.npy')
b = np.load('data/generative_2.npy')

y_test = (sigmoid(np.dot(X_test, w) + b) > 0.5).astype(int)
y = pd.DataFrame([[i + 1, int(y_test[i])] for i in range(len(y_test))], columns=['id', 'label'])
y.to_csv(sys.argv[6], index=False)
