from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import sys

X_train = pd.read_csv(sys.argv[3]).drop('fnlwgt', axis=1).values.astype(np.float64)
y_train = pd.read_csv(sys.argv[4]).values
X_test = pd.read_csv(sys.argv[5]).drop('fnlwgt', axis=1).values.astype(np.float64)

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

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
clf = AdaBoostClassifier(n_estimators=5000)  # <- 0.87334
# clf = AdaBoostClassifier(n_estimators=10000)

clf.fit(X_train, y_train.flatten())

joblib.dump(clf, 'data/adaboost')
'''
clf = joblib.load('data/ada_2')

if valid:
    X_valid = scaler.transform(X_valid)
    print(accuracy_score(y_train, clf.predict(X_train)))
    print(accuracy_score(y_valid, clf.predict(X_valid)))

y_test = clf.predict(X_test)

y = pd.DataFrame([[i + 1, y_test[i]] for i in range(len(y_test))], columns=['id', 'label'])
y.to_csv(sys.argv[6], index=False)
