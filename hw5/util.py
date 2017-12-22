import csv
import pandas as pd
import numpy as np


def readData(path):
    _list = []

    with open(path, newline='', encoding='latin-1') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            _list.append(row)
            '''
            if len(row) != 1:
                row[1] = row[1][1:]
                s = '::'.join(row)
                _list.append(s)
            else:
                _list.append(row)
            '''
    return _list[1:]


def readTrain(path):
    data = pd.read_csv(path)
    _users = data['UserID'].values
    _movies = data['MovieID'].values
    _rating = data['Rating'].values
    _n_users = data['UserID'].drop_duplicates().max()
    _n_movies = data['MovieID'].drop_duplicates().max()
    return _users, _movies, _rating, _n_users, _n_movies


def readTest(path):
    data = pd.read_csv(path)
    _users = data['UserID'].values
    _movies = data['MovieID'].values
    return _users, _movies


def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X, mean, std


def cut(Y):
    for i in range(Y.shape[0]):
        if Y[i, 0] > 5:
            Y[i, 0] = 5
        elif Y[i, 0] < 1:
            Y[i, 0] = 1
    return Y
