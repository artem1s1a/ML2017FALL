import numpy as np


def readTrain(path, label):
    with open(path, 'r', encoding='UTF-8') as f:
        data = f.read().splitlines()

    if label:
        labels = np.zeros((len(data), 1))
        lines = []

        for i in range(len(data)):
            labels[i] = data[i][0]
            lines.append(data[i][10:])

        return lines, labels

    else:
        return data


def readTest(path):
    with open(path, 'r', encoding='UTF-8') as f:
        data = f.read().splitlines()
    del data[0]

    lines = []
    for line in data:
        line = line.split(',', 1)[1]
        lines.append(line)

    return lines


def ByWord2Vector(lines, vectors, dim=256):
    _lines = []
    for line in lines:
        _line = []
        for word in line:
            if word is '_':
                _line.append(np.zeros(dim))
            elif word in vectors.wv:
                _line.append(vectors.wv[word])
            else:
                _line.append(np.zeros(dim))
        _lines.append(_line)

    return _lines


def padLines(lines, maxLen=32):
    for line in lines:
        if len(line) < maxLen:
            line.extend(['_'] * (maxLen - len(line)))

    return lines


def createDict(lines, dictSize=8000, remove=False):
    dict = {}
    if remove:
        stopwords = set('a an the to i he she for and you is are am'.split())
    else:
        stopwords = set()
    for line in lines:
        words = line.split()
        for word in words:
            if word not in stopwords:
                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 1

    list = sorted(dict, key=dict.__getitem__, reverse=True)
    #sorted(data.items(), key=lambda x: x[1])
    list = list[:dictSize]

    dict = {}
    for index, word in enumerate(list):
        dict[word] = index

    return dict


def ByDict(lines, dict):
    matrix = np.zeros((len(lines), len(dict)))

    for i, line in enumerate(lines):
        for word in line:
            if word in dict:
                matrix[i][dict[word]] += 1

    return matrix
