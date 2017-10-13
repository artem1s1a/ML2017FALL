import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import time

# validation = False
# valid = 300
# valid_times = 30
_lambda = 3000 # 3000 0.01 0.1

'''
data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])


n_row = 0
text = open('train.csv', 'r', encoding='big5')
row = csv.reader(text, delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row-1) % 18].append(float(r[i]))
            else:
                data[(n_row-1) % 18].append(float(0))
    n_row = n_row+1
text.close()

dim = 15

del data[10]
del data[13]
del data[13]

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(dim):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

x = np.concatenate((x, x ** 2), axis=1)
# x = np.concatenate((x, x ** 3), axis=1)
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

w = np.zeros(len(x[0]))
w = np.matmul(np.matmul(inv(np.matmul(x.transpose(), x) + np.identity(x.shape[1]) * _lambda), x.transpose()), y)

# save model
np.save('model_best.npy', w)
'''
# read model
w = np.load('model_best.npy')

test_x = []
n_row = 0
text = open(sys.argv[1], "r")
row = csv.reader(text, delimiter=",")

for r in row:
    if n_row % 18 == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row//18].append(float(r[i]))
    else:
        for i in range(2, 11):
            if n_row % 18 != 10 and n_row % 18 != 14 and n_row % 18 != 15:
                test_x[n_row//18].append(float(r[i]))
    n_row = n_row+1
text.close()

test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x, test_x**2), axis=1)
# test_x = np.concatenate((test_x, test_x**3), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w, test_x[i])
    if a < 0:
        a = 0
    ans[i].append(a)

text = open(sys.argv[2], "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
