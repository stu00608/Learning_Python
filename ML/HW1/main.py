import sys
import pandas as pd
import numpy as np
import math
import csv

data = pd.read_csv('./train.csv',encoding='big5')

# Preprocessing

data = data.iloc[:,3:]
data[data=='NR'] = 0
raw_data = data.to_numpy()

# Extract Feature

# 18個feature 480個hours
# 一個月共有20天的資料

month_data = {}
for month in range(12):
    sample = np.empty([18,480])
    for day in range(20):
        sample[ : , day*24:(day+1)*24 ] = raw_data[ 18*(20*month+day):18*(20*month+day+1) , : ]
    month_data[month] = sample

# 預測第10個小時的資料,所以一筆x共有9個小時,480小時當中就有471筆資料

x = np.empty([12*471,18*9], dtype= float)
y = np.empty([12*471,1], dtype= float)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour>14:
                continue
            x[month*471+day*20+hour,:] = month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1)
            y[month*471+day*20+hour,0] = month_data[month][9,day*24+hour+9]

# Normalize

mean_x = np.mean(x,axis=0)
std_x = np.std(x,axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if(std_x[j]!=0):
            x[i][j] = (x[i][j]-mean_x[j])/std_x[j]

# 拆分training set and validation set (80%)

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

# Training
# using Root Mean Square Error

dim = 18 * 9 + 1                                                            #因為常數項的存在，所以 dimension (dim) 需要多加一欄
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001                                                          #避免 adagrad 的分母為 0 而加的極小數值
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    # print(w)
np.save('weight.npy', w)

# Testing

testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

# Submitting

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)