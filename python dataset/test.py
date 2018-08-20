import os
import pymysql
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.parsers import read_csv


tf.set_random_seed(777)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) -np.min(data, 0)
    return numerator / (denominator +1e-7)

model = tf.global_variables_initializer();


data = read_csv('priceWeather2010~2017test.csv', sep=',')

print(data)
xy = np.array(data,dtype=np.float32)

# xy= MinMaxScaler(xy) #Normalize

x_data = xy[:, 1:-1]

y_data = xy[:, [-1]]

print(x_data)
print(y_data)

# placeholder
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(4001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Hypothesis에 값을 대입하여 추측

data1 = read_csv('priceWeather2016~2017.csv', sep=',')

xy1 = np.array(data1,dtype=np.float32)
x_data1 = xy1[:, 1:-1]
dict = sess.run(hypothesis, feed_dict={X: x_data})
dict1 = sess.run(hypothesis, feed_dict={X: x_data1})

plt.plot(dict)
plt.plot(dict1)
plt.show()

a = float(input())
b = float(input())
c = float(input())
d = float(input())
j = ((a,b,c,d),(1,1,1,1))
m=np.array(j, dtype=np.float32)

x_tdata = m[0:4]
print(x_tdata)
dict3 = sess.run(hypothesis, feed_dict={X: x_tdata})
print(dict3)


#
# db = pymysql.connect(
#         host='localhost',
#         port=3306,
#         user='root',
#         passwd='1234',
#         db='test',
#         charset='utf8',autocommit=True)
# # db = pymysql.connect(
# #         host='jisub3054.cafe24.com',
# #         port=3306,
# #         user='jisub3054',
# #         passwd='1q2w3e4r..',
# #         db='jisub3054',
# #         charset='utf8')
#
# import codecs
#
# conn = db.cursor()
# #string_array  =  str ( dict . flatten () . tolist ()) [ 1 : - 1 ]
#
# df = pd.DataFrame(dict)
# df.to_csv("file_test.csv")#엑셀파일로 저장
#
# data = open('file_test.csv')
# reader = csv.reader(data)
# col1=[]
# col2=[]
#
# for line in reader:
#     col1.append(line[0])
#     col2.append(line[1])
#
# def insert_func():
#     for i in range(1,len(col2)):
#         print(col2[i])
#         sql = "INSERT INTO test(price) VALUES(%s)"
#
#         conn.execute(sql,(col2[i]))
#         db.commit()
#
#
# insert_func()
#
# db.close()