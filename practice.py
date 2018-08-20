import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv

# 데이터 불러오기
data_frame = pd.read_excel("testweather.xlsx", sheet = 1)

# 최고기온, 최저기온 변수 제거

# 하이퍼 파라미터 설정
timesteps = seq_length = 6
data_dim = 4
hidden_dim = 4
output_dim = 1
learing_rate = 0.0005
iterations = 800

#데이터 조절

# Framework 제작
x = data_frame.values
y = data_frame["avgPrice"].values

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = np.copy(x[i:i + seq_length + 1])
    _x[timesteps-2][data_dim-1] = 0
    _x[timesteps-1][data_dim-1] = 0
    _x[timesteps][data_dim-1] = 0
    _y = [y[i + seq_length]]
    dataX.append(_x)
    dataY.append(_y)

#학습데이터와 테스트데이터 분류

train_size = int(len(dataY) * 0.85)
test_size = len(dataY) - train_size

trainX = np.array(dataX[:train_size])
testX = np.array(dataX[train_size : ])

trainY = np.array(dataY[:train_size])
testY = np.array(dataY[train_size : ])


#LSTM모델 구축

X = tf.placeholder(tf.float32, [None, seq_length+1, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    return cell


cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(5)], state_is_tuple=True)


outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred - Y))
train = tf.train.RMSPropOptimizer(learing_rate).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(iterations):
    _  , cost = sess.run([train ,loss], feed_dict={X: trainX, Y: trainY})
    if (i+1) % (iterations/10) == 0:
        print("[step: {}] loss: {}".format(i+1, cost))


#예측값 불러오기
train_predict = sess.run(Y_pred, feed_dict={X: trainX})
test_predict = sess.run(Y_pred, feed_dict={X: testX})

plt.plot(np.array(train_predict)[:,0])
plt.plot(np.array(test_predict)[:, 0])
plt.show()