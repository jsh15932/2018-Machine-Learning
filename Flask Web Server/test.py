# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import os
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.parsers import read_csv

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':

        tf.set_random_seed(777)
        def MinMaxScaler(data):
            numerator = data - np.min(data, 0)
            denominator = np.max(data, 0) -np.min(data, 0)
            return numerator / (denominator +1e-7)

        model = tf.global_variables_initializer();

        data = read_csv('priceWeather2010~2017test.csv', sep=',')

        xy = np.array(data,dtype=np.float32)

        x_data = xy[:, 1:-1]

        y_data = xy[:, [-1]]

        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([4,1]), name="weight")
        b = tf.Variable(tf.random_normal([1]), name="bias")

        hypothesis = tf.matmul(X, W) + b

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        train = optimizer.minimize(cost)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        for step in range(4001):
            cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

        data1 = read_csv('priceWeather2016~2017.csv', sep=',')

        xy1 = np.array(data1,dtype=np.float32)
        x_data1 = xy1[:, 1:-1]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
        dict1 = sess.run(hypothesis, feed_dict={X: x_data1})

        avg_temp = float(request.form['avg_temp'])
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rain_fall = float(request.form['rain_fall'])

        j = ((avg_temp,min_temp,max_temp,rain_fall ),(1,1,1,1))
        m=np.array(j, dtype=np.float32)

        x_tdata = m[0:4]
        dict3 = sess.run(hypothesis, feed_dict={X: x_tdata})

        price = dict3[0]
        return render_template('index.html', price=price)

if __name__ == '__main__':
   app.run(debug = True)