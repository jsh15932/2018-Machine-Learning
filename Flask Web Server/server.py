# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import datetime

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        avg_temp = int(request.form['avg_temp'])
        min_temp = int(request.form['min_temp'])
        max_temp = int(request.form['max_temp'])
        rain_fall = int(request.form['rain_fall'])

        price = 1000
        return render_template('index.html', price=price)

if __name__ == '__main__':
   app.run(debug = True)