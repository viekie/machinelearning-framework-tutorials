#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/29 13:58
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 00_simple_linear_regression.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.1)
args = parser.parse_args()

x = np.float32(np.random.rand(100, 2))
y = np.dot(x, [0.1, 0.2]) + 0.3
print(y.shape)
W = tf.Variable(tf.random_normal(shape=(2, 1), mean=0.0, dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=(1, )))

y_ = tf.squeeze(tf.matmul(x, W) + b)

print(y_.shape)

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.AdagradOptimizer(learning_rate=args.lr)
linear_regressor = optimizer.minimize(loss=loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(args.epoches):
        sess.run(linear_regressor)

        if epoch % 10 == 0:
            print('epoch :{}, W:{}, b:{}, loss:{}'.format(epoch, sess.run(W), sess.run(b), sess.run(loss)))
