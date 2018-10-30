#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/29 17:25
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 02-linear-regression.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.1)
args = parser.parse_args()

x_samples = np.float32(np.random.rand(100, 2))
y_samples = [np.dot(x, [0.1, 0.2]) + 0.3 for x in x_samples]


W = tf.Variable(tf.random_normal(shape=(2, 1)))
b = tf.Variable(tf.zeros(shape=(1)))


x_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 2))
y_placeholder = tf.placeholder(dtype=tf.float32, shape=(1))


y_ = tf.add(tf.matmul(x_placeholder, W), b)

loss = tf.reduce_mean(tf.square(y_placeholder - y_))
optimizer = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)
trainer = optimizer.minimize(loss=loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(args.epoches):

        for x, y in zip(x_samples, y_samples):
            result = sess.run(trainer, feed_dict={x_placeholder: np.reshape(x, (1,2)),
                                                  y_placeholder: np.reshape(y, (1))})


        if epoch % 10 == 0:
            print('epoch: {}, loss:{:.4f}, W:{}， b:{}'.format(epoch,
                                                              sess.run(loss,
                                                                       feed_dict={x_placeholder: np.reshape(x, (1,2)),
                                                                                  y_placeholder: np.reshape(y, (1))}),
                                                              W.eval(),
                                                              b.eval()))

    print('epoch: {}, loss:{:.4f}, W:{}， b:{}'.format(epoch,
                                                      sess.run(loss,
                                                               feed_dict={x_placeholder: np.reshape(x, (1,2)),
                                                                          y_placeholder: np.reshape(y, (1))}),
                                                      W.eval(),
                                                      b.eval()))