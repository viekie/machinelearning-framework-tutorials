#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/25 13:09
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 02_eager_estimator.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utils.iris_utils as datautils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoches', type=int, default=1000, help='epoches')
args = parser.parse_args()

tfe.enable_eager_execution()

def build_model():
    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4, )),
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(3, activation=None)
    ])
    return model

def loss(model, x, y):
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=model(x))


def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y)
    return tape.gradient(loss_value, model.variables)


def train(model, epoches, datasets, lr=0.05):

    train_loss_result = []
    train_accu_result = []

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    for i in range(epoches):
        avg_loss = tfe.metrics.Mean()
        avg_accu = tfe.metrics.Accuracy()

        for x, y in datasets:
            opt.apply_gradients(zip(grad(model, x, y), model.variables),
                                global_step=tf.train.get_or_create_global_step())
            avg_loss(loss(model, x, y))
            avg_accu(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        train_loss_result.append(avg_loss.result())
        train_accu_result.append(avg_accu.result())

    if i % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(1,
                                                                avg_loss.result(),
                                                                avg_accu.result()))


def test(model, datasets):

    avg_accu = tfe.metrics.Accuracy()
    for x, y in datasets:
        pred = tf.argmax(model(x), axis=1, output_type=tf.int32)
        avg_accu(pred, y)
    print('test accuracy is {:.3%}'.format(avg_accu.result()))


def predict(model, features):
    y_ = model(features)
    return y_

def main():
    train_file = os.path.realpath('.') + '\\datasets\\iris_training.csv'
    test_file = os.path.realpath('.') + '\\datasets\\iris_test.csv'
    train_dataset = datautils.parse_csv_by_tensorflow(file_name=train_file)
    test_dataset = datautils.parse_csv_by_tensorflow(file_name=test_file)

    class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])


    dnn_model = build_model()

    train(model=dnn_model, epoches=args.epoches, datasets=train_dataset)
    test(model=dnn_model, datasets=test_dataset)
    y_ = predict(dnn_model, predict_dataset)

    for i, logit in enumerate(y_):
        template = 'example {}, predict is {}, actual is {}'
        id = tf.argmax(logit).numpy()
        print(template.format(i, class_ids[i], y_[id]))


if __name__ == '__main__':
    main()