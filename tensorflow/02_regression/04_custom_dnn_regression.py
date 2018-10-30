#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/27 16:54
# @Author  : viekie
# @Site   :  www.ml2ai.com
# @File    : 04_custom_dnn_regression.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse

import tensorflow as tf

import utils.mobile_data as datautils

parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--factor', type=int, default=1000)
args = parser.parse_args()


def create_feature_columns():
    vocabulary_list = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_column = tf.feature_column.categorical_column_with_vocabulary_list(key='body-style',
                                                                            vocabulary_list=vocabulary_list)
    make_column = tf.feature_column.categorical_column_with_hash_bucket(key='make', hash_bucket_size=50)

    feature_columns = [
        tf.feature_column.numeric_column('curb-weight'),
        tf.feature_column.numeric_column('highway-mpg'),
        tf.feature_column.indicator_column(body_column),
        tf.feature_column.embedding_column(make_column, 3)
    ]

    return feature_columns


def build_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units, tf.nn.relu)
    net = tf.layers.dense(net, units=1, activation=None)

    predictions = tf.squeeze(net, 1)
    predictions = tf.cast(predictions, tf.float64)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.mean_squared_error(labels, predictions)
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels, predictions)
        metrics = tf.metrics.mean_squared_error(labels, predictions)
        metrics_op = {'rmse': metrics}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    (train_x, train_y), (test_x, test_y) = datautils.parse_csv_by_pandas(os.path.abspath('.') +
                                                                         '\\datasets\\imports-85.data')
    train_y /= args.factor
    test_y /= args.factor

    feature_columns = create_feature_columns()
    params = {
        'hidden_units': [20, 20],
        'feature_columns': feature_columns
    }

    classifier = tf.estimator.Estimator(model_fn=build_model, params=params)
    classifier.train(input_fn=lambda: datautils.input_train_func(train_x, train_y, args.batch_size),
                     steps=args.epoches)
    eval_result = classifier.evaluate(input_fn=lambda: datautils.input_eval_func(test_x, test_y))

    print("\n" + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}"
          .format(args.factor * eval_result["rmse"]))