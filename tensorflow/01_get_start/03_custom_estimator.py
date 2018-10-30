#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/25 18:08
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 03_custom_estimator.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import tensorflow as tf

import utils.iris_utils as dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--epoches', default=1000, type=int, help='epoches')
args = parser.parse_args()

def build_model(mode, features, labels, params):

    net = tf.feature_column.input_layer(features=features, feature_columns=params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, units=params['n_classes'])

    predict_class = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = opt.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy  = tf.metrics.accuracy(labels=labels, predictions=predict_class)
        metric = {'accuracy': accuracy}
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = {
            'class_ids': predict_class[:, tf.newaxis],
            'probabilies': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=prediction)

def main():
    (train_x, train_y), (test_x, test_y) = dataloader.parse_csv_by_pandas()

    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    params = {
        'hidden_units': [10, 10],
        'feature_columns': feature_columns,
        'n_classes': 3
    }

    classifier = tf.estimator.Estimator(model_fn=build_model, params=params)
    classifier.train(input_fn=lambda: dataloader.train_input_func(features=train_x,
                                                                  labels=train_y,
                                                                  batch_size=args.batch_size),
                     steps=args.epoches)

    eval_result = classifier.evaluate(input_fn=lambda: dataloader.eval_input_func(features=test_x,
                                                                               labels=test_y))
    print('eval accuracy {accuracy:0.3f}'.format(**eval_result))
    x, y = dataloader.make_data()
    predictions = classifier.predict(input_fn=lambda: dataloader.eval_input_func(features=x, labels=None, batch_size=1))

    for pred, y_ in zip(predictions, y):
        template = 'Prediction is "{}" ({:.1f}%), expected "{}"'
        class_ids = pred['class_ids'][0]
        probability = pred['probabilies'][class_ids]
        print(template.format(dataloader.SPECIES_NAMES[class_ids],
                              100 * probability,
                              y_))
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

