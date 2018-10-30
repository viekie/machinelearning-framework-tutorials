#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/25 8:25
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 01_premade_estimator.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse

import tensorflow as tf

import utils.iris_utils as datautils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoches', type=int, default=200, help='epoches')
args = parser.parse_args()


def main():
    (train_x, train_y), (test_x, test_y) = datautils.parse_csv_by_pandas()

    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(hidden_units=[10, 10],
                                            feature_columns=feature_columns,
                                            n_classes=3)

    classifier.train(input_fn=lambda: datautils.train_input_func(train_x,
                                                                 train_y,
                                                                 args.batch_size),
                     steps=args.epoches)

    eval_result = classifier.evaluate(input_fn=lambda: datautils.eval_input_func(test_x,
                                                                                 test_y,
                                                                                 args.batch_size),
                                      steps=1)

    print('eval accuracy: {accuracy:0.3f}'.format(**eval_result))

    x, y = datautils.make_data()

    pred = classifier.predict(lambda: datautils.eval_input_func(x, y))

    for y_, y in zip(pred, y):
        template = 'predict {} with probability of {} , acutuall type is {}'
        class_ids = y_['class_ids']

        class_id = y_['class_ids'][0]
        probability = y_['probabilities'][class_id]

        print(template.format(datautils.SPECIES_NAMES[class_id],
                              100 * probability, y))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()