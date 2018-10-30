#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/27 11:08
# @Author  : viekie
# @Site   :  www.ml2ai.com
# @File    : 03_dnn_regression.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import tensorflow as tf

import utils.mobile_data as datautils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--epoches', type=int, default=1000, help='epoches')
parser.add_argument('--factor', type=int, default=1000, help='normalize factor')
args = parser.parse_args()


def convert_feature_columns():
    body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_features = tf.feature_column.categorical_column_with_vocabulary_list(key='body-style',
                                                                              vocabulary_list=body_style_vocab)

    make_feature = tf.feature_column.categorical_column_with_hash_bucket(key='make', hash_bucket_size=50)

    feature_columns = [
        tf.feature_column.numeric_column(key='curb-weight'),
        tf.feature_column.numeric_column(key='highway-mpg'),
        tf.feature_column.indicator_column(body_features),
        tf.feature_column.embedding_column(make_feature, dimension=3)
    ]
    return feature_columns



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    (train_x, train_y), (test_x, test_y) = datautils.parse_csv_by_pandas(
        os.path.abspath('.') + '\\datasets\\imports-85.data')

    train_y /= args.factor
    test_y /= args.factor

    regressor = tf.estimator.DNNRegressor(hidden_units=[10, 10], feature_columns=convert_feature_columns())

    regressor.train(input_fn=lambda: datautils.input_train_func(train_x, train_y, args.batch_size),
                    steps=args.epoches)

    eval_result = regressor.evaluate(input_fn=lambda: datautils.input_eval_func(features=test_x, labels=test_y))

    print('eval result: {:.0f}'.format(args.factor * eval_result['average_loss'] ** 0.5))