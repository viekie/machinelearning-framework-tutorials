#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/26 14:25
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 02_linear_regression_categorical.py
# @Software: PyCharm


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import tensorflow as tf

import utils.mobile_data as datautils


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--epoches', type=int, default=2000, help='epoches')
parser.add_argument('--factor', type=int, default=1000, help='normalize factor')

args = parser.parse_args()


def main():

    file_name = os.path.abspath('.') + '\\datasets\\imports-85.data'

    (train_x, train_y), (test_x, test_y) = datautils.parse_csv_by_pandas(file_name)
    train_y /= args.factor
    test_y /= args.factor

    body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_style_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='body-style',
            vocabulary_list=body_style_vocab)

    make_column = tf.feature_column.categorical_column_with_hash_bucket(key='make',
                                                                        hash_bucket_size=50)

    feature_columns = [
        tf.feature_column.numeric_column(key="curb-weight"),
        tf.feature_column.numeric_column(key="highway-mpg"),
        body_style_column,
        make_column
    ]

    classifier = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    classifier.train(input_fn=lambda: datautils.input_train_func(train_x, train_y, args.batch_size),
                     steps=args.epoches)

    eval_result = classifier.evaluate(input_fn=lambda: datautils.input_eval_func(test_x, test_y,
                                                                                 args.batch_size))

    print('eval accuracy: {:.0f}'.format(args.factor * eval_result['average_loss'] ** 0.5))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
