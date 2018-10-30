#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/23 16:03
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : 01_linear_regression.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
import tensorflow as tf

import utils.mobile_data as datautils


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--epoches', type=int, default=1000, help='epoches')
args = parser.parse_args()


def main():
    (train_x, train_y), (test_x, test_y) = \
        datautils.parse_csv_by_pandas(os.path.abspath('.' + '\\datasets\\imports-85.data'))

    train_y /= 1000
    test_y /= 1000

    feature_columns = [
        tf.feature_column.numeric_column(key='curb-weight'),
        tf.feature_column.numeric_column(key='highway-mpg'),
    ]

    regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    regressor.train(input_fn=lambda: datautils.input_train_func(train_x, train_y, args.batch_size),
                    steps=args.epoches)

    eval_result = regressor.evaluate(input_fn=lambda: datautils.input_eval_func(test_x, test_y, args.batch_size))
    avg_loss = eval_result['average_loss']
    print('avg eval loss {:.0f}'.format(avg_loss**0.5 * 1000))

    pred_x = {
      "curb-weight": np.array([2000, 3000]),
      "highway-mpg": np.array([30, 40])
    }

    pred_result = regressor.predict(input_fn=lambda: datautils.input_eval_func(pred_x, None))

    template = ("Curb weight: {: 4d}lbs, Highway: {: 0d}mpg, Prediction: ${: 9.2f}")

    for i, y_ in enumerate(pred_result):
        print(template.format(pred_x['curb-weight'][i], pred_x['highway-mpg'][i], 1000*y_['predictions'][0]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
