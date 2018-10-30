#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/23 15:33
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : mobile_data.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

FEATURE_COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)])


def parse_csv_by_pandas(file_name, yname='price', frac=0.7, seed=None):
    if not os.path.exists(file_name):
        file_name = tf.keras.utils.get_file(file_name, DATASET_URL)

    datasets = pd.read_csv(file_name, names=FEATURE_COLUMN_TYPES.keys(),
                          dtype=FEATURE_COLUMN_TYPES, na_values='?')

    datasets = datasets.dropna()

    if seed is None:
        seed = np.random.seed()
    train_datasets = datasets.sample(frac=0.5, random_state=seed)
    test_datasets = datasets.drop(train_datasets.index)

    train_x, train_y = train_datasets, train_datasets.pop(yname)
    test_x, test_y = test_datasets, test_datasets.pop(yname)

    return (train_x, train_y), (test_x, test_y)


def input_train_func(features, label, batch_size=100):
    train_samples = (dict(features), label)
    train_tensors = tf.data.Dataset.from_tensor_slices(train_samples)
    train_tensors = train_tensors.shuffle(buffer_size=100)
    train_tensors = train_tensors.repeat()
    train_tensors = train_tensors.batch(batch_size)
    return train_tensors


def input_eval_func(features, labels, batch_size=32):
    if labels is None:
        feature = dict(features)
    else:
        feature = (dict(features), labels)
    datasets = tf.data.Dataset.from_tensor_slices(feature)
    datasets = datasets.batch(batch_size)
    return datasets
