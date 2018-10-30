#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/18 9:33
# @Author  : viekie
# @Site    : www.ml2ai.com
# @File    : iris_utils.py
# @Software: PyCharm

import os
import sys
import pandas as pd
import tensorflow as tf


data_folder = os.path.abspath('.') + '\\datasets'

train_file = 'iris_training.csv'
test_file = 'iris_test.csv'
url_prefix = 'http://download.tensorflow.org/data/'


FEATURE_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES_NAMES = ['Setosa', 'Versicolor', 'Virginica']



def _download_file(url):
    if not os.path.exists(data_folder + '\\' + os.path.basename(url)):
        file = tf.keras.utils.get_file(data_folder + '\\' + os.path.basename(url), url)
        print('file download! from {} -----> {}!'.format(url, file))
    else:
        file = data_folder + '\\' + os.path.basename(url)

    return file


#----------------------------------- parse csv by pandas -----------------------------------


def parse_csv_by_pandas(y_label='Species'):
    if not os.path.exists(data_folder + train_file):
        train_f = _download_file(url_prefix + train_file)
    if not os.path.exists((data_folder + test_file)):
        test_f = _download_file(url_prefix + test_file)

    train_samples = pd.read_csv(train_f, names=FEATURE_COLUMN_NAMES, header=0)
    train_features, train_labels = train_samples, train_samples.pop(y_label)
    test_samples = pd.read_csv(test_f, names=FEATURE_COLUMN_NAMES, header=0)
    test_features, test_labels = test_samples, test_samples.pop(y_label)

    return (train_features, train_labels), (test_features, test_labels)


def train_input_func(features, labels, batch_size=32):
    datasets = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    datasets = datasets.shuffle(1000)
    datasets = datasets.repeat()
    datasets = datasets.batch(batch_size)
    return datasets


def eval_input_func(features, labels, batch_size=32):
    if labels is None:
        feature = dict(features)
    else:
        feature = (dict(features), labels)
    datasets = tf.data.Dataset.from_tensor_slices(feature)
    datasets = datasets.batch(batch_size)
    return datasets


#--------------------------------- parse csv by tensorflow ---------------------------------

template = [[0.0], [0.0], [0.0], [0.0], [0]]
def _decode_line(line):
    sample = tf.decode_csv(line, template)
    feature = sample[:-1]
    label = sample[-1]
    return feature, label


def parse_csv_by_tensorflow(file_name, batch_size=32):
    if not os.path.exists(file_name):
        file = _download_file(url_prefix + os.path.basename(file_name), file_name)
    else:
        file = file_name
    samples = tf.data.TextLineDataset(file)
    samples = samples.skip(1)
    samples = samples.shuffle(1000)
    samples = samples.map(_decode_line)
    samples = samples.batch(batch_size=batch_size)
    return samples

#--------------------------------- parse csv by tensorflow ---------------------------------

def make_data():

    expecte_label = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    return predict_x, expecte_label
