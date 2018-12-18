#!/usr/bin/python

from __future__ import print_function

from exceptions import Exception, ValueError

import numpy as np
import tensorflow as tf


def shuffle_arrays(arr1, arr2):
    if not (type(arr1) == type(arr2) == np.ndarray):
        raise ValueError('Arrays have to be numpy.ndarray type')

    indexes = np.random.permutation(len(arr1))

    '''in numpy, multiple indexes return multiple elements'''
    return arr1[indexes], arr2[indexes]


if __name__ == '__main__':
    features = np.load('data/features.npy')
    labels = np.load('data/labels.npy')

    '''each sample have to have label'''
    if len(features) != len(labels):
        raise Exception('Error. Features and labels have different lengths!')

    '''shuffle arrays to randomize data across datasets'''
    labels, features = shuffle_arrays(labels, features)

    '''proportions for training/validation/test data is 50/25/25'''
    data_len = len(labels)

    train_offset = int(data_len * 1.0 / 2)
    val_offset = int(data_len * 1.0 / 4) + train_offset

    train_labels = labels[:train_offset]
    train_features = features[:train_offset]

    val_labels = labels[train_offset:val_offset]
    val_features = features[train_offset:val_offset]

    test_labels = labels[val_offset:]
    test_features = features[val_offset:]

    classes = 32

    '''create neural network with 4 layers'''
    model = tf.keras.models.Sequential()

    input_layer = tf.keras.layers.Dense(
        100,
        input_shape=(6804,),
        activation=tf.nn.relu
    )
    model.add(input_layer)

    hidden_layers = [100, 100]
    for hidden_layer in hidden_layers:
        model.add(tf.keras.layers.Dense(
            hidden_layer,
            activation=tf.nn.relu
        ))

    output_layer = tf.keras.layers.Dense(
        classes,
        activation=tf.nn.softmax
    )
    model.add(output_layer)

    model.compile(
        optimizer='nadam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    '''train neural network for 40 epochs'''
    model.fit(
        train_features,
        train_labels,
        epochs=20,
        validation_data=(val_features, val_labels)
    )

    print('')
    print('Final loss: {0}, Final accuracy {1}'.format(
        *model.evaluate(test_features, test_labels)
    ))
