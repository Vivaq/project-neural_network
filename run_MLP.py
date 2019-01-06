#!/usr/bin/python

from __future__ import print_function
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

    '''each sample must have a label'''
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
    feature_len = len(features[0])

    '''create neural network with 4 layers'''
    model = tf.keras.models.Sequential()

    nodes1 = 128
    hidden_layer1 = tf.keras.layers.Dense(
        nodes1,
        input_shape=(feature_len,),
        activation=tf.nn.relu
    )
    model.add(hidden_layer1)

    nodes2 = 128
    hidden_layer2 = tf.keras.layers.Dense(
        nodes2,
        activation=tf.nn.relu
    )
    model.add(hidden_layer2)

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

    '''train neural network for 10 epochs'''
    model.fit(
        train_features,
        train_labels,
        epochs=10,
        validation_data=(val_features, val_labels)
    )

    loss, accuracy = model.evaluate(test_features, test_labels)
    print('Final accuracy: {}'.format(accuracy))
