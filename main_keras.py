from exceptions import Exception

import numpy as np
import tensorflow as tf


def shuffle_arrays(arr1, arr2):
    indexes = np.random.permutation(len(arr1))
    return arr1[indexes], arr2[indexes]


if __name__ == '__main__':
    features = np.load('data/features.npy')
    labels = np.load('data/labels.npy')

    if len(features) != len(labels):
        raise Exception('Error. Features and labels have different lengths!')

    labels, features = shuffle_arrays(labels, features)

    data_len = len(labels)
    train_offset = int(data_len * 1.0 / 2)
    val_offset = int(data_len * 1.0 / 4) + train_offset

    train_labels = labels[:train_offset]
    train_features = features[:train_offset]

    val_labels = labels[train_offset:val_offset]
    val_features = features[train_offset:val_offset]

    test_labels = labels[val_offset:]
    test_features = features[val_offset:]

    model = tf.keras.models.Sequential()

    inputs = 6804
    classes = 32

    input_layer = tf.keras.layers.Dense(100, activation=tf.nn.relu)
    model.add(input_layer)

    hidden_layers = [100, 100]

    for hidden_layer in hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu))

    output_layer = tf.keras.layers.Dense(classes, activation=tf.nn.softmax)
    model.add(output_layer)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_features,
        train_labels,
        epochs=20,
        validation_data=(val_features, val_labels)
    )

    print('')
    print('Final loss: {0}, Final accuracy {1}'.format(*model.evaluate(test_features, test_labels)))
