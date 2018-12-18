#!/usr/bin/python

from __future__ import print_function

import numpy
from skimage.feature import hog

import glob
import json
import sys
import cv2
import os


def make_all_leafs_paths(folder):
    leafs_dir = '{}/Folio Leaf Dataset/Folio/*/*.jpg'.format(folder)
    leafs_paths = glob.glob(leafs_dir)

    leafs_dir = '{}/Folio Leaf Dataset/Folio/*/*.JPG'.format(folder)
    leafs_paths += glob.glob(leafs_dir)

    '''make sure labels will match features'''
    leafs_paths.sort()

    return leafs_paths


def create_features_and_labels(leafs_paths):
    j = -1
    last_folder = ''

    label_map = {}
    feature_list = []
    label_list = []

    images_num = len(numpy.asarray(leafs_paths))
    for i, lp in enumerate(leafs_paths):

        '''print logs in one line'''
        print('\r', end='')
        print('{0} out of {1}'.format(i, images_num), end='')
        sys.stdout.flush()

        curr_folder = lp.split('/')[-2]

        if curr_folder != last_folder:
            j += 1

            '''map a labels to leaf species'''
            label_map[curr_folder] = j

        image = cv2.imread(lp)

        '''resize image to 64x128 pixels'''
        image = cv2.resize(image, (64, 128))

        '''extract HOG features vector'''
        fd = hog(
            image,
            block_norm='L2-Hys'
        )

        feature_list.append(fd)
        label_list.append(j)

        last_folder = curr_folder

    print('')

    '''convert lists to numpy arrays'''
    feature_list = numpy.asarray(feature_list)
    label_list = numpy.asarray(label_list)

    save_data(feature_list, label_list, label_map)


def save_data(feature_list, label_list, label_map):
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f)

    if not os.path.isdir('data'):
        if os.path.exists('data'):
            print('Error. File named \'data\' exists.')
            sys.exit(1)
        os.mkdir('data', 0o600)

    label_filename = 'data/labels.npy'
    numpy.save(label_filename, label_list)

    feature_filename = 'data/features.npy'
    numpy.save(feature_filename, feature_list)

    print('Data saved to {0} and {1}'.format(feature_filename, label_filename))


def main():
    if len(sys.argv) < 2:
        print('Pass a folder containing \'Folio Leaf Dataset\'')

    folder = sys.argv[1]

    leafs_paths = make_all_leafs_paths(folder)
    create_features_and_labels(leafs_paths)


if __name__ == '__main__':
    main()
