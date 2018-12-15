import matplotlib.pyplot as plt

import PIL
import numpy

from skimage.feature import hog
from skimage import data, exposure

import glob
import code
import json
import sys
import cv2

feature_list = []

if len(sys.argv) < 2:
    print('Pass a folder containing \'Folio Leaf Dataset\'')

folder = sys.argv[1]

leafs_dir = '{}/Folio Leaf Dataset/Folio/*/*.jpg'.format(folder)
leafs_paths = glob.glob(leafs_dir)

leafs_dir = '{}/Folio Leaf Dataset/Folio/*/*.JPG'.format(folder)
leafs_paths += glob.glob(leafs_dir)

leafs_paths.sort()

images_num = len(numpy.asarray(leafs_paths))
last_folder = ''

label_map = {}
j = -1

for i, lp in enumerate(leafs_paths):
    print("{0} out of {1}".format(i, images_num))

    curr_folder = lp.split('/')[-2]

    if curr_folder != last_folder:
        j += 1
        label_map[curr_folder] = j

    image = numpy.asarray(PIL.Image.open(lp))
    image = cv2.resize(image, (64, 128), cv2.INTER_LINEAR)

    fd = hog(image, orientations=4, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2')
    fd = numpy.append([j], fd)

    feature_list.append(fd)

    last_folder = curr_folder

feature_list = numpy.asarray(feature_list)

with open("label_map.json", "w") as f:
    json.dump(label_map, f)

output_fn = "data/leafs.csv"
numpy.savetxt(output_fn, feature_list, '%10.10f', delimiter=',')

print("Data saved to {}".format(output_fn))
