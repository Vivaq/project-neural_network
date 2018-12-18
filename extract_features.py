import PIL
import numpy

from skimage.feature import hog
from skimage import data, exposure

import glob
import json
import sys
import cv2


if len(sys.argv) < 2:
    print('Pass a folder containing \'Folio Leaf Dataset\'')

folder = sys.argv[1]
ftype = 'numpy'

leafs_dir = '{}/Folio Leaf Dataset/Folio/*/*.jpg'.format(folder)
leafs_paths = glob.glob(leafs_dir)

leafs_dir = '{}/Folio Leaf Dataset/Folio/*/*.JPG'.format(folder)
leafs_paths += glob.glob(leafs_dir)

leafs_paths.sort()

images_num = len(numpy.asarray(leafs_paths))
last_folder = ''

label_map = {}
j = -1

feature_list = []
label_list = []

for i, lp in enumerate(leafs_paths):
    print("{0} out of {1}".format(i, images_num))

    curr_folder = lp.split('/')[-2]

    if curr_folder != last_folder:
        j += 1
        label_map[curr_folder] = j

    image = cv2.imread(lp)
    image = cv2.resize(image, (64, 128))

    fd = hog(
        image,
        block_norm='L2-Hys'
    )

    feature_list.append(fd)
    label_list.append(j)

    last_folder = curr_folder

feature_list = numpy.asarray(feature_list)
label_list = numpy.asarray(label_list)

with open("label_map.json", "w") as f:
    json.dump(label_map, f)

if ftype == 'numpy':
    feature_fn = "data/labels.npy"
    numpy.save(feature_fn, label_list)

    label_fn = "data/features.npy"
    numpy.save(label_fn, feature_list)

elif ftype == 'pandas':
    output_fn = "data/leafs.csv"
    numpy.savetxt(output_fn, feature_list, '%10.10f', delimiter=',')

print("Data saved to folder 'data' as {}".format(ftype))
