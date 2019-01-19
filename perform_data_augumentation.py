from os import listdir
import fnmatch
import os
import keras
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

directories = []
filesPaths = []

def getImagePaths():
    for directory in listdir('FolioLeafDataset/Folio/train'):
        directories.append('FolioLeafDataset/Folio/train/' + directory)

def getFilesPaths():
    for dir in directories:
        for item in (fnmatch.filter(os.listdir(dir), '*.jpg')):
            path = dir + '/' + item
            filesPaths.append(path)

def performAugumentation():
    gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                                       horizontal_flip=True, shear_range=0.15, channel_shift_range=15)

    for iterator, imagePath in enumerate(filesPaths):
        print("first iterator" + str(iterator))
        image = np.expand_dims(ndimage.imread(imagePath), 0)
        augIter = gen.flow(image)
        augImages = [next(augIter)[0].astype(np.uint8) for i in range (1)]
        for secondIterator, augImage in enumerate(augImages):
            print("second iterator" + str(secondIterator))
            imagePath = imagePath.rsplit('.', 1)[0]
            misc.imsave(imagePath + 'Augumented' + str(iterator) + str(secondIterator) + '.jpg', augImage)


getImagePaths()
getFilesPaths()
performAugumentation()
