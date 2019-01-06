from os import listdir
import random
import shutil
import os
import fnmatch

listOfDirectiories = []

for directory in listdir('FolioLeafDataset/Folio'):
    listOfDirectiories.append('FolioLeafDataset/Folio/' + directory)

filesDictionary = {}
rootPathsToCreate = []

rootPathsToCreate.append('FolioLeafDataset/Folio/train')
rootPathsToCreate.append('FolioLeafDataset/Folio/test')

subPathsToCreate = []

for item in rootPathsToCreate:
    for directory in listdir('FolioLeafDataset/Folio'):
       subPathsToCreate.append(item + '/' + directory)

print(subPathsToCreate)

for path in listOfDirectiories:
    listOfFiles = []
    for file in listdir(path):
        listOfFiles.append(path + '/' + file)
    filesDictionary[path] = listOfFiles

testFilesPaths = []
trainFilesPaths = []

for key, value in filesDictionary.items():
    trainData = random.sample(value, 15)
    testData = set(value) - set(trainData)
    testFilesPaths.append(testData)
    trainFilesPaths.append(value)

for path in rootPathsToCreate:
    os.makedirs(path)

for path in subPathsToCreate:
    os.mkdir(path)

for value in testFilesPaths:
    for item in value:
       shutil.move(item, 'FolioLeafDataset/Folio/test/' + item.rsplit('/')[2])

for dir in listOfDirectiories:
    for item in (fnmatch.filter(os.listdir(dir), '*.jpg')):
        path = dir + '/' + item
        shutil.move(path,'FolioLeafDataset/Folio/train/' + path.rsplit('/')[2])
