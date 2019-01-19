import tensorflow as tf
from tensorflow import keras
from os import listdir

classes = []
def getNamesOfClasses():
    for directory in listdir('FolioLeafDataset/Folio/test'):
        classes.append(directory)

def runCNN():
    getNamesOfClasses()
    trainData = 'FolioLeafDataset/Folio/train'
    testData = 'FolioLeafDataset/Folio/test'

    trainGenerator = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(trainData, target_size=(224,224), classes=classes, batch_size=15)

    testGenerator = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(testData, target_size=(224, 224), classes=classes, batch_size=15)

    model_vgg16 = keras.applications.vgg16.VGG16(weights='imagenet')
    model = tf.keras.models.Sequential()

    for layer in model_vgg16.layers:
        model.add(layer)

    model.pop()

    for layer in model.layers:
        layer.trainable = False

    output_layer = tf.keras.layers.Dense(
        32,
        activation='softmax'
    )
    model.add(output_layer)

    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    predictedValues = []
    trueValues = []
    testLabels = []

    batch_index = 0

    while batch_index <= 10:
        testImgs, labels = next(testGenerator)
        for value in labels:
            testLabels.append(value)
        batch_index += 1

    model.fit_generator(trainGenerator, steps_per_epoch=4, epochs=10, verbose=2)
    predictions = model.predict_generator(testGenerator, verbose=2, steps=11)


    print(testLabels)
    print(predictions)

    print(len(testLabels))
    print(len(predictions))

    for p in predictions:
        predictedValues.append(p.argmax())

    for value in testLabels:
           trueValues.append(value.argmax())

    print(trueValues)
    print(predictedValues)

    compareValuesOfLists(predictedValues, trueValues)



def compareValuesOfLists(list1, list2):
    correctPredictions = 0
    for i in range(0, len(list1)):
        if(list1[i] == list2[i]):
            correctPredictions += 1
    print(correctPredictions)

runCNN()

