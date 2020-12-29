from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import train
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import RMSprop
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join, isdir
import re

checkpoint_path = "training_1/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def createModel(dim0, dim1, num_classes):
    # - Input layer
    x = Input(shape=(dim0, dim1, 3))

    # - Convolution+Pooling layers
    h = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    h = MaxPooling2D((2, 2))(h)
    #h = Conv2D(64, (5, 5), activation='relu', padding='same')(h)
    #h = MaxPooling2D((2, 2))(h)
    #h = Conv2D(16, (5, 5), activation='relu', padding='same')(h)
    #z = MaxPooling2D((2, 2))(h)

    # - Classification header
    z = Flatten()(h)
    #z = Dense(128, activation='relu')(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(32, activation='relu')(z)
    y = Dense(num_classes, activation='sigmoid')(z)

    # - Put all in a model and compile
    cnn = Model(x, y)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(cnn.summary())
    return cnn

def saveWeights(model, epoch):
    model.save_weights(checkpoint_path.format(epoch=epoch))

def loadWeights(model):
    lastest = train.latest_checkpoint(checkpoint_dir)
    model.load_weights(lastest)
    return model

def findRecursive(path):
    images = []
    tags = np.array([])
    for file in listdir(path):
        if isfile(path + '/' + file):
            image = cv2.resize(cv2.imread(path +'/'+ file), (300, 300))
            if re.findall('attack*', os.path.basename(path)):
                #images = np.append(images, image)
                images.append(image)
                tags = np.append(tags, 1)
            else:
                #images = np.append(images, image)
                images.append(image)
                tags = np.append(tags, 0)
        elif isdir(path + '/' + file):
            image_rec, tag_rec = findRecursive(path+'/'+file)
            #images = np.append(images, image_rec)
            #images.append(image_rec)
            images = images + image_rec
            tags = np.append(tags, tag_rec)
    '''images = np.asarray(images)
    images = images.astype(np.float32)'''
    tags = np.asarray(tags)
    tags = tags.astype(np.float32)

    return images, tags

def stratifiedData(images, tags):
    size = images.shape[0]
    x_train = images[:int(size*0.8)]
    y_train = tags[:int(size*0.8)]
    x_test = images[int(size*0.8):]
    y_test = tags[int(size*0.8):]

    return x_train, y_train, x_test, y_test
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def loadModel(model):
    print('********************* LOADING MODEL *********************')
    lastest = train.latest_checkpoint(checkpoint_dir)
    model.load_weights(lastest)
    print('********************* MODEL LODADED *********************')
    return model

def APCER(yhat, groundTruth, umbral):
    yhat = np.where(yhat<umbral, 0, 1)
    Npais = len(groundTruth)
    RES = np.sum(yhat)

    return 1 - (1 / Npais) * RES

def BPCER(yhat, groundTruth, umbral):
    yhat = np.where(yhat<umbral, 0, 1)
    NBF = len(groundTruth[groundTruth == 0])
    RES = np.sum(yhat)

    return RES / NBF

def showDETCurve(yhat, groundTruth, fig, ax):
    umbrales = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    APCER_list = []
    BPCER_list = []
    for umbral in umbrales:
        APCER_list.append(APCER(yhat, groundTruth, umbral))
        BPCER_list.append(BPCER(yhat, groundTruth, umbral))
    ax.plot(APCER_list, BPCER_list)

    #ax.set_xlim([min(BPCER_list), max(BPCER_list)])
    #ax.set_ylim([min(APCER_list), max(APCER_list)])

    return fig, ax
    #ax[i].set_title('line plot with data points')

    #ax.plot(xdata2, ydata2, color='tab:orange')

def parsePredictData(yhat, groundTruth):
    yv_1hot_aux = np.zeros(len(yhat))
    yhat_aux = np.zeros(len(yhat))
    for i in range(0, len(yhat)):
        print(np.argmax(yhat[i,:]) ,'<-- yhat', np.argmax(groundTruth[i,:]), 'ground truth')
        yv_1hot_aux[i] = np.argmax(groundTruth[i,:])
        yhat_aux[i] = np.argmax(yhat[i,:])

    yv_1hot = yv_1hot_aux
    yhat = yhat_aux
    return yhat, yv_1hot

def calculateMetrics (yhat, groundTruth):
    k = 4
    hits_list = []
    count = 0
    conf_mat = confusion_matrix(groundTruth, yhat)
    print(conf_mat)
    hits = conf_mat[0, 0]+conf_mat[1, 1]
    fails = conf_mat[0, 1]+conf_mat[1, 0]
    hits_list.append(hits)
    precision = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    strlog = "Fold %d: HITS = %d, FAILS = %d, PRECISION = %f" %(count, hits, fails, precision)
    print(strlog)




#******************* MAIN ***********************


execution_type = 'test'
num_classes = 2

print('********************* LOADING DATASET *********************')
images, tags = findRecursive('D:/Descargas/REAL_SENSE_segmented/REAL_SENSE/REAL_SENSE_IR/')
images = np.asarray(images)
images = images.astype(np.float32)
images, tags = shuffle_in_unison(images, tags)
x_train, y_train, x_test, y_test = stratifiedData(images, tags)
x_test, y_test, x_valid, y_valid = stratifiedData(x_test, y_test)
print('********************* DATASET LOADED  *********************')


#--- Get info of train and test data sets
N_train,dim0,dim1,canals = x_train.shape
N_test,dim0,dim1, canals = x_test.shape
N_valid,dim0,dim1, canals = x_valid.shape

num_pixels = dim0*dim1

cnn = createModel(dim0, dim1, num_classes)

x_tensor = x_train.reshape((N_train,dim0,dim1,3))



y_1hot = to_categorical(y_train, num_classes=num_classes)
yt_1hot = to_categorical(y_test, num_classes=num_classes)
yv_1hot = to_categorical(y_valid, num_classes=num_classes)

checkpoint_path = "training_1/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)


cnn = loadModel(cnn)

if execution_type == 'train':
    cp_callback = ModelCheckpoint(filepath=checkpoint_dir, monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max')
    #es_callback = EarlyStopping(monitor='val_accuracy', patience='10')
    N_epochs = 5
    batch_size = 16
    cnn.fit(x_tensor, y_1hot, epochs=N_epochs, batch_size=batch_size, validation_data=(x_test, yt_1hot), callbacks=[cp_callback])



plotshat = []
plotsValid = []
yhat = cnn.predict(x_valid)
yhat_prob = yhat
yhat, yv_1hot = parsePredictData(yhat, yv_1hot)

calculateMetrics(yhat, yv_1hot)

yhat_prob = (yhat_prob - np.min(yhat_prob)) / (np.max(yhat_prob) - np.min(yhat_prob))
plotshat.append(yhat_prob)
plotsValid.append(yv_1hot)

yhat = cnn.predict(x_test)
yhat_prob = yhat
yhat, yv_1hot = parsePredictData(yhat, yt_1hot)

calculateMetrics(yhat, yv_1hot)
yhat_prob = (yhat_prob - np.min(yhat_prob)) / (np.max(yhat_prob) - np.min(yhat_prob))
plotshat.append(yhat_prob)
plotsValid.append(yv_1hot)





fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(0, len(plotshat)):
    fig, ax = showDETCurve(plotshat[i], plotsValid[i], fig, ax)
    #ax.append(ax_aux)
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
#plt.legend(loc='best')
# display the plot
plt.show()