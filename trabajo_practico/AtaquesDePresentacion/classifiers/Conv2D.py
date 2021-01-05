from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import train
from sklearn.metrics import confusion_matrix


import numpy as np
import os



def createModel(dim0, dim1, num_classes):
    x = Input(shape=(dim0, dim1, 3))

    h = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    h = MaxPooling2D((2, 2))(h)

    z = Flatten()(h)
    z = Dense(64, activation='relu')(z)
    z = Dense(32, activation='relu')(z)
    y = Dense(num_classes, activation='sigmoid')(z)


    cnn = Model(x, y)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(cnn.summary())
    return cnn

def saveWeights(model, epoch):
    checkpoint_path = "training_1/weights.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.save_weights(checkpoint_path.format(epoch=epoch))

def loadWeights(model):
    checkpoint_path = "training_1/weights.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    lastest = train.latest_checkpoint(checkpoint_dir)
    model.load_weights(lastest)
    return model

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
    checkpoint_path = "./training_1/weights.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print('********************* LOADING MODEL *********************')
    lastest = train.latest_checkpoint(checkpoint_dir)
    model.load_weights(lastest)
    print('********************* MODEL LODADED *********************')
    return model



def parsePredictData(yhat, groundTruth):
    yv_1hot_aux = np.zeros(len(yhat))
    yhat_aux = np.zeros(len(yhat))
    for i in range(0, len(yhat)):
        print(np.argmax(yhat[i,:]) ,'<-- yhat', np.argmax(groundTruth[i,:]), 'ground truth')
        yv_1hot_aux[i] = np.argmax(groundTruth[i,:])
        yhat_aux[i] = np.argmax(yhat[i,:])

    y_prob = np.amax(yhat, axis=1)

    yv_1hot = yv_1hot_aux
    yhat = yhat_aux
    return yhat, yv_1hot, y_prob

def calculateMetrics (yhat, groundTruth):
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


def Conv2Dclassification(images, tags):
    num_classes = 2


    N_train,dim0,dim1,canals = images.shape

    cnn = createModel(dim0, dim1, num_classes)

    images = images.reshape((N_train,dim0,dim1,3))

    y_1hot = to_categorical(tags, num_classes=num_classes)


    cnn = loadModel(cnn)


    plotshat = []
    plotsValid = []
    plotProb = []
    yhat = cnn.predict(images)
    yhat, y_1hot, yhat_prob = parsePredictData(yhat, y_1hot)

    calculateMetrics(yhat, y_1hot)

    yhat_prob = (yhat_prob - np.min(yhat_prob)) / (np.max(yhat_prob) - np.min(yhat_prob))
    plotshat.append(yhat)
    plotProb.append(yhat_prob)
    plotsValid.append(y_1hot)

    return yhat_prob, yhat,  y_1hot

def train_model(x_train, y_train, x_valid, y_valid):
    checkpoint_path = "training_1/weights.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    num_classes = 2


    N_train,dim0,dim1,canals = x_train.shape

    cnn = createModel(dim0, dim1, num_classes)

    y_1hot = to_categorical(y_train, num_classes=num_classes)
    yv_1hot = to_categorical(y_valid, num_classes=num_classes)

    cp_callback = ModelCheckpoint(filepath=checkpoint_dir, monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max')
    #es_callback = EarlyStopping(monitor='val_accuracy', patience='10')
    N_epochs = 5
    batch_size = 16
    cnn.fit(x_train, y_1hot, epochs=N_epochs, batch_size=batch_size, validation_data=(x_valid, yv_1hot), callbacks=[cp_callback])

