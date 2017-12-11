#ld use a generator to load data and preprocess it on the fly, in batch size portions to feed into your Behavioral Cloning model .

import os
import csv

samples = []
with open('mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('second track/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from numpy.random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    #count = 0
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'mydata/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                if center_image is None:
                    name = 'second track/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                #image loading check
                if center_image is None:
                    print("Image loading failed")
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip( center_image, 1 ))    
                angles.append(-center_angle)
            #print(images[0])
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #count += len(images)
            #print(count)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples[1:32], batch_size=32)
validation_generator = generator(validation_samples[1:32], batch_size=32)

# All required packages
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import Model
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt

# To get sample image
name = 'mydata/IMG/'+samples[1][0].split('\\')[-1]
image = cv2.imread(name)
crop_top = 50
crop_bottom = 20
shape = (image.shape[0]-(crop_top+crop_bottom), image.shape[1], image.shape[2])
print(shape)

#get lenet model
def getLenetModel():
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=image.shape))
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=shape))
    model.add(Convolution2D(6, 5, 5,border_mode='valid', activation='elu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5,border_mode='valid', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5,border_mode='valid', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

#get nvidia model
def getNvidiaModel():
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=image.shape))
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=shape))
    model.add(Convolution2D(24, 5, 5,border_mode='valid',subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5,border_mode='valid', subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48, 3, 3,border_mode='valid', subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3,border_mode='valid', subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2048, W_regularizer=l2(0.001)))
    model.add(Activation('elu'))
    model.add(Dense(256, W_regularizer=l2(0.001)))
    model.add(Activation('elu'))
    model.add(Dense(64, W_regularizer=l2(0.001)))
    model.add(Activation('elu'))
    model.add(Dense(16, W_regularizer=l2(0.001)))
    model.add(Activation('elu'))
    model.add(Dense(1))
    return model

def runModel(model, filename):    
    history_object = model.fit_generator(train_generator, samples_per_epoch =
        len(train_samples), validation_data = 
        validation_generator,
        nb_val_samples = len(validation_samples), 
        nb_epoch=1, verbose=1)

    # Save model data
    model.save(filename+'.h5')
    json_string = model.to_json()
    with open(filename +'.json', 'w') as f:
        f.write(json_string)

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# run lenet model
model1 = getLenetModel()
model1.compile(loss='mse', optimizer='adam')
print("Running lenet model")
runModel(model1, 'lmodel')

# run nvidia model
model = getNvidiaModel()
model.compile(loss='mse', optimizer='adam')
print("Running nvidia model")
runModel(model, 'nmodel')