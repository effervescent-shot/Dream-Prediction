import math as m
import numpy as np
np.random.seed(1234)

import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, generic_utils

from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras import initializers
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.preprocessing.image import ImageDataGenerator
#import cv2 as cv
from dreamUtils import *

print('DreamNetworks is ready!')

def Baseline_NN(input_dimenson = 256):
    """ 
   input_dimension = number of features
   
   2 hidden layer NN. All layers are Dense
   128-32-8 filters, 3 class
   kernel_initializer = normal
   activation = sigmoid
   
   loss = categorical_crossentropy
   optimizer = adam
   metrics = ['accuracy']
   
    """
    model = Sequential()
    # first layer
    model.add(Dense(128,   # or 100
                    input_dim=input_dimenson, 
                    kernel_initializer='normal',   # 'normal', initializers.Constant(value=0), ...
    #                 kernel_regularizer=regularizers.l2(0.01),  # smooth filters, but bad accuracy
                    activation='sigmoid'))  # 'relu', 'sigmoid', 'tanh', ...
    # second layer
    model.add(Dense(32, 
                    kernel_initializer='normal',   # 'normal', ...
    #                 kernel_regularizer=regularizers.l2(0.1),  # smooth filters, but bad accuracy
                    activation='sigmoid'))
    # third layer
    model.add(Dense(8, 
                    kernel_initializer='normal',   # 'normal', ...
    #                 kernel_regularizer=regularizers.l2(0.1),  # smooth filters, but bad accuracy
                    activation='sigmoid'))
    # last layer
    model.add(Dense(3, 
                    kernel_initializer='normal',   # 'normal', ...
    #                 kernel_regularizer=regularizers.l2(0.1),  # smooth filters, but bad accuracy
                    activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', # 'adam', optimizers.SGD(lr=0.1), ...
                  metrics=['accuracy'])
    
    return model

def Simple_CNN_Image16():
    """ 
   Expecting 16x16x1 image data as input
   
   Conv2D<64> - Conv2D<32> -  Dense<3>
   kernel_initializer = None
   activation = relu
   kernel_size = 3
   padding = valid(default)
   
   loss = categorical_crossentropy
   optimizer = adam
   metrics = ['accuracy']
   
    """
    
    # create model
    model = Sequential()
    # add layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # fully connected
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

def CNN_Image32():
    """ 
   Expecting 32x32x1 image data as input
   
   Conv2D<32> - Conv2D<32> - Conv2D<32> - Conv2D<32> - MaxPool2D<2,2> - 
   Conv2D<64> - Conv2D<64> - MaxPool2D<2,2> - 
   Conv2D<128> -  MaxPool2D<2,2> -
   Dense<512> - Dense<3>
   
   
   kernel_initializer = glorot_uniform
   activation = None
   kernel_size = 3
   padding = same for convolution, valid for pooling layers
   
   loss = categorical_crossentropy
   optimizer = adam
   metrics = ['accuracy']
   
    """
    # create model
    model = Sequential()
    # add layers
    # activation is NONE  
    #model.add(Conv2D(32, kernel_size=3, activation='relu'))

    model.add(Conv2D(32, kernel_size=3, input_shape=(32,32,1), kernel_initializer='glorot_uniform'))
    model.add(Conv2D(32, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(32, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(32, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))
    # after max-pooling
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))
    # another max-pooling
    model.add(Conv2D(128, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))
    # fully connected layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
    
def CNN_Multichannel_Image32(num_color_chan = 2):
    """ 
   Expecting 32x32x2 image data as input
   num_color_chan := number of color channel/frequency bands
   
   Conv2D<32> - Conv2D<32> - Conv2D<32> - Conv2D<32> - MaxPool2D<2,2> - 
   Conv2D<64> - Conv2D<64> - MaxPool2D<2,2> - 
   Conv2D<128> -  MaxPool2D<2,2> -
   Dense<512> - Dense<3>
   
   
   kernel_initializer = glorot_uniform
   activation = None
   kernel_size = 3
   padding = same for convolution, valid for pooling layers
   
   loss = categorical_crossentropy
   optimizer = adam
   metrics = ['accuracy']
   
    """    
    # create model
    model = Sequential()
    # add layers
    # activation is NONE  
    #model.add(Conv2D(32, kernel_size=3, activation='relu'))

    model.add(Conv2D(32, kernel_size=3, input_shape=(32,32,num_color_chan), 
                     kernel_initializer='glorot_uniform'))
    
    model.add(Conv2D(32, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(32, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(32, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))
    # after max-pooling
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))
    # another max-pooling
    model.add(Conv2D(128, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))
    # fully connected layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
    
def CNN_Video32(time_slot = 10):
    """ 
   Expecting Tx32x32x1 image data as input
   T = time_slot
   
   Conv3D<32> - Conv3D<32> - Conv3D<32> - Conv3D<32> - MaxPool3D<2,2> - 
   Conv3D<64> - Conv3D<64> - MaxPool3D<2,2,2> - 
   Conv3D<128> -  Dense<3>
   
   
   kernel_initializer = glorot_uniform
   activation = None
   kernel_size = 3
   padding = same for convolution, valid for pooling layers
   
   loss = categorical_crossentropy
   optimizer = adam
   metrics = ['accuracy']
   
    """    
    # create model
    model = Sequential()
    # add layers
    model.add(Conv3D(32, kernel_size=3, input_shape=(time_slot,32,32,1)))
    model.add(Conv3D(32, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(Conv3D(32, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(Conv3D(32, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_last'))
    # new layer
    model.add(Conv3D(64, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(Conv3D(64, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_last'))
    # flatten and check
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(3, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def CNN_Multichannel_Video32(num_color_chan=2, time_slot=10):
    """ 
   Expecting Tx32x32x1 image data as input
   T = time_slot
   num_color_chan := number of color channel/frequency bands

   Conv3D<32> - Conv3D<32> - Conv3D<32> - Conv3D<32> - MaxPool3D<2,2,2> - 
   Conv3D<64> - Conv3D<64> - MaxPool3D<2,2,2> - 
   Conv3D<128> -  Dense<3>
   
   
   kernel_initializer = glorot_uniform
   activation = None
   kernel_size = 3
   padding = valid
   
   loss = categorical_crossentropy
   optimizer = adam
   metrics = ['accuracy']
   
    """    
    # create model
    model = Sequential()
    # add layers
    model.add(Conv3D(32, kernel_size=3, input_shape=(time_slot,32,32,num_color_chan)))
    model.add(Conv3D(32, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(Conv3D(32, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(Conv3D(32, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_last'))
    # new layer
    model.add(Conv3D(64, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(Conv3D(64, kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1), padding='same' ))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_last'))
    # flatten and check
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(3, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    

def CNN_LSTM():
    pass 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    