
import math as m
import numpy as np
np.random.seed(1234)

import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils, generic_utils

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#import cv2 as cv
from tf_dreamUtils import *

print('Dream Networks are ready!')


def LogisticRegression(nb_channels=3, k_regularizer = regularizers.l1(0.001), input_dimension = 256):
     """ 
    Neural network implemented fro Logistic Regression classification as another baseline model

    :param nb_channels: number of class
    :param act: activation function
    :param k_regularizer: kernel regularizer
    :param input_dimension: size of the input 
    """
    model = Sequential()
    # first layer
    model.add(Dense(nb_channels, input_dim=input_dimension,  kernel_initializer='glorot_uniform', activation='sigmoid', kernel_regularizer=k_regularizer)) 
    return model



def Baseline_NN(nb_channels=3, dropoutRate = 0.5, act = 'relu', k_regularizer = regularizers.l2(0.001), input_dimension = 256):
    """ 
    Fully connected dense neural network

    :param nb_channels: number of class
    :param dropoutRate: drop-out rate of last layer
    :param act: activation function
    :param k_regularizer: kernel regularizer
    :param input_dimension: size of the input 
    """
    model = Sequential()
    # first layer
    model.add(Dense(128,   # or 100
                    input_dim=input_dimension, 
                    kernel_initializer='glorot_uniform', activation='sigmoid', kernel_regularizer=k_regularizer))
                    # 'normal', initializers.Constant(value=0), ...
                    #kernel_regularizer=regularizers.l2(0.01),  # smooth filters, but bad accuracy
                    #activation='sigmoid'))  # 'relu', 'sigmoid', 'tanh', ...
    # second layer
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    # third layer
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    # fourth layer
    model.add(Dense(16, kernel_initializer='glorot_uniform',  activation=act, kernel_regularizer=k_regularizer))
    # drop-out layer to prevent overfitting
    model.add(Dropout(rate=dropoutRate))
    
    # last layer
    model.add(Dense(nb_channels ,kernel_initializer='glorot_uniform',activation='softmax'))

    
    return model

def Simple_CNN():
    """     
    Simple convolutional neural network, implemented for comparison purposes.    

   Expecting 32x32x1 image data as input
   
   Conv2D<64> - Conv2D<32> -  Dense<3>
   kernel_initializer = None
   activation = relu
   kernel_size = 3
   padding = valid(default)
   
    """
    
    # create model
    model = Sequential()
    # add concolutional layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # fully connected layer
    model.add(Flatten())
    model.add(Dropout(rate=0.35))
    model.add(Dense(3, activation='softmax'))

    
    return model
    

def CNN_Image(nb_channels=3, dropoutRate = 0.5, act='relu', k_size=3, d_layer = 512, 
	k_regularizer = regularizers.l2(0.001), img_size=32,num_color_chan = 1):
    """ 
    Deep convolutional 2D neural network with softmax classifier

    :param nb_channels: number of class
    :param dropoutRate: drop-out rate of last layer
    :param act: activation function
    :param k_size: convolutional kernel size
    :param k_regularizer: kernel regularizer
    :param d_layer: number of hidden unit in the last layer
    :param img_size: image size
    :param num_color_chan = number of color channel in the image, no RGB values used but real electrode values are used
    :param input_dimension: size of the input 
 
   Expecting 32x32x1 image data as input
   
   Conv2D<32> - Conv2D<32> - Conv2D<32> - Conv2D<32> - MaxPool2D<2,2> - 
   Conv2D<64> - Conv2D<64> - MaxPool2D<2,2> - 
   Conv2D<128> -  MaxPool2D<2,2> -
   Dense<512> - Dense<3>
   
    """
    strides = None
    print('PARAMETERS OF MODELS: ', act, ' ', k_size, ' ', d_layer, ' ', dropoutRate)

    # create model
    model = Sequential()
    # add layers
    model.add(Conv2D(32, kernel_size=k_size, input_shape=(img_size,img_size,num_color_chan), 
    	kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer  ))
    model.add(Conv2D(32, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer) )
    model.add(Conv2D(32, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer) )
    model.add(Conv2D(32, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer) )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=strides, padding='valid', data_format='channels_last'))
    #after max-pooling
    model.add(Conv2D(64, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    model.add(Conv2D(64, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=strides, padding='valid', data_format='channels_last'))
    # another max-pooling
    model.add(Conv2D(128, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=strides, padding='valid', data_format='channels_last'))
    # fully connected layer
    model.add(Flatten())
    model.add(Dense(d_layer))
    model.add(Dropout(rate=dropoutRate))
    model.add(Dense(nb_channels, activation='softmax'))
    
    return model

    
def CNN_Image_Multi(nb_channels=3, dropoutRate = 0.5, act='relu', k_size=3, d_layer = 512, 
	k_regularizer = regularizers.l2(0.001), img_size=32,num_color_chan = 2):
   """ 
    Deep convolutional 2D neural network with softmax classifier

    :param nb_channels: number of class
    :param dropoutRate: drop-out rate of last layer
    :param act: activation function
    :param k_size: convolutional kernel size
    :param k_regularizer: kernel regularizer
    :param d_layer: number of hidden unit in the last layer
    :param img_size: image size
    :param num_color_chan = number of color channel in the image, no RGB values used but delta and beta-gamma power
    values of electrodes are used
    :param input_dimension: size of the input 
 
   Expecting 32x32x2 image data as input
   
   Conv2D<32> - Conv2D<32> - Conv2D<32> - Conv2D<32> - MaxPool2D<2,2> - 
   Conv2D<64> - Conv2D<64> - MaxPool2D<2,2> - 
   Conv2D<128> -  MaxPool2D<2,2> -
   Dense<512> - Dense<3>
   
    """
    strides = None
    print('PARAMETERS OF MODELS: ', act, ' ', k_size, ' ', d_layer)
    # create model
    model = Sequential()
    # add layers
    # activation is NONE  

    model.add(Conv2D(32, kernel_size=k_size, input_shape=(img_size,img_size,num_color_chan), 
    	kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer  ))
    model.add(Conv2D(32, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer) )
    model.add(Conv2D(32, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer) )
    model.add(Conv2D(32, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer) )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,  padding='valid', data_format='channels_last'))
    # after max-pooling
    model.add(Conv2D(64, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    model.add(Conv2D(64, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,  padding='valid', data_format='channels_last'))
    #another max-pooling
    model.add(Conv2D(128, kernel_size=k_size, padding='same', kernel_initializer='glorot_uniform', activation=act, kernel_regularizer=k_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides=strides, padding='valid', data_format='channels_last'))
    # fully connected layer
    model.add(Flatten())
    model.add(Dense(d_layer))
    model.add(Dropout(rate=dropoutRate))
    model.add(Dense(nb_channels, activation='softmax'))
    
    return model

    
    
def CNN_Video(nb_channels=3, dropoutRate = 0.5, act='relu', k_size=3, d_layer = 512, 
	k_regularizer = regularizers.l1(0.001), img_size=32,time_slot = 100,num_color_chan=1):

    """ 
    Deep convolutional 3D neural network with softmax classifier

    :param nb_channels: number of class
    :param dropoutRate: drop-out rate of last layer
    :param act: activation function
    :param k_size: convolutional kernel size
    :param k_regularizer: kernel regularizer
    :param d_layer: number of hidden unit in the last layer
    :param img_size: image size
    :param time_slot: number of frames/images in a video, length of the video
    :param num_color_chan = number of color channel in the image/frame, no RGB values used real values of electrodes are used
    :param input_dimension: size of the input 
 
   Expecting 100x32x32x1 video data as input
   
   Conv3D<32> - Conv3D<32> - Conv3D<32> - Conv3D<32> - MaxPool3D<2,2,2> - 
   Conv3D<64> - Conv3D<64> - MaxPool3D<2,2,2> - 
   Dense<512> - Dense<3>
   
    """
 
    strides = None
    # In each convolutional layer, 10 consecutive images are convolved
    kernel = (10, k_size, k_size)

    print('PARAMETERS OF MODELS: ', act, ' ', k_size, ' ', d_layer)
  
    model = Sequential()
    # add layers
    model.add(Conv3D(32, kernel_size=kernel, input_shape=(time_slot,img_size,img_size,num_color_chan), activation=act))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act ))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act ))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act ))
    model.add(MaxPooling3D(pool_size=kernel, strides=strides, data_format='channels_last'))
    # new layer
    model.add(Conv3D(64, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act))
    model.add(Conv3D(64, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=strides, data_format='channels_last'))
    
    # flatten and check
    model.add(Flatten())
    model.add(Dense(d_layer))
    model.add(Dropout(rate=dropoutRate))
    model.add(Dense(nb_channels, activation='softmax'))
    
    return model


def CNN_Video_Multi(nb_channels=3, dropoutRate = 0.5, act='relu', k_size=3, d_layer = 512, 
	k_regularizer = regularizers.l1(0.001), img_size=32,time_slot = 10, num_color_chan=2):
 """ 
    Deep convolutional 3D neural network with softmax classifier

    :param nb_channels: number of class
    :param dropoutRate: drop-out rate of last layer
    :param act: activation function
    :param k_size: convolutional kernel size
    :param k_regularizer: kernel regularizer
    :param d_layer: number of hidden unit in the last layer
    :param img_size: image size
    :param time_slot: number of frames/images in a video, length of the video
    :param num_color_chan = number of color channel in the image/frame, no RGB values used but delta and beta-gamma power
    values of electrodes are used
   Expecting 100x32x32x1 video data as input
   
   Conv3D<32> - Conv3D<32> - Conv3D<32> - Conv3D<32> - MaxPool3D<2,2,2> - 
   Conv3D<64> - Conv3D<64> - MaxPool3D<2,2,2> - 
   Dense<512> - Dense<3>
   
    """
    strides = None
    kernel = (2, k_size, k_size)
    print('PARAMETERS OF MODELS: ', act, ' ', k_size, ' ', d_layer)
     
    model = Sequential()
    # add layers
    model.add(Conv3D(32, kernel_size=kernel, input_shape=(time_slot,img_size,img_size,num_color_chan), activation=act))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act ))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act ))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act ))
    model.add(MaxPooling3D(pool_size=(2,2,1), strides=strides, data_format='channels_last'))
    # new layer
    model.add(Conv3D(64, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act))
    model.add(Conv3D(64, kernel_size=kernel, padding='same', kernel_initializer='glorot_uniform', activation=act))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=strides, data_format='channels_last'))
    
    # flatten and check
    model.add(Flatten())
    model.add(Dense(d_layer))
    model.add(Dropout(rate=dropoutRate))
    model.add(Dense(nb_channels, activation='softmax'))
    
    
    return model

    


def LSTM(nb_channels=3, img_size=32,time_slot = 10,num_color_chan=1):
    """
    Recurrent Neural Network for image data
    Never tested.
    :param nb_channels: number of class
    :param img_size: image size
    :param time_slot: number of frames/images in a video, length of the video
    :param num_color_chan = number of color channel in the image/frame
    """
    model = Sequential()
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(time_slot, input_shape=(img_size, img_size, num_color_chan), return_sequences=True, activation='sigmoid'))

    # flatten and check
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(rate=0.5))
    model.add(Dense(nb_channels, activation='softmax'))
    
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
