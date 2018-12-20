#!/usr/bin/python


import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from dreamUtils import *
from dreamNetworks import *


def lab(y, sample_size):
    return [y]*sample_size

def load_data():
    
    DATA_DIR = './data/'
    channels_coord = DATA_DIR + 'channelcoords.mat'
    files = os.listdir(DATA_DIR)
    files.remove('channelcoords.mat')
    
    files_labels = {'file': ['JL02trial_8.mat',
                       'JL02trial_9.mat','JL02trial_10.mat','JL02trial_11.mat','JL02trial_12.mat', 
                       'JL02trial_13.mat','JL02trial_14.mat','JL02trial_15.mat','JL02trial_16.mat',
                       'JL02trial_17.mat'],'label': [1,2,2,0,2,2,2,1,2,1]}

    labels_df =  pd.DataFrame.from_dict(files_labels).set_index('file')
    locs_3D = sio.loadmat(channels_coord)['locstemp']
    locs_2D = map_to_2d(locs_3D)
    
    
    all_data = []
    all_data_labels = []

    for single_trial in labels_df.index:
        label = labels_df.get_value(single_trial, 'label')
        #take last 20 seconds and 256 electrodes
        #referecence electrode is excluded
        a_trial = ((sio.loadmat(DATA_DIR + single_trial)['a_trial']).T)[-10000:,0:256]
        all_data.append(a_trial)
        all_data_labels.append(label)
        
        
    # assign each image the label of trial
    # here, 10000 image is taken from each trial is known
    labels= np.concatenate( [lab(y, 10000) for y in all_data_labels]  )
    
    # create an image per time step
    data = np.concatenate([gen_images(locs_2D, X , 32, normalize=False,
                augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False) for X in all_data])
        
    X_train, X_test, y_train, y_test = train_test_split(data, labels,test_size=0.2, random_state=1453)
    
    return X_train, X_test, y_train, y_test
        

def run_model(X_train, X_test, y_train, y_test):
    model = CNN_Image32()
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Make labels categorical
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    #correct input shape before feeding to CNN
    X_train_32 = X_train.reshape(-1,32,32,1)
    X_test_32 = X_test.reshape(-1,32,32,1)
    
    # train the model
    model.fit(X_train_32, y_train, validation_data=(X_test_32, y_test), epochs=10)


if __name__ == '__main__':
    
    print('Houston, we are ready for departure')

    X_train, X_test, y_train, y_test = load_data()
    run_model(X_train, X_test, y_train, y_test)
    
    
    
    
    
    
        
        
        
        