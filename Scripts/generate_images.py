#!/usr/bin/python


import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

from dreamUtils import *
np.random.seed(1453)


DATA_DIR = './'
channels_coord = DATA_DIR + 'channelcoords.mat'
locs_3D = sio.loadmat(channels_coord)['locstemp']
locs_2D = map_to_2d(locs_3D)
print(locs_2D.shape)



def create_images_single(image_size, sec):
    # Load raw data 
    filename = DATA_DIR + str(sec) + 'sec_raw_data_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    
    # centerize raw data for reference electrode
    rawdata_normalized = []
    scaler = MinMaxScaler(feature_range=(-1,1))
    for trial in rawdata:
        #centerized
        n_trial =  centerize_reference(trial)
        #min-max scale
        #scaler.fit(n_trial)
        #mm_trial = scaler.transform(n_trial)
        rawdata_normalized.append(n_trial)

    rawdata = np.array(rawdata_normalized)   
    
    
    x_tr, x_te, y_tr, y_te = tt_split(rawdata, label, ratio=0.9)
    
    
    # To balance the data set, sample from surplus labels
    y_train_list = []
    x_train_list = []
    for ind, value in enumerate(x_tr):
        if y_tr[ind] == 2:
            selected = value[np.random.randint(value.shape[0], size=int(value.shape[0] * 0.35)), :]
            x_train_list.append(selected)
            y_train_list.append(lab(2, selected.shape[0]))

        elif y_tr[ind] == 1:
            selected = value[np.random.randint(value.shape[0], size=int(value.shape[0] * 0.45)), :]
            x_train_list.append(selected)
            y_train_list.append(lab(1, selected.shape[0]))
        else :
            x_train_list.append(value)
            y_train_list.append(lab(0, value.shape[0]))
    
    
    
    #Create train matrix  
    X_train = np.concatenate(x_train_list, axis = 0)
    y_train = np.concatenate(y_train_list, axis = 0)
    
    #Create test matrix
    X_test = np.concatenate(x_te, axis=0)
    y_test = np.concatenate( [lab(y, x_te[0].shape[0]) for y in y_te]  )
    
    # encode labels as one-hot vectors
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    train_images = gen_images(locs_2D, X_train, image_size)
    test_images = gen_images(locs_2D, X_test, image_size)
    
    fileName =  str(image_size)+'_'+str(image_size)+'_last'+str(sec)+'sec_img'
    np.savez_compressed(fileName, train_img=train_images, train_labels=y_train, test_img=test_images, test_labels=y_test)
    
    
    
def create_images_multi(image_size):
    # Load fft data 
    filename = DATA_DIR +'fft_data_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    x_tr, x_te, y_tr, y_te = tt_split(rawdata, label, ratio=0.9)
    
    # To balance the data set, sample from surplus labels
    y_train_list = []
    x_train_list = []
    for ind, value in enumerate(x_tr):
        if y_tr[ind] == 2:
            selected = value[np.random.randint(value.shape[0], size=int(value.shape[0] * 0.33)), :]
            x_train_list.append(selected)
            y_train_list.append(lab(2, selected.shape[0]))

        elif y_tr[ind] == 1:
            selected = value[np.random.randint(value.shape[0], size=int(value.shape[0] * 0.42)), :]
            x_train_list.append(selected)
            y_train_list.append(lab(1, selected.shape[0]))
        else :
            x_train_list.append(value)
            y_train_list.append(lab(0, value.shape[0]))
    
    
    
    #Create train matrix  
    X_train = np.concatenate(x_train_list, axis = 0)
    y_train = np.concatenate(y_train_list, axis = 0)
    
    #Create test matrix
    X_test = np.concatenate(x_te, axis=0)
    y_test = np.concatenate( [lab(y, x_te[0].shape[0]) for y in y_te]  )
    
    # encode labels as one-hot vectors
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    train_images = gen_images(locs_2D, X_train, image_size)
    test_images = gen_images(locs_2D, X_test, image_size)
    
    fileName =  str(image_size)+'_'+str(image_size)+'_multichannel_img'
    np.savez_compressed(fileName, train_img=train_images, train_labels=y_train, test_img=test_images, test_labels=y_test)
   
    

def main():
    image_size = 32
    sec = 20
    try:
        image_size = int(sys.argv[1])
        sec = int(sys.argv[2])
    except Exception  as e:
        print('Identify image size and second')

        
    print ('Start interpolate images of ', image_size, 'x', image_size)    
    create_images_single(image_size, sec )
    create_images_multi(image_size)
    print('All done!')  
        
# Command line args are in sys.argv[1], sys.argv[2] ..
# sys.argv[0] is the script name itself and can be ignored

# Standard boilerplate to call the main() function to begin
# the program.

if __name__ == '__main__':
      main()      
    
    
    
    
    
        
        
        
        