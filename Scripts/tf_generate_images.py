#!/usr/bin/python


import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio

from tensorflow.python.keras.utils import np_utils

from tf_dreamUtils import *
np.random.seed(19762326)


DATA_DIR = './'
channels_coord = DATA_DIR + 'channelcoords.mat'
locs_3D = sio.loadmat(channels_coord)['locstemp']
locs_2D = map_to_2d(locs_3D)
print(locs_2D.shape)



def create_images_single(image_size):
	"""
    Images created from from provided dataset. Each image assigned the label of the trial it comes from.
    :param image_size: Size of the images to created.
	"""
    # Load raw data 
    filename = DATA_DIR + '20sec_raw_data_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    
    # Centerize raw data for reference electrode
    # Last electrode value is always 0 and values of others written according to last
    # Raw mean subtracted from raw
    rawdata_normalized = []
    for trial in rawdata:
        n_trial =  centerize_reference(trial)
        rawdata_normalized.append(n_trial)

    rawdata = np.array(rawdata_normalized)   
    
    # Data is splitted train, validation and test sets   
    x_tr, y_tr, x_va, y_va, x_te, y_te = tt_split(rawdata, label, ratio=0.9)
     
    print('Balancing the dataset with ')
    
    lab0 = np.squeeze(np.argwhere(y_tr == 0))
    lab1 = np.squeeze(np.argwhere(y_tr == 1))
    lab2 = np.squeeze(np.argwhere(y_tr == 2))
    np.random.shuffle(lab1)
    np.random.shuffle(lab2)
    
    # Half of DEWR and half of DE are icluded. All NE included. 
    indices = np.concatenate((lab0,lab1[:int(len(lab1)*0.5)],lab2[:int(len(lab2)*0.5)]), axis=0)
    
    x_tr_new = x_tr[indices]
    y_tr_new = y_tr[indices]
    
   
    unique, counts = np.unique(y_tr_new , return_counts=True)
    print('Label distribution in the train set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_va , return_counts=True)
    print('Label distribution in the validation set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_te , return_counts=True)
    print('Label distribution in the test set ', np.asarray((unique, counts)).T)
    
    #Create train matrix  
    X_train = np.concatenate(x_tr_new, axis = 0)
    y_train = np.concatenate([lab(y, x_tr_new[0].shape[0]) for y in y_tr_new])
    
    #Create validation matrix
    X_valid = np.concatenate(x_va, axis=0)
    y_valid= np.concatenate( [lab(y, x_va[0].shape[0]) for y in y_va]  )

    #Create test matrix
    X_test = np.concatenate(x_te, axis=0)
    y_test = np.concatenate( [lab(y, x_te[0].shape[0]) for y in y_te]  )
    
    train_images = gen_images(locs_2D, X_train, image_size, normalize=True)
    valid_images = gen_images(locs_2D, X_valid, image_size, normalize = True)
    test_images = gen_images(locs_2D, X_test, image_size, normalize=True)
    
    fileName =  str(image_size)+'_'+str(image_size)+'_last20sec_img'
    np.savez_compressed(fileName, train_img=train_images, train_labels=y_train, test_img=test_images, test_labels=y_test, valid_img=valid_images, valid_labels = y_valid )
    
    
    
def create_images_multi(image_size):
	"""
    Images created from from provided dataset. Each image assigned the label of the trial it comes from.
    Multichannel images consists of two channels: the first comprises delta power, second comprises beta-gamma power 
    :param image_size: Size of the images to created.
	"""

    # Load fft data 
    filename = DATA_DIR +'2sec_fft_data_SW_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    
    # centerize raw data for reference electrode
    rawdata_normalized = []
    for trial in rawdata:
        n_trial =  centerize_reference(trial)
        rawdata_normalized.append(n_trial)

    rawdata = np.array(rawdata_normalized)   
    
    
    # Data is splitted train, validation and test sets  
    x_tr, y_tr, x_va, y_va, x_te, y_te = tt_split(rawdata, label, ratio=0.9)
    
    
    print('Balancing the dataset with ')
    
    lab0 = np.squeeze(np.argwhere(y_tr == 0))
    lab1 = np.squeeze(np.argwhere(y_tr == 1))
    lab2 = np.squeeze(np.argwhere(y_tr == 2))
    np.random.shuffle(lab1)
    np.random.shuffle(lab2)

     # Half of DEWR and half of DE are icluded. All NE included.
    indices = np.concatenate((lab0,lab1[:int(len(lab1)*0.5)],lab2[:int(len(lab2)*0.5)]), axis=0)
    
    x_tr_new = x_tr[indices]
    y_tr_new = y_tr[indices]
    
   
    unique, counts = np.unique(y_tr_new , return_counts=True)
    print('Label distribution in the train set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_va , return_counts=True)
    print('Label distribution in the validation set ', np.asarray((unique, counts)).T)

    unique, counts = np.unique(y_te , return_counts=True)
    print('Label distribution in the test set ', np.asarray((unique, counts)).T)

    
    #Create train matrix  
    X_train = np.concatenate(x_tr_new, axis = 0)
    y_train = np.concatenate([lab(y, x_tr_new[0].shape[0]) for y in y_tr_new])
   
    #Create validation matrix
    X_valid = np.concatenate(x_va, axis=0)
    y_valid= np.concatenate( [lab(y, x_va[0].shape[0]) for y in y_va]  )

    #Create test matrix
    X_test = np.concatenate(x_te, axis=0)
    y_test = np.concatenate( [lab(y, x_te[0].shape[0]) for y in y_te]  )
    
    train_images = gen_images(locs_2D, X_train, image_size, normalize=True)
    valid_images = gen_images(locs_2D, X_valid, image_size, normalize = True)
    test_images = gen_images(locs_2D, X_test, image_size, normalize=True)
      
    
    fileName =  str(image_size)+'_'+str(image_size)+'_multichannel_img'
    np.savez_compressed(fileName, train_img=train_images, train_labels=y_train, test_img=test_images, test_labels=y_test, valid_img=valid_images, valid_labels = y_valid )
   
    

def main():
    image_size = 32
    try:
        image_size = int(sys.argv[1])
    except Exception  as e:
        print('Please identify image size; default 32')

    print ('Start interpolate images of ',image_size, 'x', image_size, 'x', 1)    
    create_images_single(image_size)
    print ('Start interpolate images of ',image_size, 'x', image_size, 'x', 2)    
    create_images_multi(image_size)
    print('All done! Images are created and dumped.')  
        

if __name__ == '__main__':
      main()      
    
    
    
    
    
        
        
        
        
