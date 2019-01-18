#!/usr/bin/python


import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio
from tensorflow.python.keras.utils import np_utils

from tf_dreamUtils import *


DATA_DIR = './'
channels_coord = DATA_DIR + 'channelcoords.mat'
locs_3D = sio.loadmat(channels_coord)['locstemp']
locs_2D = map_to_2d(locs_3D)
print(locs_2D.shape)



def create_videos_single(video_size, image_size):
    """
    Images created from from provided dataset. Each image assigned the label of the trial it comes from.
    :param image_size: Size of the images to created.
    :param video_size: How many frames(images) will be in video 
    """

    # Load raw data 
    filename = DATA_DIR + '20sec_raw_data_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    
    # centerize raw data for reference electrode
    rawdata_normalized = []
    for trial in rawdata:
        #centerized
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
    
    indices = np.concatenate((lab0,lab1[:int(len(lab1)*0.5)],lab2[:int(len(lab2)*0.5)]), axis=0)
    
    x_tr_new = x_tr[indices]
    y_tr_new = y_tr[indices]

    unique, counts = np.unique(y_tr_new , return_counts=True)
    print('Label distribution in the train set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_va , return_counts=True)
    print('Label distribution in the validation set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_te , return_counts=True)
    print('Label distribution in the test set ', np.asarray((unique, counts)).T)

    VIDEO_SIZE = video_size
    SLIDE = 10

    y_train_list = []
    x_train_list = []
    
    #Clip the video of 100 frames for train data
    for ind, trial in enumerate(x_tr_new):
        num_video = (trial.shape[0]-VIDEO_SIZE-SLIDE)//VIDEO_SIZE
        start = 0
        x_tr_video=[]
        for i in range(num_video):
            a_video = trial[start: start+VIDEO_SIZE, :]
            x_tr_video.append(a_video)
            start += SLIDE
        
        value = np.array(x_tr_video)
        x_train_list.append(value)
        y_train_list.append(lab(y_tr_new[ind], value.shape[0]))

        
    #Create train matrix  
    X_train = np.concatenate(x_train_list, axis = 0)
    y_train = np.concatenate(y_train_list, axis = 0)
    
    video_images = [gen_images(locs_2D, x, image_size, normalize=True) for x in X_train]
    X_train_video_images = np.array(video_images)
    
  
    
    VIDEO_SIZE = video_size
    SLIDE = 10
    x_test_list = []
    y_test_list = []

    #Clip the video of <video_size> frames for test data
    #Default is 100
    for ind, trial in enumerate(x_te):
        num_video = (trial.shape[0]-VIDEO_SIZE-SLIDE)//VIDEO_SIZE
        start = 0
        x_te_video=[]
        for i in range(num_video):
            a_video = trial[start: start+VIDEO_SIZE, :]
            x_te_video.append(a_video)
            start += SLIDE
    
        value = np.array(x_te_video)
        x_test_list.append(value)
        y_test_list.append(lab(y_te[ind], value.shape[0]))
    
    X_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)  
    
    video_images = [gen_images(locs_2D, x, image_size, normalize=True) for x in X_test]
    X_test_video_images = np.array(video_images)
    

    VIDEO_SIZE = video_size
    SLIDE = 10
    x_valid_list = []
    y_valid_list = []

    #Clip the video of <video_size> frames for validation data
    #Default is 100
    for ind, trial in enumerate(x_va):
        num_video = (trial.shape[0]-VIDEO_SIZE-SLIDE)//VIDEO_SIZE
        start = 0
        x_va_video=[]
        for i in range(num_video):
            a_video = trial[start: start+VIDEO_SIZE, :]
            x_va_video.append(a_video)
            start += SLIDE
    
        value = np.array(x_va_video)
        x_valid_list.append(value)
        y_valid_list.append(lab(y_va[ind], value.shape[0]))
    
    X_valid = np.concatenate(x_valid_list, axis=0)
    y_valid = np.concatenate(y_valid_list, axis=0)  
    
    video_images = [gen_images(locs_2D, x, image_size, normalize=True) for x in X_valid]
    X_valid_video_images = np.array(video_images)


    fileName =  str(image_size)+'_'+str(image_size)+'_last20sec_videos'
    np.savez_compressed(fileName, train_video=X_train_video_images, train_labels=y_train, test_video=X_test_video_images, test_labels=y_test, valid_video= X_valid_video_images, valid_labels = y_valid)
    
    
    
def create_videos_multi(video_size, image_size):
    """
    Images created from from provided dataset. Each image assigned the label of the trial it comes from.
    :param image_size: Size of the images to created.
    :param video_size: How many frames(images) will be in video 
    """

    # Load fft data 
    filename = DATA_DIR +'2sec_fft_data_SW_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    # Data is splitted train, validation and test sets   
    x_tr, y_tr, x_va, y_va, x_te, y_te = tt_split(rawdata, label, ratio=0.9)


    print('Balancing the dataset with ')
    
    lab0 = np.squeeze(np.argwhere(y_tr == 0))
    lab1 = np.squeeze(np.argwhere(y_tr == 1))
    lab2 = np.squeeze(np.argwhere(y_tr == 2))
    np.random.shuffle(lab1)
    np.random.shuffle(lab2)
    
    indices = np.concatenate((lab0,lab1[:int(len(lab1)*0.5)],lab2[:int(len(lab2)*0.5)]), axis=0)
    
    x_tr_new = x_tr[indices]
    y_tr_new = y_tr[indices]
    
    unique, counts = np.unique(y_tr_new , return_counts=True)
    print('Label distribution in the train set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_va , return_counts=True)
    print('Label distribution in the validation set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_te , return_counts=True)
    print('Label distribution in the test set ', np.asarray((unique, counts)).T)

    
    VIDEO_SIZE = video_size
    SLIDE = 10

    y_train_list = []
    x_train_list = []
    
    #Clip the video of 100 frames for train data
    for ind, trial in enumerate(x_tr_new):
        num_video = (trial.shape[0]-VIDEO_SIZE-SLIDE)//VIDEO_SIZE
        start = 0
        x_tr_video=[]
        for i in range(num_video):
            a_video = trial[start: start+VIDEO_SIZE, :]
            x_tr_video.append(a_video)
            start += SLIDE
        
        value = np.array(x_tr_video)
        x_train_list.append(value)
        y_train_list.append(lab(y_tr_new[ind], value.shape[0]))

        
    #Create train matrix  
    X_train = np.concatenate(x_train_list, axis = 0)
    y_train = np.concatenate(y_train_list, axis = 0)
    
    video_images = [gen_images(locs_2D, x, image_size, normalize=True) for x in X_train]
    X_train_video_images = np.array(video_images)
    
  
    
    VIDEO_SIZE = video_size
    SLIDE = 10
    x_test_list = []
    y_test_list = []

    #Clip the video of <video_size> frames for test data
    #Default is 100
    for ind, trial in enumerate(x_te):
        num_video = (trial.shape[0]-VIDEO_SIZE-SLIDE)//VIDEO_SIZE
        start = 0
        x_te_video=[]
        for i in range(num_video):
            a_video = trial[start: start+VIDEO_SIZE, :]
            x_te_video.append(a_video)
            start += SLIDE
    
        value = np.array(x_te_video)
        x_test_list.append(value)
        y_test_list.append(lab(y_te[ind], value.shape[0]))
    
    X_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)  
    
    video_images = [gen_images(locs_2D, x, image_size, normalize=True) for x in X_test]
    X_test_video_images = np.array(video_images)
    

    VIDEO_SIZE = video_size
    SLIDE = 10
    x_valid_list = []
    y_valid_list = []

    #Clip the video of <video_size> frames for validation data
    #Default is 100
    for ind, trial in enumerate(x_va):
        num_video = (trial.shape[0]-VIDEO_SIZE-SLIDE)//VIDEO_SIZE
        start = 0
        x_va_video=[]
        for i in range(num_video):
            a_video = trial[start: start+VIDEO_SIZE, :]
            x_va_video.append(a_video)
            start += SLIDE
    
        value = np.array(x_va_video)
        x_valid_list.append(value)
        y_valid_list.append(lab(y_va[ind], value.shape[0]))
    
    X_valid = np.concatenate(x_valid_list, axis=0)
    y_valid = np.concatenate(y_valid_list, axis=0)  
    
    video_images = [gen_images(locs_2D, x, image_size, normalize=True) for x in X_valid]
    X_valid_video_images = np.array(video_images)
    
    fileName =  str(image_size)+'_'+str(image_size)+'_multichannel_videos'
    np.savez_compressed(fileName, train_video=X_train_video_images, train_labels=y_train, test_video=X_test_video_images, test_labels=y_test, valid_video= X_valid_video_images, valid_labels = y_valid)
   
    

def main():
    image_size = 32
    video_size_single=100
    video_size_multi = 10
    try:
        image_size = int(sys.argv[1])
    except Exception  as e:
        print('Identify image size. Video size is 100 frame for single channeş, 10 frame for multichannel videos')

    print ('Start interpolate videos of ',video_size_single, 'x', image_size, 'x', image_size)    
    create_videos_single(video_size_single, image_size)
    print ('Start interpolate videos of ',video_size_multi, 'x', image_size, 'x', image_size, 'x',2 )
    create_videos_multi(video_size_multi, image_size)
    print('All done!')  
       

if __name__ == '__main__':
      main()      
    
    
    
    
    
        
        
        
        
