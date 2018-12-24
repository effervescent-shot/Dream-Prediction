import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.utils import np_utils

from dreamUtils import *
from dreamNetworks import *

def run_CNN_32Image():

    loaded = np.load('./32_32_last20sec_img.npz')
    X_train = loaded['train_img']
    y_train = loaded['train_labels']

    X_test = loaded['test_img']
    y_test = loaded['test_labels']

    X_train = np.array( [np.rollaxis(x,0,3) for x in X_train ] )
    X_test = np.array( [np.rollaxis(x,0,3) for x in X_test ] )


    print('Train matrix: ', X_train.shape, y_train.shape)
    print('Test matrix: ', X_test.shape, y_test.shape)
    
    print('Single channel 32x32 images --- simple')
    model = Simple_CNN_Image32()
    model.fit(X_train, y_train, epochs=100, batch_size=1000, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
    print("CNN_Image32 Error: %.2f%%" % (100-scores[1]*100))

    
    
    print('Single channel 32x32 images --- Arch D')
    model = CNN_Image32()
    model.fit(X_train, y_train, epochs=100, batch_size=1000, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
    print("Simple_CNN_Image32() Error: %.2f%%" % (100-scores[1]*100))

def run_CNN_32ImageMulti():


    loaded_fft = np.load('./32_32_multichannel_img.npz')

    X_train_fft = loaded_fft['train_img']
    y_train_fft = loaded_fft['train_labels']

    X_test_fft = loaded_fft['test_img']
    y_test_fft = loaded_fft['test_labels']

    X_train_fft = np.array( [np.rollaxis(x,0,3) for x in X_train_fft ] )
    X_test_fft = np.array( [np.rollaxis(x,0,3) for x in X_test_fft ] )

    print('Train matrix: ',X_train_fft.shape, y_train_fft.shape)
    print('Test matrix: ',X_test_fft.shape, y_test_fft.shape)

    
    print('Multi channel 32x32 images')
    model = CNN_Multichannel_Image32()
    model.fit(X_train_fft, y_train_fft, epochs=1000, batch_size=100, verbose=2)
    scores = model.evaluate(X_test_fft, y_test_fft, verbose=2)
    print(scores)
    print("CNN_Multichannel_Image32 Error: %.2f%%" % (100-scores[1]*100))

    
def main():
    
    run_CNN_32Image()
    run_CNN_32ImageMulti()

        
if __name__ == '__main__':
    main()      