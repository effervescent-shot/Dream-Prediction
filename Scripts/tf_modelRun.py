import os, sys, ast
import math as m
import numpy as np
np.random.seed(1234)

import pandas as pd
import scipy.io as sio

import tensorflow as tf
from tensorflow.python.keras.utils import np_utils, generic_utils

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, LSTM
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tf_dreamUtils import *
from tf_dreamNetworks import *

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
np.random.seed(1235)

from tensorflow.python.keras import backend as K

def as_keras_metric(method):
    """
    Multiclass precision - recall 
    Override Keras default
    """
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)



def reformat_input(filename, model_type, sample, alter_label=False):
    """
    Read datafiles and reformat matrices according to input shape of models
    Alter 3-class to 2-class if specified, sample data set if specified
    Make target values (train_y, validation_y, test_y) categorical
    Suffle the indices then return train, validation and test sets

    :param filename: data file name
    :param model_type: model to be tested. Each model has its own dataset
    :param sample: boolean, if true train, validation and test sets are downsampled with fraction 0.2
    :param alter_label: boolean for 2-class classification, if true only NE and DE labels are included, DEWR are excluded 
    """

    if model_type == 'Baseline' or model_type == 'Logistic':
        loaded = np.load(filename)
        rawdata = loaded['data']
        label = loaded['labels']
        
        # centerize raw data for reference electrode
        rawdata_normalized = []
        for trial in rawdata:
            #centerized
            n_trial =  centerize_reference(trial)
            n_trial = normalize_through_time(n_trial)
            rawdata_normalized.append(n_trial)

        rawdata = np.array(rawdata_normalized)   
        
        # Train, validation and test split
        x_tr, y_tr, x_va, y_va, x_te, y_te = tt_split(rawdata, label, ratio=0.9)
     
        print('Balancing the dataset with ')
        
        lab0 = np.squeeze(np.argwhere(y_tr == 0))
        lab1 = np.squeeze(np.argwhere(y_tr == 1))
        lab2 = np.squeeze(np.argwhere(y_tr == 2))
        np.random.shuffle(lab1)
        np.random.shuffle(lab2)
        indices_tr = np.concatenate((lab0,lab1[:int(len(lab1)*0.5)],lab2[:int(len(lab2)*0.5)]), axis=0)
        
        if alter_label:
            indices_tr = np.concatenate((lab0,lab2[:int(len(lab2)*0.5)]), axis=0)
            
            lab0 = np.squeeze(np.argwhere(y_va == 0))
            lab2 = np.squeeze(np.argwhere(y_va == 2))
            indices_va = np.concatenate(([lab0],lab2), axis=0)
            x_va = x_va[indices_va]
            y_va = y_va[indices_va]

            lab0= np.squeeze(np.argwhere(y_te == 0))
            lab2 = np.squeeze(np.argwhere(y_te == 2))
            indices_te = np.concatenate((lab0,lab2), axis=0)
            x_te = x_te[indices_te]
            y_te = y_te[indices_te]



        x_tr_new = x_tr[indices_tr]
        y_tr_new = y_tr[indices_tr]

       
        #Assign the same label of the trial to each data point in the trial   
        #Create train matrix  
        X_train = np.concatenate(x_tr_new, axis = 0)
        y_train = np.concatenate([lab(y, x_tr_new[0].shape[0]) for y in y_tr_new])
        #y_train[y_train==2] = 1
        #Create validation matrix
        X_valid = np.concatenate(x_va, axis=0)
        y_valid= np.concatenate( [lab(y, x_va[0].shape[0]) for y in y_va]  )
        #y_valid[y_valid==2] = 1
        #Create test matrix
        X_test = np.concatenate(x_te, axis=0)
        y_test = np.concatenate( [lab(y, x_te[0].shape[0]) for y in y_te]  )
        #y_test[y_test==2] = 1
        
        if alter_label:
            y_train[y_train==2] = 1
            y_valid[y_valid==2] = 1
            y_test[y_test==2] = 1
   
        
    elif model_type == 'Image-Single' or model_type == 'Image-Multi' or model_type == 'Image-Simple':
        loaded = np.load(filename)
        X_train = loaded['train_img']
        y_train = loaded['train_labels']

        X_valid = loaded['valid_img']
        y_valid = loaded['valid_labels']
 
        X_test = loaded['test_img']
        y_test = loaded['test_labels']

  
        X_train = np.array( [np.rollaxis(x,0,3) for x in X_train ] )
        X_valid = np.array( [np.rollaxis(x,0,3) for x in X_valid] )
        X_test = np.array( [np.rollaxis(x,0,3) for x in X_test ] )

    
    elif model_type == 'Video-Single' or model_type == 'Video-Multi':
        loaded = np.load(filename)
        X_train = loaded['train_video']
        y_train = loaded['train_labels']
        
        X_valid = loaded['valid_video']
        y_valid = loaded['valid_labels']


        X_test = loaded['test_video']
        y_test = loaded['test_labels']
 
        X_train = np.array( [np.rollaxis(x,1,4) for x in X_train] )
        X_valid = np.array( [np.rollaxis(x,1,4) for x in X_valid ] )
        X_test = np.array( [np.rollaxis(x,1,4) for x in X_test ] )
       

    ################# Alter labels to 2 Class case: DE vs NE #################
    if alter_label and (model_type != 'Baseline' and model_type!='Logistic') :
        #Choose data has label NE(1) and DE(2)

        lab0 = np.squeeze(np.argwhere(y_train == 0))
        lab2 = np.squeeze(np.argwhere(y_train == 2))
        indices_train = np.concatenate((lab0,lab2), axis=0)

        lab0 = np.squeeze(np.argwhere(y_valid == 0))
        lab2 = np.squeeze(np.argwhere(y_valid == 2))
        indices_valid = np.concatenate((lab0,lab2), axis=0)

        lab0 = np.squeeze(np.argwhere(y_test == 0))
        lab2 = np.squeeze(np.argwhere(y_test == 2))
        indices_test = np.concatenate((lab0,lab2), axis=0)
        
        #Change labelling 2->1 for 2-class case 
        X_train = X_train[indices_train]
        y_train = y_train[indices_train] 
        y_train[y_train==2] = 1

        X_test = X_test[indices_test]
        y_test = y_test[indices_test]
        y_test[y_test==2] = 1

        X_valid = X_valid[indices_valid]
        y_valid = y_valid[indices_valid]
        y_valid[y_valid==2] = 1
    ###################################################################################

 
    #Shuffle the datasets
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]  

    indices_test = np.arange(X_test.shape[0])
    np.random.shuffle(indices_test)
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]


    indices_valid = np.arange(X_valid.shape[0])
    np.random.shuffle(indices_valid)
    X_valid = X_valid[indices_valid]
    y_valid = y_valid[indices_valid]

    # Sample the dataset for big samples
    if sample:
        sample_frac = 0.2

        np.random.shuffle(indices_train)
        indices_train = indices_train[:int(len(indices_train)*sample_frac)]
        X_train = (X_train[indices_train]) #[:1000]
        y_train = (y_train[indices_train]) #[:1000]
       
      
        np.random.shuffle(indices_valid)
        indices_valid = indices_valid[:int(len(indices_valid)*sample_frac)]
        X_valid = X_valid[indices_valid]
        y_valid = y_valid[indices_valid]


        np.random.shuffle(indices_test)
        indices_test = indices_test[:int(len(indices_test)*sample_frac)]
        X_test = X_test[indices_test]
        y_test = y_test[indices_test]

    # Print label count of datasets by trial
    unique, counts = np.unique(y_train , return_counts=True)
    print('Label distribution in the train set ', np.asarray((unique, counts)).T)

    unique, counts = np.unique(y_valid , return_counts=True)
    print('Label distribution in the validation set ', np.asarray((unique, counts)).T)
    
    unique, counts = np.unique(y_test , return_counts=True)
    print('Label distribution in the test set ', np.asarray((unique, counts)).T)


    # encode labels as one-hot vectors
    y_train = np_utils.to_categorical(y_train)
    y_valid = np_utils.to_categorical(y_valid)
    y_test = np_utils.to_categorical(y_test)


    print('Train matrix: ', X_train.shape, y_train.shape)
    print('Validation matrix: ', X_valid.shape, y_valid.shape)
    print('Test matrix: ', X_test.shape, y_test.shape)


    return X_train, y_train, X_test, y_test, X_valid, y_valid  


def train_tune(sample,filename, model_type, batch_size=32, num_epochs=10, run_id=0):
    """
    Tune hyper-parameters of models via grid search
    :param sample: boolean, if true data is downsampled with fraction of 0.2 
    :param filename: data file name
    :param model_type: model to be tested
    :param batch_size: batch_size of SGD
    :param num_epochs:  number of epochs of training 
    :param run_id: runner identifier 
    """

    X_train, y_train , X_test, y_test, X_valid, y_valid = reformat_input(filename, model_type, sample)
    
    X_train = X_train.astype("float32", casting='unsafe')
    X_valid = X_valid.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    
    ############################ Play with parameters ##############################
    nb_channels=3
    dropoutRate = 0.5
    acts= ['sigmoid','relu','tanh',None]
    k_sizes= [3,5,7]
    d_layers = [128,256,512]
    k_regularizer = regularizers.l2(0.001)
    input_dim = 256 #512
    ###############################################################################
    

    # Create neural network models
    print("Building model and compiling functions...")
    # Building the appropriate model

    if model_type == 'Baseline':
        model = Baseline_NN(nb_channels=nb_channels, dropoutRate = dropoutRate, k_regularizer = k_regularizer, input_dimension = input_dim)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
  
   
    elif model_type == 'Image-Simple':
        model = Simple_CNN()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall])


    elif model_type == 'Image-Single':
        for act in acts:
            for k_size in k_sizes:
                for d_layer in d_layers:
                         
                     model = CNN_Image(nb_channels=nb_channels, dropoutRate = dropoutRate, act=act, k_size=k_size, d_layer = d_layer, 
                                       k_regularizer = k_regularizer, img_size=X_train.shape[1], num_color_chan=X_train.shape[3])
                     sgd= optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #this is  for raw image
                     model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
                     print("Starting training...")
                     model.summary()
                     model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)
                     scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
                     print("Accuracy: %.2f%%" % (scores[1]*100))
    
    elif model_type == 'Image-Multi':
       for act in acts:
            for k_size in k_sizes:
                for d_layer in d_layers:
                         
                     model = CNN_Image_Multi(nb_channels=nb_channels, dropoutRate = dropoutRate, act=act, k_size=k_size, d_layer = d_layer, 
                                       k_regularizer = k_regularizer, img_size=X_train.shape[1], num_color_chan=X_train.shape[3])
                     sgd= optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #this is  for raw image
                     model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
                     print("Starting training...")
                     model.summary()
                     model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)
                     scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
                     print("Accuracy: %.2f%%" % (scores[1]*100))
        

    elif model_type == 'Video-Single':
        for act in acts:
            for k_size in k_sizes:
                for d_layer in d_layers:

                     model = CNN_Video(nb_channels=nb_channels, dropoutRate = dropoutRate, act=act, k_size=k_size, d_layer = d_layer, k_regularizer = k_regularizer)
                     sgd =  optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #this is for single channel video
                     model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
                     print("Starting training...: ", model_type )
                     model.summary()
                     model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)
                     scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
                     print("Accuracy: %.2f%%" % (scores[1]*100))
    
    elif model_type == 'Video-Multi':
        for act in acts:
            for k_size in k_sizes:
                for d_layer in d_layers:

                    model = CNN_Video_Multi(nb_channels=nb_channels, dropoutRate = dropoutRate, act=act, k_size=k_size, d_layer = d_layer, k_regularizer = k_regularizer)
                    sgd =  optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #this is for multichannel video
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
                    print("Starting training...: ", model_type )
                    model.summary()
                    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)
                    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
                    print("Accuracy: %.2f%%" % (scores[1]*100))

    elif model_type == 'LSTM':
        model = LSTM(time_slot= X_train.shape[1], img_size=X_train.shape[2], num_color_chan=X_train.shape[4])
    
    else:
        raise ValueError("Model not supported []")




    #print("Starting training...")
    #model.summary()
    #model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)
    #scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    #print("Accuracy: %.2f%%" % (scores[1]*100))
    # Save the model
    #model.save('./results/'+model_type+'/run'+str(run_id)+'_'+filename+str(num_epochs)+'.h5')


def train(sample, filename, model_type, batch_size=32, num_epochs=10, run_id=0):
    """
    Train models

    :param sample: boolean, if true data is downsampled with fraction of 0.2 
    :param filename: data file name
    :param model_type: model to be tested
    :param batch_size: batch_size of SGD
    :param num_epochs:  number of epochs of training 
    :param run_id: runner identifier 
    """

    # Flag To change labelling to 2-class
    label_two = False
  
    X_train, y_train , X_test, y_test, X_valid, y_valid = reformat_input(filename, model_type, sample, alter_label = label_two)
    
    X_train = X_train.astype("float32", casting='unsafe')
    X_valid = X_valid.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    

    nb_channels=3
    # If true, train for 2-class
    if(label_two): 
        nb_channels = 2
    


    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    # Building the appropriate model
    if model_type == 'Logistic':
        model  = LogisticRegression(nb_channels= nb_channels, k_regularizer = regularizers.l2(0.001), input_dimension = X_train.shape[1])
        sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True) #this is for multichannel image
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', precision,recall])


    elif model_type == 'Baseline':
        model = Baseline_NN(nb_channels=nb_channels, dropoutRate = 0.5, k_regularizer = regularizers.l2(0.0001), input_dimension = X_train.shape[1])
        sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True) 
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
  
   
    elif model_type == 'Image-Simple':
        model = Simple_CNN()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall])

    elif model_type == 'Image-Single':
        model = CNN_Image(nb_channels=nb_channels, dropoutRate = 0.5, act='relu', k_size=5, d_layer = 256, 
                          k_regularizer = regularizers.l2(0.0001), img_size=X_train.shape[1], num_color_chan=X_train.shape[3])
        sgd= optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True) #this is  for raw image
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
    
    elif model_type == 'Image-Multi':
        model = CNN_Image_Multi(nb_channels=nb_channels, dropoutRate = 0.5, act='relu', k_size=3, d_layer = 256, 
                                k_regularizer = regularizers.l2(0.0001), img_size=X_train.shape[1], num_color_chan=X_train.shape[3])
        sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True) #this is for multichannel image
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])


    elif model_type == 'Video-Single':

        model = CNN_Video(nb_channels=nb_channels, dropoutRate = 0.5, act='relu', k_size=5, d_layer = 256, k_regularizer = regularizers.l1(0.0001))
        sgd =  optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True) #this is for single channel video
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])
    
    elif model_type == 'Video-Multi':
      
        model = CNN_Video_Multi(nb_channels=nb_channels, dropoutRate = 0.5, act='relu', k_size=3, d_layer = 128, k_regularizer = regularizers.l2(0.0001))
        sgd =  optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True) #this is for multichannel video
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',precision,recall])

    elif model_type == 'LSTM':
        model = LSTM(time_slot= X_train.shape[1], img_size=X_train.shape[2], num_color_chan=X_train.shape[4])
    
    else:
        raise ValueError("Model not supported []")

    print("Starting training...: model ", model_type, ' batch size ', batch_size, ' with input ', filename)
    # Model summary 
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)
    prediction = model.predict(X_test)
    # Count number of unique target values for each possible one
    print(np.bincount(np.argmax(prediction,axis=-1))  )
    
    # Calculate the metrics specified for training
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # Save the model
    model.save('./results/'+model_type+'/run'+str(run_id)+'_'+ str(nb_channels)+'_class_'+filename[:-4]+ '_' + str(num_epochs)+'.h5')



def main():
    
    try:
        filename = (sys.argv[1]) # Data file name
        model_type =(sys.argv[2]) # Model type
        nb_epochs = int(sys.argv[3]) # Number of epochs
        b_size = int(sys.argv[4]) # Batch size of SGD
        sample = ast.literal_eval(sys.argv[5]) # If true data downsampled to 0.2 of size
        r_id = (sys.argv[6]) # running identifier
    except:
        print('Please specify the training parameters: filename, model type, number of epochs, batch size, if sample and run identifier ')

    train(sample, filename, model_type, batch_size=b_size, num_epochs=nb_epochs, run_id = r_id)
    
    #train(True, '20sec_raw_data_zip.npz', 'Baseline', batch_size=100, num_epochs=30, run_id = 1)
    #train(False, '2sec_fft_data_SW_zip.npz', 'Baseline', batch_size=10, num_epochs=30, run_id = 1)

    #train(True, '32_32_last20sec_img.npz', 'Image-Single', batch_size=100, num_epochs=30, run_id = 1)
    #train(False, '32_32_multichannel_img.npz', 'Image-Multi', batch_size=10, num_epochs=2, run_id = 1)

    #train(False, '32_32_last20sec_videos.npz', 'Video-Single', batch_size=100, num_epochs=15, run_id = 1)
    #train(False,'32_32_multichannel_videos.npz', 'Video-Multi', batch_size=10, num_epochs=10, run_id = 1)

    # Hyper-parameter tuning
    #train_tune(True, '32_32_last20sec_videos.npz', 'Video-Single', batch_size=32, num_epochs=3, run_id = 0)
    #train_tune(False, '32_32_multichannel_videos.npz', 'Video-Multi', batch_size=10, num_epochs=3, run_id = 0)


if __name__ == '__main__':
      main()      

