import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from dreamUtils import *
from dreamNetworks import *
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = './'

def run_baseline_NN(run_id, sec):

    # Load raw data 
    filename = DATA_DIR + str(sec) + 'sec_raw_data_zip'
    loaded = np.load(filename+'.npz')
    rawdata = loaded['data']
    label = loaded['labels']
    
    
    # centerize raw data for reference electrode
    rawdata_normalized = []
    #scaler = MinMaxScaler(feature_range=(-1,1))
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
    

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
           
    print ('Start running baseline ', str(run_id), ' for ', str(sec), 'raw data')  
    ### Baseline NN
    model = Baseline_NN()



    # fit and evaluate model
    model.fit(X_train, y_train,epochs=100, batch_size=1000, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=2)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.save(DATA_DIR+'results/baseline-run-'+str(run_id)+'.h5')
    


def main():
    run_id = 0
    sec = 20
    try:
        run_id = int(sys.argv[1])
        sec = int(sys.argv[2])
    except Exception  as e:
        print('Identify run_id and second')

    run_baseline_NN(run_id, sec)
  
    print('All done!')  
        
# Command line args are in sys.argv[1], sys.argv[2] ..
# sys.argv[0] is the script name itself and can be ignored

# Standard boilerplate to call the main() function to begin
# the program.

if __name__ == '__main__':
      main()      
    