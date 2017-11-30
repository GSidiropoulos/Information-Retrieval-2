# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:32:44 2017

@author: theft
"""

import pickle
import json
import numpy as np
from random import sample


# splits an input_file with split_length
def split_file(input_file, output_file, split_length=1e5, print_every=1e4):
    
    with open(input_file) as fin:
        
        fout = output_file

        lines = []
        
        # loop over the input_file and append the lines in a list
        for i,line in enumerate(fin):
            
            lines.append(line)
            if (i+1)%split_length==0 and i>0:

                # pickle the list after split length steps
                with open(fout+"_"+str(int(i/(split_length-1))),'wb') as f2:
                    pickle.dump(lines, f2)

                print("Split at "+str(i+1)+" lines")
                lines = []

# loads a pickle file
def load_pickle(input_file):
    
    with open (input_file, 'rb') as fp:
        pickle_file = pickle.load(fp)
    return pickle_file
    

# transforms a pickle to a json file
def load_json(input_file):
    
    return [json.loads(j) for j in input_file]   
        

            
# get a batch with a qd representation
def get_batch_qd(dataset , batch_size, start=0):
    
    # check if batch exceeds batch_size
    if start+batch_size <= len(dataset):
        
        
        serps = dataset[start:start+batch_size]
        #serps = sample(dataset,128)
    
    else:
        # get again the first nth training to complete the batch
        # or sample from the whole dataset for the rest of the batch
        # serps = sample(dataset,128)
        serps = dataset[start:]
        serps += dataset[0:batch_size-(len(dataset)-start)]
        

    # create the zero arrays
    representations = np.zeros((batch_size,11,10242),dtype=np.int32)
    
    targets = np.zeros((batch_size, 10), dtype=np.int32)
    
    # for each query documents pair
    for i, serp in enumerate(serps):
        
        # for each document in a serp
        for j , num_of_clicks in enumerate(serp['behavioral_features']['D']):
        
            # get representations for docs
            representations[i][j+1][np.asarray(list(num_of_clicks.keys()), dtype=np.int32)+1] = np.asarray(list(num_of_clicks.values()), dtype=np.int32)
        
        # get representations for interactions
        representations[i,:,-1][2:] = np.asarray(list(serp['click_pattern'])[:-1], dtype=np.int32)
        
        # get the targets
        targets[i] = np.asarray(list(serp['click_pattern']), dtype=np.int32)
            
    return representations , targets


    
    
# get a batch with representations for QD + Q set
def get_batch_qd_plus_q(dataset, batch_size, start=0):
    
    # check if batch exceeds batch_size
    if start+batch_size <= len(dataset):
        
        
        serps = dataset[start:start+batch_size]
        #serps = sample(dataset,128)
    
    else:
        # get again the first nth training to complete the batch
        # or sample from the whole dataset for the rest of the batch
        # serps = sample(dataset,128)
        serps = dataset[start:]
        serps += dataset[0:batch_size-(len(dataset)-start)]
    
    
    # create the zero-like arrays
    representations = np.zeros((batch_size,11,10240+1024+1),dtype=np.int32)
    
    targets = np.zeros((batch_size, 10), dtype=np.int32)
    
    
    # foreach serp
    for i, serp in enumerate(serps):
        
        # get the representations for the queries
        representations[i][0][np.asarray(list(serp['behavioral_features']['Q'].keys()), dtype=np.int32)] = np.asarray(list(serp['behavioral_features']['Q'].values()), dtype=np.int32)
        
        # get the representations for the interactions
        representations[i,:,-1][2:] = np.asarray(list(serp['click_pattern'])[:-1], dtype=np.int32)
        
        # get the targets to return
        targets[i] = np.asarray(list(serp['click_pattern']), dtype=np.int32)

        # foreach doc get the representations
        for j , num_of_clicks in enumerate(serp['behavioral_features']['D']):
            
            representations[i][j+1][np.asarray(list(num_of_clicks.keys()), dtype=np.int32)+1024] = np.asarray(list(num_of_clicks.values()), dtype=np.int32)
    
    return representations , targets




# name of the file             
#input_file = "dataset/train_query_sessions_with_behavioral_features.part_0"

# split file
#split_file(input_file, "split")


# load pickle file
#pickle_file = load_pickle("split_1")

# trasnform the pickle file to get the json_file again
#dataset = load_json(pickle_file)


#x_train_qd, y_train_qd = get_batch_qd(dataset,128)
    
#x_train_qd_plus_q , y_train_plus_q = get_batch_qd_plus_q(dataset,128)
