# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 00:14:52 2017

@author: theft
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import argparse
import json

import numpy as np
import tensorflow as tf

import dataset
import ndcg_preprocess 
import ndcg

import pickle

from model import ClickPredictionModel

def train(config):
  
    
    
    tf.set_random_seed(42)
    np.random.seed(42)
    
    
    if config.representation == 'QD':
        get_batch = dataset.get_batch_qd
        representations_dims = 10242
        print('QD Representations\n')
    else:
        get_batch = dataset.get_batch_qd_plus_q
        representations_dims = 10240+1024+1
        print('QD+Q Representations\n')

    # Initialize the model
    model = ClickPredictionModel(
        batch_size=config.batch_size,
        num_of_dims = representations_dims,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers
    )

   
    # monitor progress
    saver = tf.train.Saver(max_to_keep= 40)
    model_type = 'lstm'+config.representation
    checkpoint_path = './checkpoints/' + model_type +'/'
    if not tf.gfile.Exists(checkpoint_path):
      tf.gfile.MakeDirs(checkpoint_path)
    
    

 
    
    sess = tf.Session()
    
    saver.restore(sess, checkpoint_path+config.model2load)
    

    
    
    print("\n\nStarting Test at : ",datetime.now().strftime("%Y-%m-%d %H:%M"))

   

    ndcg_at_1=[]
    ndcg_at_3=[]
    ndcg_at_5=[]
    ndcg_at_10=[]
    # test the model over the various test files and report the mean NDCG over the whole set
    for test_file in range(config.num_of_test_files):
        
        print("\nStart NDCG Test on File : ",test_file)
        input_file = 'dataset/full_ndcg'+str(test_file)+'.test'
        
        gzip_file = open(input_file,'r')
    
        dataset_ = []
            
        # loop over the input_file and append the lines in a list
        for train_step,line in enumerate(gzip_file):
            
            dataset_.append(json.loads(line.strip("\n")))
            
            if (train_step+1)%config.batch_size==0 and train_step>0:
                
                # get the batches
                batch_inputs, batch_targets, queries, docs =  get_batch(dataset_,config.batch_size)
            
                probs = sess.run ([model._probabilities],
                                               feed_dict={model.inputs:batch_inputs, 
                                     model.targets:batch_targets,
                                     model._state_placeholder:np.zeros((config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden)),
                                     model.keep_prob : 1.})
        
                # calculate the ndcg scores at differen ranks
                ndcg_probs = np.reshape(probs,[config.batch_size,10])
                rel = np.zeros((config.batch_size,10),dtype=np.int32)
                for i,d in enumerate(docs):
                    for d1,d2 in enumerate(d):
                        if d2 in relevance[str(queries[i])]['relevant']:
                            rel[i][d1]=1


                ndcg_at_1.append(ndcg.ndcg(ndcg_probs, rel,1))
                ndcg_at_3.append(ndcg.ndcg(ndcg_probs, rel,3))
                ndcg_at_5.append(ndcg.ndcg(ndcg_probs, rel,5))
                ndcg_at_10.append(ndcg.ndcg(ndcg_probs, rel,10))
                

                dataset_ = []

       
    # report the mean NDCG scores over the dataset and store them in a file
    print("Ndcg @1 : ",sum(ndcg_at_1)/len(ndcg_at_1))
    print("Ndcg @3 : ",sum(ndcg_at_3)/len(ndcg_at_3))
    print("Ndcg @5 : ",sum(ndcg_at_5)/len(ndcg_at_5))
    print("Ndcg @10 : ",sum(ndcg_at_10)/len(ndcg_at_10))
    print("\nFinished Testing at : ",datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    with open('dataset/lstm_ndcg_at_1'+config.representation,'wb') as f1:
        pickle.dump(ndcg_at_1, f1)
    with open('dataset/lstm_ndcg_at_3'+config.representation,'wb') as f2:
        pickle.dump(ndcg_at_3, f2)
    with open('dataset/lstm_ndcg_at_5'+config.representation,'wb') as f3:
        pickle.dump(ndcg_at_5, f3)
    with open('dataset/lstm_ndcg_at_10'+config.representation,'wb') as f4:
        pickle.dump(ndcg_at_10, f4)
    



            
            
            
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--num_of_test_files', type=int, default=4, help='Numbers of training files')
    parser.add_argument('--representation', type=str, default='QD+Q', help='Type of Representation')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--model2load', type=str, default='model_3-896127', help='The name of the model you want to restore')
    parser.add_argument('--test_files_path', type=str, default='dataset/test_query_sessions_with_behavioral_features.part_'
                        , help='The path of the test files')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1., help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=1.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./lstm/summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')

    config = parser.parse_args()

    
    
    

    path = 'dataset/Trainq.txt'

    lines = ndcg_preprocess.read_file(path)
    
    
    relevance = ndcg_preprocess.trainq2dict(lines)
    
    relevance = ndcg_preprocess.values2counter(relevance)
    
    
    relevance = ndcg_preprocess.validate_rel(relevance)
    
    relevance = ndcg_preprocess.filter_rel(relevance)
    
    
    tf.reset_default_graph()
    
    train(config)
        
        
 