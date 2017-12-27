from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import argparse
import gzip
import json

import numpy as np
import tensorflow as tf

import dataset
import pickle

from model import ClickPredictionModel


def train(config):
  
    
    # set seeds for reproduce
    
    tf.set_random_seed(42)
    np.random.seed(42)
    
    
    # set the representation dimensions
    
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
    model_name = 'model'
    model_type = 'lstm'+config.representation
    checkpoint_path = './checkpoints/' + model_type +'/'
    if not tf.gfile.Exists(checkpoint_path):
      tf.gfile.MakeDirs(checkpoint_path)
    
    summary_writer = tf.summary.FileWriter(config.summary_path+ model_name + '/')
    
    
    # if training 
    
    if config.train == True:

        # Define the optimizer
        
        # Passing global_step to minimize() will increment it at each step.
        global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
        starter_learning_rate = config.learning_rate
        
        optimizer = tf.train.AdadeltaOptimizer(starter_learning_rate, epsilon=1e-05)
        
    
        # Compute the gradients for each variable
        grads_and_vars = optimizer.compute_gradients(-model.loss)
        
        # gradient clipping
        grads, variables = zip(*grads_and_vars)
        grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
        apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)
    
        # start the Session        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
       
        
        # start training
        
        print("Start Training at : ",datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        # looping over all the train files
        
        for train_file in range(config.num_of_train_files):
        
            print("\nStart File : ",train_file)
            input_file = config.train_files_path+str(train_file)+'.gz'
            
            gzip_file = gzip.open(input_file,'r')
        
            dataset_ = []
                
            # loop over the input_file and append the lines in a list
            for train_step,line in enumerate(gzip_file):
                
                dataset_.append(json.loads(line.decode("utf-8").strip("\n")))
                
                
                if (train_step+1)%config.batch_size==0 and train_step>0:
                    
                    # get the batches
                    batch_inputs, batch_targets, queries, docs =  get_batch(dataset_,config.batch_size)
                
                    # training step - apply operation
                    _, loss_,pred,probs = sess.run ([ apply_gradients_op, model.loss,model._predictions,model._probabilities],
                                                   feed_dict={model.inputs:batch_inputs, 
                                         model.targets:batch_targets,
                                         model._state_placeholder:np.zeros((config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden)),
                                         model.keep_prob : config.dropout_keep_prob})
                    # monitor progress        
                    summary_ = tf.Summary()
                    summary_.value.add(tag="Loss", simple_value=loss_)
                    summary_writer.add_summary(summary_, global_step = train_step+1)
            
                    # Output the training progress
                    if int(train_step/config.batch_size) % config.print_every == 0:
                        print("\n[{}] Train Step {:04d}/{:04d}, Batch Size = {},  Loss = {}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), int(train_step/config.batch_size)+1,
                            int(config.train_steps/config.batch_size), config.batch_size, loss_
                        ))
                        
                        
                        perplexity, _ = compute_perplexity(probs, batch_targets)
                        
                        print("Perplexity",perplexity)
                        print("Pred",len(np.nonzero(pred)[0]))
                        print("Targets",len(np.nonzero(batch_targets)[0]))
                        
                        save_path = saver.save(sess, checkpoint_path + model_name+"_"+str(train_file), global_step=train_step)
                        print('Model saved at %s' % (save_path))
                        
                        
                    dataset_ = [] 
    
        print("\nFinished at : ",datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("Final Loss : ",loss_)
        print("Final Perplexity : ",perplexity)
        
    else:
        
        # test the model
        
        sess = tf.Session()
        
        # load the model to test
        saver.restore(sess, checkpoint_path+config.model2load)
        
        # start testing
        print("\n\nStarting Test at : ",datetime.now().strftime("%Y-%m-%d %H:%M"))
    
        total_loss = []
        total_perplexity = []
        total_perplexity_at_r = []
    
        
        # test the model over the various test files and report the mean over the whole set
        for test_file in range(config.num_of_test_files):
            
            print("\nStart Test on File : ",test_file)
            input_file = config.test_files_path+str(test_file)+'.gz'
            
            gzip_file = gzip.open(input_file,'r')
        
            dataset_ = []
                
            # loop over the input_file and append the lines in a list
            for train_step,line in enumerate(gzip_file):
                
                dataset_.append(json.loads(line.decode("utf-8").strip("\n")))
                
                if (train_step+1)%config.batch_size==0 and train_step>0:
                    
                    # get the batches
                    batch_inputs, batch_targets, queries, docs =  get_batch(dataset_,config.batch_size)
                
                    # get the loss and the probabilities that the model outputs
                    loss_,probs = sess.run ([model.loss,model._probabilities],
                                                   feed_dict={model.inputs:batch_inputs, 
                                         model.targets:batch_targets,
                                         model._state_placeholder:np.zeros((config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden)),
                                         model.keep_prob : 1.})
            
                    
                    perplexity, perplexity_at_r = compute_perplexity(probs, batch_targets,False)
                    
                    total_loss.append(loss_)
                    
                    total_perplexity.append(perplexity)
                    
                    total_perplexity_at_r.append(perplexity_at_r)
                    
                    
            
                    # Output the training progress
                    if int(train_step/config.batch_size) % config.print_every == 0:
                        
                        print("\n[{}] Test Step {:04d}/{:04d}, Batch Size = {}, Loss = {}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), int(train_step/config.batch_size)+1,
                            int(config.train_steps/config.batch_size), config.batch_size, loss_
                        ))
                        
                        print("Perplexity", perplexity)
                        
                        
                        
                    dataset_ = []
    
        # report the mean over the dataset and log the statistics in files for statistical significance test  
        print("\nFinished Testing at : ",datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("Final Loss : ",sum(total_loss)/len(total_loss))
        print("Final Perplexity : ",sum(total_perplexity_at_r)/len(total_perplexity_at_r))
        print("Final Perplexity : ",sum(total_perplexity)/len(total_perplexity))
                
        with open('dataset/lstm_losses_'+config.representation,'wb') as f1:
            pickle.dump(total_loss, f1)
        with open('dataset/lstm_perplexity_'+config.representation,'wb') as f2:
            pickle.dump(total_perplexity, f2)
    
    
# returns the perplexity and the perplexity at rank 
def  compute_perplexity(probs, targets, report_per_rank=True):
    
    
    probs = np.reshape(probs,[config.batch_size,10])
    targets = np.reshape(targets,[config.batch_size,10])
    
    prob_clicks = np.log2(probs) * targets + (np.log2(1 - probs) * (1 - targets))
    
    
    perplexity_at_r = np.power(2,(-1/config.batch_size)*np.sum(prob_clicks,axis=0))
    
    if report_per_rank:
        print("Perplexity per Rank",perplexity_at_r)
        
    perplexity = np.mean(perplexity_at_r)
    

    return perplexity, perplexity_at_r
            
            
            
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--train', type=bool, default=True, help="Training or Testing")
    parser.add_argument('--num_of_train_files', type=int, default=4, help='Numbers of training files')
    parser.add_argument('--num_of_test_files', type=int, default=4, help='Numbers of training files')
    parser.add_argument('--representation', type=str, default='QD+Q', help='Type of Representation')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--model2load', type=str, default='model_3-896127', help='The name of the model you want to restore')
    parser.add_argument('--train_files_path', type=str, default='dataset/train_query_sessions_with_behavioral_features.part_'
                        , help='The path of the train files')
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

    
    
  
    
    tf.reset_default_graph()
    
    # train the model
    
    train(config)
        
        
