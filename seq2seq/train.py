from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

#import utils
from model import ClickPredictionModel
from encoder import EncoderModel


def train(config):
  
    # reset graph
    tf.reset_default_graph()
    

    # Initialize the model
    with tf.variable_scope('dec'): 
      model = ClickPredictionModel(
          batch_size=config.batch_size,
          seq_length=config.seq_length,
          lstm_num_hidden=config.lstm_num_hidden,
          lstm_num_layers=config.lstm_num_layers
      )
    with tf.variable_scope('enc'): 
      encoder = EncoderModel(
          batch_size=config.batch_size,
          seq_length=config.seq_length,
          lstm_num_hidden=config.lstm_num_hidden,
          lstm_num_layers=config.lstm_num_layers
      )
    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Define the optimizer
    
    # Passing global_step to minimize() will increment it at each step.
    global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
    starter_learning_rate = config.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           config.learning_rate_step, config.learning_rate_decay, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)


    ###########################################################################
    # Implement code here.
    ###########################################################################
    
    # monitor progress
    saver = tf.train.Saver(max_to_keep= 40)
    model_name = 'model'
    model_type = 'seq2seq'
    checkpoint_path = './checkpoints/' + model_type +'/'
    if not tf.gfile.Exists(checkpoint_path):
      tf.gfile.MakeDirs(checkpoint_path)
    
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(config.summary_path+ model_name + '/')
  
    save_freq = 10
    #
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # If uncomment the next line -> put in comment sess.run(tf.global_variables_initializer()) as well as anything that seems necessary
    #saver.restore(sess, checkpoint_path+"The name of the model you want to restore")
    
        
    for train_step in range(int(config.train_steps)):
        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################################
        # Implement code here.
        #######################################################################

        batch_inputs, batch_targets =  np.load('./batches/inputs.npy'), np.load('./batches/targets.npy')
        batch_inputs_enc = np.load('./batches/inputs.npy');
        batch_inputs_enc = batch_inputs_enc[:,:,:-1]
    
        # sess.run ( .. )
        st_ = sess.run ([ encoder._states_enc], feed_dict={encoder._inputs_no_i:batch_inputs_enc, 
                             encoder._state_placeholder_:np.zeros((config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden)),
                             encoder.keep_prob_ : config.dropout_keep_prob})
  
        init_state_dec = st_[0]
        
        _, loss_ = sess.run ([ apply_gradients_op, model.loss], feed_dict={model.inputs:batch_inputs, 
                             model.targets:batch_targets,
                             model._state_placeholder:init_state_dec,
                             model.keep_prob : config.dropout_keep_prob})

        # monitor progress        
        summary_ = tf.Summary()
        summary_.value.add(tag="Loss", simple_value=loss_)
        summary_writer.add_summary(summary_, global_step = train_step+1)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(config.train_steps), config.batch_size, examples_per_second, loss_
            ))
            
        # Save model
        if(train_step % save_freq == 0):
          save_path = saver.save(sess, checkpoint_path + model_name, global_step=train_step)
          print('Model saved at %s' % (save_path))
            


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)