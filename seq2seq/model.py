from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class ClickPredictionModel(object):

    def __init__(self, batch_size, seq_length,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size

        # Initialization:
        self._inputs = tf.placeholder(tf.float32, 
                                      shape=[128, 11, 10242],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32,
                                       shape=[128, 10],
                                       name='targets')
        
        self._targets_rshaped = tf.reshape(self._targets, [-1,1])
        
        # encode to one hot representation
        #self._targets_one_hot = tf.one_hot(self._targets, 2)
        #self._targets_one_hot = tf.reshape(self._targets_one_hot, [-1, 2])

        with tf.variable_scope('model'):
          self._logits_per_step = self._build_model()
          self._probabilities = self.probabilities()
          self._predictions = self.predictions()
          self._loss = self._compute_loss()
          
    def _build_model(self):

        with tf.variable_scope('states'):        
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            
            stacked_cells = rnn.MultiRNNCell(
                [rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units=self._lstm_num_hidden, forget_bias=1.0), output_keep_prob=self.keep_prob)
                for _ in range(self._lstm_num_layers)])
            
            self._state_placeholder = tf.placeholder(tf.float32, [self._lstm_num_layers, 2, None, self._lstm_num_hidden])
            
            l = tf.unstack(self._state_placeholder, axis=1)
            
            self._rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
                                    for idx in range(self._lstm_num_layers)])
      
            outputs, self._states = tf.nn.dynamic_rnn(cell=stacked_cells,
                                               inputs=self._inputs,
                                               initial_state=self._rnn_tuple_state,
                                               dtype=tf.float32)   
            
            
            # since for the first we only have s_0 and no click prediction
            outputs_ =  outputs[:,1:,:]
            
            outputs_rshaped = tf.reshape(tensor=outputs_, shape=[-1, self._lstm_num_hidden])
            

        with tf.variable_scope("predictions"):
            W_out = tf.get_variable("W_out", 
                                    shape=[self._lstm_num_hidden, 1],
                                    initializer=tf.variance_scaling_initializer())

            b_out = tf.get_variable("b_out", shape=[1],
                                       initializer=tf.constant_initializer(0.0))
            
            predictions = tf.nn.bias_add(tf.matmul(outputs_rshaped, W_out), b_out)
            

        return predictions
           
           
    def _compute_loss(self):
        # Cross-entropy loss, averaged over timestep and batch
        with tf.name_scope('log_likelihood'):
          loss= tf.reduce_mean(( tf.losses.log_loss(labels=self._targets_rshaped, predictions=self._probabilities)))

        return loss

    def probabilities(self):
        # Returns the normalized per-step probabilities
        probabilities = tf.nn.sigmoid(self._logits_per_step)
        return probabilities

    def predictions(self):
        # Returns the per-step predictions
        predictions = tf.round(self._probabilities)
        return predictions
      
      
    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets


    @property
    def loss(self):
        return self._loss
      