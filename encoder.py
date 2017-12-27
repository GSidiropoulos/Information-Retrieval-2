#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 23:40:04 2017

@author: georgios
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# encoder module

class EncoderModel(object):
    def __init__(self, batch_size,lstm_num_hidden,
                 num_of_dims, lstm_num_layers):

        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._representations_dims = num_of_dims-1

        

        # Initialization:
        self._inputs_no_i = tf.placeholder(tf.float32, 
                                      shape=[self._batch_size, 11, self._representations_dims],
                                      name='inputs_enc')
        
        
        with tf.variable_scope('model_enc'):
          self._states_enc = self._build_model()
 
    def _build_model(self):

        with tf.variable_scope('states_enc'):        
            self.keep_prob_ = tf.placeholder(tf.float32, name='keep_prob_enc')
            
            stacked_cells_ = rnn.MultiRNNCell(
                [rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units=self._lstm_num_hidden, forget_bias=1.0), output_keep_prob=self.keep_prob_)
                for _ in range(self._lstm_num_layers)])
            
            self._state_placeholder_ = tf.placeholder(tf.float32, [self._lstm_num_layers, 2, None, self._lstm_num_hidden])
            
            l_ = tf.unstack(self._state_placeholder_, axis=1)
            
            self._rnn_tuple_state_ = tuple([rnn.LSTMStateTuple(l_[idx][0],l_[idx][1])
                                    for idx in range(self._lstm_num_layers)])
      
            outputs, states = tf.nn.dynamic_rnn(cell=stacked_cells_,
                                               inputs=self._inputs_no_i,
                                               initial_state=self._rnn_tuple_state_,
                                               dtype=tf.float32)   
            
            
            return states
           
