from helper import utils as ut
from helper import dt_utils as dut

import tensorflow as tf

import numpy as np
import random

class Model():
  def __init__(self,params, infer=False):

    self.is_training = tf.placeholder(tf.bool)
    self.output_keep_prob = tf.placeholder(tf.float32)

    num_layers=params['nlayer']
    rnn_size=params['n_hidden']
    grad_clip=2

    cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    cell = cell_fn(rnn_size)#RNN size
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)
    self.cell = cell

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], 1024])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], NOUT])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    ran_noise = tf.random_normal(shape=[params["batch_size"], params['seq_length'], 1024], mean=0, stddev=0.00008)
    self.input_data=tf.select(self.is_training,self.input_data+ran_noise,self.input_data)

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    inputs = tf.split(1, params['seq_length'], self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    self.tt=inputs
    self.pre_outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
    self.output = tf.reshape(tf.concat(1, self.pre_outputs), [-1, rnn_size])
    self.final_output = tf.nn.xw_plus_b(self.output, output_w, output_b)
    self.final_state = last_state


    # reshape target data so that it is compatible with prediction shape
    target_data = tf.split(1, params['seq_length'], self.target_data)
    self.pre_target_data = [tf.squeeze(y_, [1]) for y_ in target_data]
    self.final_target_data=tf.reshape(tf.concat(1, self.pre_target_data), [-1, NOUT])

    self.flat_target_data = tf.reshape(self.final_target_data, [-1])
    self.flat_output_data = tf.reshape(self.final_output, [-1])
    print self.pre_outputs
    print self.output
    print self.final_output
    print self.flat_output_data

    # [x1_data, x2_data, eos_data] = tf.split(1, 3, flat_target_data)

    # long method:
    #flat_target_data = tf.split(1, args.seq_length, self.target_data)
    #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]


    self.cost = tf.reduce_sum(tf.square(self.flat_output_data - self.flat_target_data))

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
