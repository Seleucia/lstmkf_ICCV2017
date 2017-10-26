from helper import utils as ut
from helper import dt_utils as dut
from tensorflow.python.ops import rnn
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

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    lstm_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * num_layers)
    lstm_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * num_layers)
    lstm_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_bw, output_keep_prob = self.output_keep_prob)
    lstm_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_fw, output_keep_prob = self.output_keep_prob)

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.target_data =tf.placeholder(tf.float32, [params["batch_size"]*params["seq_length"],params["n_output"]])
    self.fw_initial_state = lstm_fw.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
    self.bw_initial_state = lstm_bw.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    ran_noise = tf.random_normal(shape=[params["batch_size"],params['seq_length'], params['n_input']], mean=0, stddev=0.00008)
    self.input_data=tf.select(self.is_training,self.input_data+ran_noise,self.input_data)

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [rnn_size*2, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    outputs = []
    fw_state= self.fw_initial_state
    bw_state = self.bw_initial_state

    # with tf.variable_scope("rnnlm"):
      # for time_step in range(params['seq_length']):
    inputs = tf.transpose(self.input_data, [1, 0, 2])
    inputs = tf.reshape(inputs, [-1, params['n_input']])
    inputs = tf.split(0, params['seq_length'], inputs)

    # inputs= self.input_data[:,time_step,:]
        # print inputs
        # if time_step > 0: tf.get_variable_scope().reuse_variables()
    cell_output, fw_state, bw_state = rnn.bidirectional_rnn(lstm_fw, lstm_bw,inputs=inputs,
                                                                initial_state_fw=fw_state, initial_state_bw=bw_state,dtype=tf.float32)


    output = tf.reshape(tf.concat(1, cell_output), [-1, params['n_hidden']*2])
    self.final_output = tf.tanh(tf.matmul(output, output_w) + output_b)
    tmp = self.final_output - self.target_data
    loss=  tf.nn.l2_loss(tmp)
    self.cost = tf.reduce_mean(loss)
    self.fw_final_state = fw_state
    self.bw_final_state = bw_state


    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))