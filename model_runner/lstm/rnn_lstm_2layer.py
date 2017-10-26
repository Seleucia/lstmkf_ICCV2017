from tensorflow.python.framework import ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import clip_ops
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
    grad_clip=10

    cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    cell = cell_fn(rnn_size)#RNN size
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)
    self.cell = cell

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.input_zero = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.target_data =tf.placeholder(tf.float32, [params["batch_size"]*params["seq_length"],params["n_output"]])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    ran_noise = tf.random_normal(shape=[params["batch_size"],params['seq_length'], params['n_input']], mean=0, stddev=0.00008)
    ran_noise=tf.mul(ran_noise,self.input_zero)
    tmp_input=tf.nn.relu(self.input_data+ran_noise)
    self.input_data=tf.select(self.is_training,tmp_input,self.input_data)

    outputs = []
    state = self.initial_state
    with tf.variable_scope("rnnlm"):
      for time_step in range(params['seq_length']):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(self.input_data[:,time_step,:], state)
        outputs.append(cell_output)
    rnn_output = tf.reshape(tf.concat(1, outputs), [-1, params['n_hidden']])

    with tf.variable_scope('rnnlm'):
      output_w1 = tf.get_variable("output_w1", [rnn_size, rnn_size])
      output_b1 = tf.get_variable("output_b1", [rnn_size])

      output_w2 = tf.get_variable("output_w2", [rnn_size, NOUT])
      output_b2 = tf.get_variable("output_b3", [NOUT])

    hidden_1 = tf.add(tf.matmul(rnn_output, output_w1),output_b1)
    self.final_output = tf.add(tf.matmul(hidden_1, output_w2),output_b2)


    tmp = self.final_output - self.target_data
    loss=  tf.nn.l2_loss(tmp)
    self.cost = tf.reduce_mean(loss)
    self.final_state = state
    tf.scalar_summary('losses/total_loss', loss)



    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    for grad in grads:
      # if isinstance(grad, ops.grads):
      #   grad_values = grad.values
      # else:
      #   grad_values = grad
      grad_values = grad
      logging_ops.histogram_summary(grad.op.name + ':gradient', grad_values)
      logging_ops.histogram_summary(grad.op.name + ':gradient_norm', clip_ops.global_norm([grad_values]))
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

