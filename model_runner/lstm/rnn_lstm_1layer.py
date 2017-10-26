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


    cell_lst=[]
    for i in range(num_layers):
      cell = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.contrib.layers.xavier_initializer(uniform=False)
                                   , forget_bias=1.0)
      # if i==0:
      #   cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob= self.output_keep_prob)
      #   cell=cell_drop
      cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob= self.output_keep_prob)
      cell=cell_drop
      cell_lst.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cell_lst)

    # cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob= self.output_keep_prob)
    # cell=cell_drop
    self.cell = cell

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.input_zero = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[None, params['seq_length']])
    self.target_data =tf.placeholder(tf.float32, [None,params["seq_length"],params["n_output"]])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    #Noise applied only training phase and if only std bigger than 0
    if(params["noise_std"]>0.0):
      ran_noise = tf.random_normal(shape=[params["batch_size"],params['seq_length'], params['n_input']], mean=0, stddev=params['noise_std'])
      # ran_noise=tf.mul(ran_noise,self.input_zero)
      tmp_input=tf.nn.relu(self.input_data+ran_noise)
      self.input_data=tf.select(self.is_training,tmp_input,self.input_data)

    outputs = []
    state = self.initial_state
    with tf.variable_scope("rnnlm"):
      for time_step in range(params['seq_length']):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(self.input_data[:,time_step,:], state)
        outputs.append(cell_output)
    rnn_output = tf.reshape(tf.transpose(tf.pack(outputs),[1,0,2]), [-1, params['n_hidden']])

    with tf.variable_scope('rnnlm'):
      output_w1 = tf.get_variable("output_w1", [rnn_size, NOUT],initializer=tf.contrib.layers.xavier_initializer() )
      output_b1 = tf.get_variable("output_b1", [NOUT])

    self.final_output = tf.add(tf.matmul(rnn_output, output_w1),output_b1)

    flt=tf.squeeze(tf.reshape(self.repeat_data,[-1,1]),[1])
    where_flt=tf.not_equal(flt,0)
    indices=tf.where(where_flt)
    tmp = self.final_output -  tf.reshape(self.target_data,[-1,params["n_output"]])
    tmp=tf.gather(tmp,tf.squeeze(indices,[1]))
    loss=  tf.nn.l2_loss(tmp)
    self.cost = tf.reduce_mean(loss)
    self.final_state = state
    tf.scalar_summary('losses/total_loss', loss)



    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    total_parameters = 0
    for variable in tvars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    self.total_parameters=total_parameters
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

