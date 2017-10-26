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

    # cell_lst=[]
    # cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    # cell = cell_fn(rnn_size)#RNN size
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)
    # cell_lst.append(cell)
    # cell_lst.append(cell_fn(rnn_size))
    # cell_lst.append(cell_fn(rnn_size))
    #
    # cell = tf.nn.rnn_cell.MultiRNNCell(cell_lst)
    # cell = tf.nn.dynamic_rnn.MultiRNNCell(cell_lst)

    # self.cell = cell

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.input_zero = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.target_data =tf.placeholder(tf.int64, [params["batch_size"],params["n_output"]])
    self.initial_state = tf.placeholder(dtype=tf.float32, shape=[params['nlayer']*2,params['batch_size'], params['n_hidden']])

    ran_noise = tf.random_normal(shape=[params["batch_size"],params['seq_length'], params['n_input']], mean=0, stddev=0.08)
    ran_noise=tf.mul(ran_noise,self.input_zero)
    tmp_input=tf.nn.relu(self.input_data+ran_noise)
    tmp_input=self.input_data+ran_noise
    self.input_data=tf.select(self.is_training,tmp_input,self.input_data)

    with tf.variable_scope('rnnlm'):
      output_w1 = tf.get_variable("output_w1", [params['n_input'], params['n_input']])
      output_b1 = tf.get_variable("output_b1", [params['n_input']])

      output_w2 = tf.get_variable("output_w2", [params['n_input'], params['n_input']])
      output_b2 = tf.get_variable("output_b2", [params['n_input']])

      output_w3 = tf.get_variable("output_w3", [params['n_input'], NOUT])
      output_b3 = tf.get_variable("output_b3", [NOUT])
      #
      # output_wseq = tf.get_variable("output_wseq", [params['seq_length'], 1])
      # output_bseq = tf.get_variable("output_bseq", [1])

    outputs = []
    y=[]
    state = self.initial_state
    with tf.variable_scope("rnnlm"):
      # self.input_data[:,time_step,:]
      for time_step in range(tf.shape(self.input_data)[1]):
        # if time_step > 0: tf.get_variable_scope().reuse_variables()
        # (cell_output, state) = cell(self.input_data[:,time_step,:], state)
        cell_output=tf.nn.relu(tf.add(tf.matmul(self.input_data[:,time_step,:], output_w1),output_b1))
        cell_output=tf.nn.relu(tf.add(tf.matmul(cell_output, output_w2),output_b2))
        cell_output=tf.nn.relu(tf.add(tf.matmul(cell_output, output_w3),output_b3))
        outputs.append(cell_output)
        # y.append(self.target_data[:,time_step,:])
    self.outputs=outputs
    print(outputs)
    outputs=tf.pack(axis=0,values= outputs)
    print(outputs)
    # rnn_output = tf.reshape(outputs, [-1, params['n_hidden']])
    # y_output = tf.reshape(tf.concat(1, y), [-1, params['n_hidden']])
    # print(rnn_output)

    # hidden_1 = tf.nn.relu(tf.add(tf.matmul(rnn_output, output_w1),output_b1))
    # self.final_output = tf.add(tf.matmul(hidden_1, output_w2),output_b2)
    # tmp=tf.reshape(self.final_output,shape=[params["batch_size"],params['seq_length'],NOUT])
    # print(tmp)
    # print(self.target_data)
    cost=0
    final_seq_lst=[]

    for t in range(tf.shape(self.input_data)[0]):
      seq=outputs[:,t,:]
      # target=self.y_output[t,:,:]
      # tseq=tf.transpose(seq)
      # final_seq=tf.transpose(tf.matmul(tseq, output_wseq))
      final_seq=tf.reduce_mean(seq,0,keep_dims=True)
      # final_seq= tf.gather(seq,49)
      final_seq_lst.append(final_seq)
    final_output=tf.squeeze(tf.pack(axis=0,values= final_seq_lst))
    # final_output=tf.squeeze(final_seq)
    print(final_output)
    print(self.target_data)

    self.y_tf=tf.argmax(self.target_data, 1)
    self.y_hat_softmax=tf.nn.softmax(final_output)
    print(self.y_hat_softmax)
    self.pred=tf.argmax(self.y_hat_softmax, 1)

    cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(dtype=tf.float32,x=self.target_data) * tf.log(self.y_hat_softmax), [1]))
    # cost=tf.nn.softmax_cross_entropy_with_logits(final_output,self.target_data)
    print(self.pred)
    print(self.target_data)
    accuracy =tf.reduce_sum(tf.cast(tf.equal(self.pred,self.y_tf), tf.float32))
    accuracy=accuracy/float(tf.shape(self.input_data)[0])

    # logits= tf.reshape(final_seq_lst, [-1])
    # y=tf.reshape(self.target_data, [-1])
    # print(logits)
    # print(y)
    # cost = tf.nn.seq2seq.sequence_loss_by_example(
    #     [logits],
    #     [y],
    #     [tf.ones([params["batch_size"] * params["seq_length"]], dtype=tf.float32)])
    # for t in range(params['batch_size']):
    #   seq=tmp[t,:,:]
    #   target=self.y_output[t,:,:]
    #   tseq=tf.transpose(seq)
    #   final_seq=tf.transpose(tf.matmul(tseq, output_wseq))
      # final_seq=tf.reduce_mean(seq,0,keep_dims=True)
      # final_seq_lst.append(final_seq)
      # sseq=tf.nn.softmax(final_seq)
      # if t==0:
      #   cost=tf.nn.softmax_cross_entropy_with_logits(final_seq,target)
      #   accuracy =tf.reduce_sum(tf.cast(tf.equal(tf.argmax(sseq, 1), tf.argmax(target, 1)), tf.float32))
      # else:
      #   cost=cost+tf.nn.softmax_cross_entropy_with_logits(final_seq,target)
      #   accuracy = accuracy+tf.reduce_sum(tf.cast(tf.equal(tf.argmax(sseq, 1), tf.argmax(target, 1)), tf.float32))
    #
    # accuracy=accuracy/float(params['batch_size'])

    # print(self.final_output)
    # print(self.target_data)
    # print(tf.argmax(self.final_output, 1))
    # print(tf.argmax(self.target_data, 1))

    # tmp = self.final_output - self.target_data
    # loss=  tf.nn.l2_loss(tmp)
    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(self.final_output, 1), tf.argmax(self.target_data, 1))
    # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    self.accuracy = accuracy
    # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.final_output, self.target_data))
    self.cost = tf.reduce_mean(cost)
    self.final_state = state
    tf.scalar_summary('losses/total_loss', self.cost)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads = tf.gradients(self.cost, tvars)
    # for grad in grads:
      # if isinstance(grad, ops.grads):
      #   grad_values = grad.values
      # else:
      #   grad_values = grad
      # grad_values = grad
      # logging_ops.histogram_summary(grad.op.name + ':gradient', grad_values)
      # logging_ops.histogram_summary(grad.op.name + ':gradient_norm', clip_ops.global_norm([grad_values]))
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

