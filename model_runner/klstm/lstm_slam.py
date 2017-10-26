import tensorflow as tf
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import logging_ops

from model_runner.common import rnn_cellv3 as rnncell
import math

class Model():
  def __init__(self,params, is_training=True):

    self.is_training = tf.placeholder(tf.bool)
    self.output_keep_prob = tf.placeholder(tf.float32)

    num_layers=params['nlayer']
    rnn_size=params['n_hidden']
    grad_clip = params["grad_clip"]

    cell_lst = []
    for i in range(num_layers):
      cell =  tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1,initializer=tf.orthogonal_initializer())
      if i>10 and is_training==True:
        cell_drop = rnncell.DropoutWrapper(cell,output_keep_prob= self.output_keep_prob)
        cell=cell_drop
      cell_lst.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cell_lst)

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[params["batch_size"],params['seq_length'], params['n_input']],name='input_data')
    self.input_zero = tf.placeholder(dtype=tf.float32, shape=[params["batch_size"],params['seq_length'], params['n_input']],name='input_zero')
    self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"],params['seq_length']],name='repeat_data')
    self.target_data =tf.placeholder(tf.float32, [None,None,params["n_output"]])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)


    outputs = []
    state = self.initial_state
    pre_state=state
    seq_ls_internal=[]
    with tf.variable_scope("rnnlm"):
      for time_step in range(params['seq_length']):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        # r=self.repeat_data[:,time_step]
        # where_flt=tf.not_equal(r,0)
        (cell_output, state) = cell(self.input_data[:,time_step,:], state)
        outputs.append(cell_output)
    rnn_output = tf.reshape(tf.transpose(tf.stack(outputs),[1,0,2]), [-1, params['n_hidden']])

    with tf.variable_scope('rnnlm'):
      output_w1 = tf.get_variable("output_w1", [rnn_size, NOUT],initializer=tf.orthogonal_initializer())
      output_b1 = tf.get_variable("output_b1", [NOUT])

    final_output = tf.add(tf.matmul(rnn_output, output_w1),output_b1)
    self.seq_ls_internal=seq_ls_internal

    flt=tf.squeeze(tf.reshape(self.repeat_data,[-1,1]),[1])
    where_flt=tf.not_equal(flt,0)
    indices=tf.where(where_flt)

    y=tf.reshape(self.target_data,[-1,params["n_output"]])
    self.final_output=tf.gather(final_output,tf.squeeze(indices,[1]))
    self.y=tf.gather(y,tf.squeeze(indices,[1]))

    if params["data_mode"]=="q":
            pose_q=self.y
            predicted_q=self.final_output
            q1=tf.divide(pose_q,tf.expand_dims(tf.norm(pose_q,axis=1),1))
            q2=tf.divide(predicted_q,tf.expand_dims(tf.norm(predicted_q,axis=1),1))
            d = tf.abs(tf.reduce_sum(tf.multiply(q1,q2),axis=1))
            # theta = 2 * tf.acos(d) * 180/math.pi
            theta = tf.acos(d)
            loss=theta
    else:
        tmp = self.final_output - self.y
        loss = tf.nn.l2_loss(tmp)
        # pose_q=tf.stack(tf.unstack(self.y,axis=1)[3:7],axis=1)
        # predicted_q=tf.stack(tf.unstack(self.final_output,axis=1)[3:7],axis=1)
        # q1=tf.divide(pose_q,tf.expand_dims(tf.norm(pose_q,axis=1),1))
        # q2=tf.divide(predicted_q,tf.expand_dims(tf.norm(predicted_q,axis=1),1))
        # d = tf.abs(tf.reduce_sum(tf.multiply(q1,q2),axis=1))
        # theta = 2*tf.acos(d) * 180/math.pi
        # # loss=0.02*theta_loss+loss
        # pose_x=tf.stack(tf.unstack(self.y,axis=1)[0:3],axis=1)
        # predicted_x=tf.stack(tf.unstack(self.final_output,axis=1)[0:3],axis=1)
        # error_x = tf.norm(pose_x-predicted_x,axis=1)
        # loss=error_x+1000000.0*theta




    # tmp = self.final_output - self.y
    # loss=  tf.nn.l2_loss(tmp)
    self.tvars = tf.trainable_variables()
    l2_reg=tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
    l2_reg=tf.multiply(l2_reg,1e-4)
    self.cost = tf.reduce_mean(loss)+l2_reg
    self.states = {}
    self.states["lstm_t"] = state

    self.lr = tf.Variable(0.0, trainable=False)
    total_parameters = 0
    for variable in self.tvars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    self.total_parameters=total_parameters
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, self.tvars))