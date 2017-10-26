import tensorflow as tf
from model_runner.common.mdn_layer import MDN
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import logging_ops
import locale
locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

from model_runner.common import rnn_cellv3 as rnncell


class Model():

  def __init__(self,params, infer=False):


    self.is_training = tf.placeholder(tf.bool)
    self.output_keep_prob = tf.placeholder(tf.float32)

    num_layers=params['nlayer']
    rnn_size=params['n_hidden']
    grad_clip=10
    mode=1
    KMIX = params['KMIX'] # end_of_stroke + prob + 2*(mu + sig) + corr
    if mode==0:
        NOUT =KMIX+ KMIX * params['n_output']+ KMIX *params['n_output']# pi, mu, stdev
    else:
        NOUT =KMIX+ KMIX * (params['n_output'])+ KMIX# pi, mu, stdev

    mdn=MDN(n_output=params['n_output'],NOUT=NOUT,KMIX=KMIX,mode=mode)
    cell_lst=[]
    # p1  = tf.get_variable("p1", [1])
    # p2  = tf.get_variable("p2", [1])
    # p3  = tf.get_variable("p3", [1])
    # p_lst=[]
    # p_lst.append(p1)
    # p_lst.append(p2)
    # p_lst.append(p3)
    for i in range(num_layers):
      cell =  rnncell.LSTMCell(rnn_size, initializer= tf.contrib.layers.xavier_initializer()
                                   , forget_bias=1.0,num_proj=None)
      # if i==0:
      #   cell_drop = rnncell.DropoutWrapper(cell,input_keep_prob= 0.8)
      #   cell=cell_drop
      if i>-1:
        cell_drop = rnncell.DropoutWrapper(cell,output_keep_prob= self.output_keep_prob)
        cell=cell_drop
      cell_lst.append(cell)

    cell = rnncell.MultiRNNCell(cell_lst)
    self.cell = cell


    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.input_zero = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_input']])
    self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[None, params['seq_length']])
    self.target_data =tf.placeholder(tf.float32, [None,params["seq_length"],params["n_output"]])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    #Noise applied only training phase and if only std bigger than 0
    if(params["noise_std"]>0.0):
      ran_noise = tf.random_normal(shape=[params["batch_size"],params['seq_length'], params['n_input']], mean=0, stddev=params['noise_std'])
      # ran_noise=tf.mul(ran_noise,self.input_zero)
      # tmp_input=tf.nn.relu(self.input_data+ran_noise)
      tmp_input=self.input_data+ran_noise
      self.input_data=tf.select(self.is_training,tmp_input,self.input_data)

    outputs = []
    state = self.initial_state
    pre_state=state
    seq_ls_internal=[]
    with tf.variable_scope("rnnlm"):
      for time_step in range(params['seq_length']):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        r=self.repeat_data[:,time_step]
        where_flt=tf.not_equal(r,0)
        (cell_output, state,ls_internals) = cell(self.input_data[:,time_step,:], state)
        seq_ls_internal.append(ls_internals)
        new_state=[]
        for i in range(params['nlayer']):
          s=[]
          s.append(tf.select(where_flt,state[i][0],pre_state[i][0]))
          s.append(tf.select(where_flt,state[i][1],pre_state[i][1]))
          new_state.append(tuple(s))
        state=new_state
        pre_state=state

        outputs.append(cell_output)
    rnn_output = tf.reshape(tf.transpose(tf.pack(outputs),[1,0,2]), [-1, params['n_hidden']])

    with tf.variable_scope('rnnlm'):
      output_w1 = tf.get_variable("output_w1", [rnn_size, rnn_size],initializer=tf.contrib.layers.xavier_initializer() )
      output_b1 = tf.get_variable("output_b1", [rnn_size])
      output_w2 = tf.get_variable("output_w2", [rnn_size, NOUT],initializer=tf.contrib.layers.xavier_initializer() )
      output_b2 = tf.get_variable("output_b2", [NOUT])

    output = tf.add(tf.matmul(rnn_output, output_w1),output_b1)
    final_output = tf.add(tf.matmul(output, output_w2),output_b2)

    flt=tf.squeeze(tf.reshape(self.repeat_data,[-1,1]),[1])
    where_flt=tf.not_equal(flt,0)
    indices=tf.where(where_flt)
    y=tf.reshape(self.target_data,[-1,params["n_output"]])

    self.seq_ls_internal=seq_ls_internal
    self.final_output=tf.gather(final_output,tf.squeeze(indices,[1]))
    self.y=tf.gather(y,tf.squeeze(indices,[1]))
    # self.cost = tf.reduce_mean(loss)

    # loss,result_sum,weighted_ker,pack_weighted_nom,exp_kr,pack_ker_rslt=mdn.get_lossfunc(final_output=self.final_output,y=y)
    # self.cost = loss
    # self.result_sum = result_sum
    # self.weighted_ker = weighted_ker
    # self.pack_weighted_nom = pack_weighted_nom
    # self.exp_kr = exp_kr
    # self.pack_ker_rslt = pack_ker_rslt
    # self.result_sum = 0
    # self.weighted_ker = 0
    # self.pack_weighted_nom = 0
    # self.exp_kr = 0
    # self.pack_ker_rslt = 0
    self.tvars = tf.trainable_variables()
    l2_reg=tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
    l2_reg=tf.mul(l2_reg,1e-4)
    loss=mdn.get_multi_normal_loss(final_output=self.final_output,y=self.y)
    self.l2loss=0
    self.loss=loss
    self.cost = loss+l2_reg
    out_pi, out_sigma, out_mu=mdn.get_mixture_coef(final_output=self.final_output)
    self.out_pi=out_pi
    self.out_sigma=out_sigma
    self.out_mu=out_mu
    self.final_state = state
    tf.scalar_summary('losses/total_loss', loss)

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
    for grad in grads:
      grad_values = grad
      logging_ops.histogram_summary(grad.op.name + ':gradient', grad_values)
      logging_ops.histogram_summary(grad.op.name + ':gradient_norm', clip_ops.global_norm([grad_values]))
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, self.tvars))