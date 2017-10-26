from helper import utils as ut
from helper import dt_utils as dut

import tensorflow as tf

import numpy as np
import random

class Model():
  def __init__(self,params):

    self.is_training = tf.placeholder(tf.bool)
    num_layers=params['nlayer']
    rnn_size=params['n_hidden']
    self.output_keep_prob = tf.placeholder(tf.float32)
    grad_clip=10

    cell_fn = tf.nn.rnn_cell.BasicLSTMCell


    cell = cell_fn(rnn_size)#RNN size


    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)

    self.cell = cell
    NOUT = (params['n_output']*2+1)*30 # end_of_stroke + prob + 2*(mu + sig) + corr

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], 1024])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, params['seq_length'], params['n_output']])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    target_data = tf.split(1, params['seq_length'], self.target_data)
    self.pre_target_data = [tf.squeeze(y_, [1]) for y_ in target_data]
    self.final_target_data=tf.reshape(tf.concat(1, self.pre_target_data), [-1, params['n_output']])

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
    # target_data = tf.split(1, params['seq_length'], self.target_data)
    # self.pre_target_data = [tf.squeeze(y_, [1]) for y_ in target_data]
    # self.final_target_data=tf.reshape(tf.concat(1, self.pre_target_data), [-1, NOUT])
    #
    # self.flat_target_data = tf.reshape(self.final_target_data, [-1])
    # self.flat_output_data = tf.reshape(self.final_output, [-1])

    # [x1_data, x2_data, eos_data] = tf.split(1, 3, flat_target_data)

    # long method:
    #flat_target_data = tf.split(1, args.seq_length, self.target_data)
    #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]


    # self.cost = tf.reduce_sum(tf.square(self.flat_output_data - self.flat_target_data))
    #
    # self.lr = tf.Variable(0.0, trainable=False)
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    # optimizer = tf.train.AdamOptimizer(self.lr)
    # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def tf_2d_normal(x_lst,mu_lst,sigma_lst,rho):
      # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
      # norm1 = tf.sub(x1, mu1)
      # norm2 = tf.sub(x2, mu2)
      # s1s2 = tf.mul(s1, s2)

      norm_lst=[0.]*len(x_lst)
      for i in range(len(x_lst)):
        norm_lst[i]=tf.sub(x_lst[i],mu_lst[i])

      norm_div_lst_sum=0.
      for i in range(len(sigma_lst)):
        norm_div_lst_sum=norm_div_lst_sum+tf.square(tf.div(norm_lst[i],sigma_lst[i]))

      sls_lst_mul=1.
      for i in range(len(sigma_lst)):
        sls_lst_mul=tf.mul(sls_lst_mul,sigma_lst[i])

      norm_lst_mul=1.
      for i in range(len(norm_lst)):
        norm_lst_mul=tf.mul(norm_lst_mul,norm_lst[i])

      # z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
      z = norm_div_lst_sum-2*tf.div(tf.mul(rho, norm_lst_mul), sls_lst_mul)

      negRho = 1-tf.square(rho)
      result = tf.exp(tf.div(-z,2*negRho))
      # denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
      denom = 2*np.pi*tf.mul(norm_lst_mul, tf.sqrt(negRho))
      result = tf.div(result, denom)
      return result


    def get_Nd_loss(x_lst,mu_lst,sigma_lst,z_pi):
      ps = tf.exp(-(tf.pow(tf.sub(x_lst,mu_lst),2))/(2*tf.pow(sigma_lst,2)))/(tf.mul(sigma_lst,np.sqrt(2* np.pi))) #[nb x M]
      pin = ps * z_pi
      lp = -tf.log(tf.reduce_sum(pin, 1, keep_dims=True)) #[nb x 1] (sum across dimension 1)
      loss = tf.reduce_sum(lp) # scalar
      return loss

    def get_lossfunc(z_pi,z_corr, x_lst, mu_lst, sigma_lst):
      result0 = tf_2d_normal(x_lst, mu_lst, sigma_lst, z_corr)
      # implementing eq # 26 of http://arxiv.org/abs/1308.0850
      epsilon = 1e-20
      result1 = tf.mul(result0, z_pi)
      result2 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(tf.maximum(result2,epsilon)) # at the beginning, some errors are exactly zero.

      # result2 = tf.mul(z_eos, eos_data) + tf.mul(1-z_eos, 1-eos_data)
      # result2 = -tf.log(result2)
      # result = result1 + result2
      result = result1
      return [tf.reduce_sum(result),tf.reduce_sum(result2)]

    def NLL(mu, sigma, mixing, y):
      """Computes the mean of negative log likelihood for P(y|x)

      y = T.matrix('y') # (minibatch_size, output_size)
      mu = T.tensor3('mu') # (minibatch_size, output_size, n_components)
      sigma = T.matrix('sigma') # (minibatch_size, n_components)
      mixing = T.matrix('mixing') # (minibatch_size, n_components)
      """

      # multivariate Gaussian
      exponent = -0.5 * tf.inv(sigma) * tf.reduce_sum((y.dimshuffle(0,1,'x') - mu)**2, axis=1)
      normalizer = (2 * np.pi * sigma)
      exponent = exponent + tf.log(mixing) - (y.shape[1]*.5)*tf.log(normalizer)
      max_exponent = tf.max(exponent ,axis=1, keepdims=True)
      mod_exponent = exponent - max_exponent
      gauss_mix = tf.reduce_sum(tf.exp(mod_exponent),axis=1)
      log_gauss = max_exponent + tf.log(gauss_mix)
      res = -tf.reduce_mean(log_gauss)
      return res

    def get_test_loss(o_pi,o_mu_lst,o_sigma_lst):
      o_mu_lst=tf.pack(o_mu_lst)
      o_sigma_lst=tf.pack(o_sigma_lst)
      out_pi=tf.unpack(o_pi)
      out_mu=tf.unpack(tf.transpose(o_mu_lst,perm=(1, 2, 0)))#(150,10,48)
      out_sigma=tf.unpack(tf.transpose(o_sigma_lst,perm=(1, 2, 0)))#(150,10,48)
      m = o_pi.get_shape()[1]
      shp=o_mu_lst.get_shape()
      d=shp[2]
      n=shp[0]
      result = tf.zeros((n, d))

      for i in range(n):
        x=tf.random_uniform(shape=(1), minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
        #
        accumulate = 0.
        # c=0
        i_pi=tf.unpack(out_pi[i])
        unf=tf.random_uniform(shape=(1), minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
        c = tf.select(unf < i_pi[0], 0, tf.select(unf < i_pi[0]+ i_pi[1], 1,2 ))

        # c = int(np.random.choice(range(x), size=1, replace=True, p=i_pi))
        # for j in range(0, m):
        #   accumulate += i_pi[j]
        #   if tf.less(x,accumulate):
        #     c= i
        #     break

        # c = tf.int32(np.random.choice(range(m), size=1, replace=True, p=out_pi[i, :]))
        mu = tf.unpack(out_mu[i])[c]
        sig = tf.diag(tf.unpack(out_sigma[i])[c])
        sample_c = tf.contrib.distributions.MultivariateNormal(mu, sig ** 2, 1)
        result[i, :] = sample_c
      return result


    def get_mixture_coef(output):
        # returns the tf slices containing mdn dist params
        # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
        z = output
        # z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(1, 6,z)
        #48 =16*3, z_pi+2*48(mu, sigma)+z_corr
        z_list= tf.split(1, (params['n_output']*2+1),z)
        z_pi=z_list[0]
        z_mu_lst=z_list[1:49]
        z_sigma_lst=z_list[49:97]

        # process output z's into MDN paramters

        epsilon = 1e-20
        # softmax all the pi's:
        max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
        z_pi_1 = tf.sub(z_pi, max_pi)
        z_pi_2 = tf.exp(z_pi_1)
        normalize_pi = tf.inv(tf.reduce_sum(z_pi_2, 1, keep_dims=True))
        z_pi_3 = tf.mul(normalize_pi, z_pi_2)

        # exponentiate the sigmas and also make corr between -1 and 1.
        # for i in range(len(z_mu_lst)):
        #   z_mu_lst[i]=tf.exp(z_mu_lst[i])

        for i in range(len(z_sigma_lst)):
          z_sigma_lst[i]=tf.exp(z_sigma_lst[i])

        # z_sigma1 = tf.exp(z_sigma1)
        # z_sigma2 = tf.exp(z_sigma2)

        return [z_pi_3, z_mu_lst, z_sigma_lst]

    flat_target_data = tf.reshape(self.target_data,[-1, params['n_output']])
    x_lst = tf.split(1, params['n_output'], flat_target_data)
    # [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = get_mixture_coef(output)

    [o_pi,o_mu_lst, o_sigma_lst] = get_mixture_coef(self.final_output)

    # lossfunc,lossfunc2 = get_lossfunc(o_pi, o_corr, x_lst,o_mu_lst,o_sigma_lst)
    lossfunc = get_Nd_loss(x_lst,o_mu_lst,o_sigma_lst,o_pi)#/(params['batch_size']*params['seq_length'])
    lossfunc2=lossfunc
    # self.cost = lossfunc / (params["batch_size"] * params['seq_length'])
    self.cost = lossfunc
    self.cost2 = lossfunc2
    # self.test_result=get_test_loss(o_pi,o_mu_lst,o_sigma_lst)

    self.pi = o_pi
    self.mu_lst = o_mu_lst
    self.sigma_lst =o_sigma_lst




    # lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, x1_data, x2_data, eos_data)
    # self.cost = lossfunc / (args.batch_size * args.seq_length)
    #
    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))