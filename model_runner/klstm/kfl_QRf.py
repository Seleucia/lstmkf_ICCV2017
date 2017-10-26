import tensorflow as tf
from model_runner.common import rnn_cellv3 as rnncell
import numpy as np


class Model(object):
    def __init__(self,params,is_training = True):
        self.is_training=is_training
        batch_size=params["batch_size"]
        num_layers=params['nlayer']
        rnn_size=params['n_hidden']
        grad_clip = params["grad_clip"]
        self.output_keep_prob = tf.placeholder(tf.float32)
        self.input_keep_prob = tf.placeholder(tf.float32)

        NOUT = params['n_output']

        # Transition LSTM
        cell_lst = []
        for i in range(num_layers):
            cell = rnncell.ModifiedLSTMCell(rnn_size, forget_bias=1, initializer=tf.contrib.layers.xavier_initializer(),
                                            num_proj=None, is_training=self.is_training)
            if i > -1 and is_training == True:
                cell_drop = rnncell.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
                cell = cell_drop
            if i > 10 and params['input_keep_prob']<1:
                cell_drop = rnncell.DropoutWrapper(cell,input_keep_prob=self.input_keep_prob)
                cell = cell_drop
            cell_lst.append(cell)
        self.cell = rnncell.MultiRNNCell(cell_lst)

        # LSTM for Q noise
        cell_lst = []
        for i in range(params['Qnlayer']):
            cell_Q_noise = rnncell.ModifiedLSTMCell(params['Qn_hidden'], forget_bias=1, initializer=tf.contrib.layers.xavier_initializer(),
                                            num_proj=None, is_training=self.is_training)
            if i > -1 and is_training == True:
                cell_drop = rnncell.DropoutWrapper(cell_Q_noise, output_keep_prob=self.output_keep_prob)
                cell_Q_noise = cell_drop
            if i > 10 and params['input_keep_prob']<1:
                cell_drop = rnncell.DropoutWrapper(cell,input_keep_prob=self.input_keep_prob)
                cell = cell_drop
            cell_lst.append(cell_Q_noise)
        self.cell_Q_noise = rnncell.MultiRNNCell(cell_lst)

        # LSTM for R noise
        cell_lst = []
        for i in range(params['Rnlayer']):
            cell_R_noise = rnncell.ModifiedLSTMCell(params['Rn_hidden'], forget_bias=1,
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    num_proj=None, is_training=self.is_training)
            if i > -1 and is_training == True:
                cell_drop = rnncell.DropoutWrapper(cell_R_noise, output_keep_prob=self.output_keep_prob)
                cell_R_noise = cell_drop
            if i >10 and params['input_keep_prob']<1:
                cell_drop = rnncell.DropoutWrapper(cell,input_keep_prob=self.input_keep_prob)
                cell = cell_drop
            cell_lst.append(cell_R_noise)
        self.cell_R_noise = rnncell.MultiRNNCell(cell_lst)


        self.initial_state = self.cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_Q_noise = self.cell_Q_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        # self.initial_state_R_noise = self.cell_Q_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_R_noise = self.cell_R_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"], params['seq_length']])

        #Measurements
        self._z = tf.placeholder(dtype=tf.float32,
                                 shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature
        self._x_inp = tf.placeholder(dtype=tf.float32,
                                 shape=[None, NOUT],name='Initialx')  # batch size, seqlength, feature
        self.target_data = tf.placeholder(dtype=tf.float32,
                                          shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature
        self._P_inp = tf.placeholder(dtype=tf.float32, shape=[None, NOUT, NOUT], name='P')
        self._F = 0.0  # state transition matrix
        self._alpha_sq = 1.  # fading memory control
        self.M = 0.0  # process-measurement cross correlation
        self._I = tf.placeholder(dtype=tf.float32, shape=[None, NOUT, NOUT], name='I')
        self.u = 0.0

        xres_lst=[]
        xpred_lst=[]
        pres_lst=[]
        tres_lst=[]
        kres_lst=[]
        qres_lst=[]
        rres_lst=[]
        with tf.variable_scope('rnnlm'):
            output_w1 = tf.get_variable("output_w1", [rnn_size, rnn_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            output_b1 = tf.get_variable("output_b1", [rnn_size])
            output_w2 = tf.get_variable("output_w2", [rnn_size, rnn_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            output_b2 = tf.get_variable("output_b2", [rnn_size])
            output_w3 = tf.get_variable("output_w3", [rnn_size, NOUT],
                                        initializer=tf.contrib.layers.xavier_initializer())
            output_b3 = tf.get_variable("output_b3", [NOUT])

            output_w1_Q_noise = tf.get_variable("output_w_Q_noise", [params['Qn_hidden'], NOUT],
                                                initializer=tf.contrib.layers.xavier_initializer())
            output_b1_Q_noise = tf.get_variable("output_b_Q_noise", [NOUT])
            output_w1_R_noise = tf.get_variable("output_w_R_noise", [params['Rn_hidden'], NOUT],
                                                initializer=tf.contrib.layers.xavier_initializer())
            # output_b1_R_noise = tf.get_variable("output_b_R_noise", [NOUT],initializer=tf.ones_initializer())
            output_b1_R_noise = tf.get_variable("output_b_R_noise", [NOUT])
        #
        indices = list(zip(*np.tril_indices(NOUT)))
        indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

        state_F = self.initial_state
        state_Q = self.initial_state_Q_noise
        state_R = self.initial_state_R_noise
        with tf.variable_scope("rnnlm"):
            for time_step in range(params['seq_length']):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                z=self._z[:,time_step,:] #bs,features
                if time_step == 0:
                    # self._x= z
                    self._x= self._x_inp
                    self._P=self._P_inp
                with tf.variable_scope("transitionF"):
                    (pred, state_F, ls_internals)= self.cell(self._x, state_F)
                    # pred  = tf.matmul(pred,output_w1)+output_b1
                    pred = tf.nn.relu(tf.add(tf.matmul(pred, output_w1), output_b1))
                    pred = tf.nn.relu(tf.add(tf.matmul(pred, output_w2), output_b2))
                    pred = tf.add(tf.matmul(pred, output_w3), output_b3)

                with tf.variable_scope("noiseQ"):
                    (pred_Q_noise, state_Q, ls_internals) = self.cell_Q_noise(self._x, state_Q)
                    pred_Q_noise  = tf.matmul(pred_Q_noise,output_w1_Q_noise)+output_b1_Q_noise

                # one_mask = tf.ones(shape=(batch_size, NOUT))
                # zero_mask = tf.zeros(shape=(batch_size, NOUT))
                # random_mask = tf.random_uniform(shape=(batch_size, NOUT))
                # means = tf.mul(tf.ones(shape=(batch_size, NOUT)), 1 - self.output_keep_prob)
                # mask = tf.select(random_mask - means > 0.5, zero_mask, one_mask)
                # meas_z = tf.select(self.output_keep_prob >= 1, z, tf.mul(z, mask))
                # norm = tf.random_normal(shape=(batch_size, NOUT), mean=0, stddev=0.01)
                # meas_z = tf.select(self.output_keep_prob >= 1, z, tf.add(z, norm))
                meas_z=z

                with tf.variable_scope("noiseR"):
                    (pred_R_noise, state_R, ls_internals) = self.cell_R_noise(meas_z, state_R)
                    pred_R_noise  = tf.matmul(pred_R_noise,output_w1_R_noise)+output_b1_R_noise
                #
                self._x=pred

                # lst=tf.unpack(pred, axis=1)

                # Q= tf.sparse_to_dense(sparse_indices=indices, output_shape=[batch_size,NOUT, NOUT], \
                #                           sparse_values=pred_Q_noise, default_value=0, \
                #                           validate_indices=True)
                #
                Q=tf.matrix_diag(tf.exp(pred_Q_noise))
                R=tf.matrix_diag(tf.exp(pred_R_noise))
                # Q=tf.matmul(tf.matrix_diag(tf.exp(pred_Q_noise)),tf.matrix_diag(tf.exp(pred_Q_noise)))
                # R=tf.matmul(tf.matrix_diag(tf.exp(pred_R_noise)),tf.matrix_diag(tf.exp(pred_R_noise)))

                #predict
                P = self._P
                self._P = P + Q

                #update
                P = self._P
                x = self._x

                self._y =meas_z- x

                # S = HPH' + R
                # project system uncertainty into measurement space
                S = P + R
                # S = P

                # K = PH'inv(S)
                # map system uncertainty into kalman gain
                K = tf.matmul(P, tf.matrix_inverse(S)) #(Q+P_init/(R+Q+P_init))

                # x = x + Ky
                # predict new x with residual scaled by the kalman gain
                self._x = tf.squeeze(tf.matmul(K, tf.expand_dims(self._y,2)),-1) #K-->>1, _x=z, K-->>0, _x=x,
                xpred_lst.append(x)
                xres_lst.append(self._x)
                tres_lst.append(meas_z)
                kres_lst.append(tf.matrix_diag_part(K))
                rres_lst.append(tf.matrix_diag_part(R))
                qres_lst.append(tf.matrix_diag_part(Q))


                # P = (I-KH)P(I-KH)' + KRK'
                I_KH = self._I - K
                self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matmul(R, tf.matrix_transpose(K)))
                # self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matrix_transpose(K))

                self._S = S
                self._K = K
        final_output = tf.reshape(tf.transpose(tf.stack(xres_lst),[1,0,2]), [-1, params['n_output']])
        final_pred_output = tf.reshape(tf.transpose(tf.stack(xpred_lst),[1,0,2]), [-1, params['n_output']])
        final_q_output = tf.reshape(tf.transpose(tf.stack(qres_lst),[1,0,2]), [-1, params['n_output']])
        final_r_output = tf.reshape(tf.transpose(tf.stack(rres_lst),[1,0,2]), [-1, params['n_output']])
        final_k_output = tf.reshape(tf.transpose(tf.stack(kres_lst),[1,0,2]), [-1, params['n_output']])
        final_meas_output = tf.reshape(tf.transpose(tf.stack(tres_lst),[1,0,2]), [-1, params['n_output']])
        flt=tf.squeeze(tf.reshape(self.repeat_data,[-1,1]),[1])
        where_flt=tf.not_equal(flt,0)
        indices=tf.where(where_flt)

        y=tf.reshape(self.target_data,[-1,params["n_output"]])
        self.final_output=tf.gather(final_output,tf.squeeze(indices,[1]))
        self.final_pred_output=tf.gather(final_pred_output,tf.squeeze(indices,[1]))
        self.final_q_output=tf.gather(final_q_output,tf.squeeze(indices,[1]))
        self.final_r_output=tf.gather(final_r_output,tf.squeeze(indices,[1]))
        self.final_k_output=tf.gather(final_k_output,tf.squeeze(indices,[1]))
        self.final_meas_output=tf.gather(final_meas_output,tf.squeeze(indices,[1]))
        self.y=tf.gather(y,tf.squeeze(indices,[1]))

        tmp = self.final_output - self.y
        loss = tf.nn.l2_loss(tmp)
        tmp_pred = self.final_pred_output - self.y
        loss_pred = tf.nn.l2_loss(tmp_pred)

        # tmp_pred = self.final_pred_output - self.y
        # loss_pred = tf.nn.l2_loss(tmp_pred)

        self.tvars = tf.trainable_variables()
        l2_reg=tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
        l2_reg=tf.multiply(l2_reg,1e-4)
        self.cost = tf.reduce_mean(loss)+l2_reg+0.8*tf.reduce_mean(loss_pred)
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        total_parameters=0
        for variable in self.tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        self.total_parameters=total_parameters
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.states = {}
        self.states["F_t"] = state_F
        self.states["Q_t"] = state_Q
        self.states["R_t"] = state_R
        self.states["PCov_t"] = self._P
        self.states["_x_t"] = self._x
        self.xres_lst=xres_lst
        self.pres_lst=pres_lst
        self.tres_lst=tres_lst
        self.kres_lst=kres_lst