import tensorflow as tf
import math

# from model_runner.common import rnn_cellv3 as rnncell


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
        dropout_start=10
        cell_lst = []
        for i in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=5, initializer=tf.orthogonal_initializer())
            if i > dropout_start and is_training == True:
                cell_drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
                cell = cell_drop
            cell_lst.append(cell)
        self.cell = tf.contrib.rnn.MultiRNNCell(cell_lst)

        # LSTM for Q noise
        cell_lst = []
        for i in range(params['Qnlayer']):
            cell_Q_noise = tf.contrib.rnn.LSTMCell(params['Qn_hidden'],
                                                        forget_bias=1, initializer=tf.orthogonal_initializer())
            if i > dropout_start and is_training == True:
                cell_drop = tf.contrib.rnn.DropoutWrapper(cell_Q_noise, output_keep_prob=self.output_keep_prob)
                cell_Q_noise = cell_drop
            cell_lst.append(cell_Q_noise)
        self.cell_Q_noise = tf.contrib.rnn.MultiRNNCell(cell_lst)

        # LSTM for R noise
        cell_lst = []
        for i in range(params['Rnlayer']):
            cell_R_noise = tf.contrib.rnn.LSTMCell(params['Rn_hidden'], forget_bias=1,
                                                    initializer=tf.orthogonal_initializer())
            if i > dropout_start and is_training == True:
                cell_drop = tf.contrib.rnn.DropoutWrapper(cell_R_noise, output_keep_prob=self.output_keep_prob)
                cell_R_noise = cell_drop
            cell_lst.append(cell_R_noise)
        self.cell_R_noise = tf.contrib.rnn.MultiRNNCell(cell_lst)


        self.initial_state = self.cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_Q_noise = self.cell_Q_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        # self.initial_state_R_noise = self.cell_Q_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_R_noise = self.cell_R_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"], params['seq_length']])

        #Measurements
        self._z = tf.placeholder(dtype=tf.float32,
                                 shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature
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
        with tf.variable_scope('rnnlm'):
            # output_w1 = tf.get_variable("output_w", [rnn_size, NOUT],
            #                             initializer=tf.orthogonal_initializer())
            # output_b1 = tf.get_variable("output_b", [NOUT])
            #
            output_w1 = tf.get_variable("output_w", [rnn_size, rnn_size],
                                        initializer=tf.orthogonal_initializer())
            output_b1 = tf.get_variable("output_b", [rnn_size])

            output_w1_pre = tf.get_variable("output_w_pre", [rnn_size, NOUT],
                                        initializer=tf.orthogonal_initializer())
            output_b1_pre = tf.get_variable("output_b_pre", [NOUT])
            #
            # output_w2_pre = tf.get_variable("output_w2_pre", [rnn_size, NOUT],
            #                             initializer=tf.orthogonal_initializer())
            # output_b2_pre = tf.get_variable("output_b2_pre", [NOUT])
            #
            # output_w3_pre = tf.get_variable("output_w3_pre", [rnn_size, rnn_size],
            #                             initializer=tf.contrib.layers.xavier_initializer())
            # output_b3_pre = tf.get_variable("output_b3_pre", [rnn_size])


            # output_w1 = tf.get_variable("output_w", [rnn_size, rnn_size],
            #                             initializer=tf.contrib.layers.xavier_initializer())
            # output_b1 = tf.get_variable("output_b", [rnn_size])
            #
            # output_w2 = tf.get_variable("output_w2", [rnn_size, rnn_size],
            #                             initializer=tf.contrib.layers.xavier_initializer())
            # output_b2 = tf.get_variable("output_b2", [rnn_size])
            #
            # output_w3 = tf.get_variable("output_w3", [rnn_size, NOUT],
            #                             initializer=tf.contrib.layers.xavier_initializer())
            # output_b3 = tf.get_variable("output_b3", [NOUT])

            output_w1_Q_noise = tf.get_variable("output_w_Q_noise", [params['Qn_hidden'], NOUT],
                                                initializer=tf.orthogonal_initializer())
            output_b1_Q_noise = tf.get_variable("output_b_Q_noise", [NOUT])
            output_w1_R_noise = tf.get_variable("output_w_R_noise", [params['Rn_hidden'], NOUT],
                                                initializer=tf.orthogonal_initializer())
            output_b1_R_noise = tf.get_variable("output_b_R_noise", [NOUT])


        state_F = self.initial_state
        state_Q = self.initial_state_Q_noise
        state_R = self.initial_state_R_noise
        qnoise_lst=[]
        with tf.variable_scope("rnnlm"):
            for time_step in range(params['seq_length']):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                z=self._z[:,time_step,:] #bs,features
                if time_step == 0:
                    self._x= z
                    self._P=self._P_inp

                with tf.variable_scope("transitionF"):
                    (pred, state_F)= self.cell(self._x, state_F)
                    pred  = tf.matmul(pred,output_w1)+output_b1
                    pred  = tf.matmul(pred,output_w1)+output_b1
                    pred  = tf.matmul(pred,output_w1_pre)+output_b1_pre
                    # pred  = tf.matmul(pred,output_w2_pre)+output_b2_pre
                    # pred  = tf.matmul(pred,output_w3)+output_b3

                with tf.variable_scope("noiseQ"):
                    (pred_Q_noise, state_Q) = self.cell_Q_noise(pred, state_Q)
                    pred_Q_noise  = tf.matmul(pred_Q_noise,output_w1_Q_noise)+output_b1_Q_noise

                with tf.variable_scope("noiseR"):
                    (pred_R_noise, state_R) = self.cell_R_noise(z, state_R)
                    pred_R_noise  = tf.matmul(pred_R_noise,output_w1_R_noise)+output_b1_R_noise

                self._x=pred

                # lst=tf.unpack(pred, axis=1)
                qnoise_lst.append(pred_Q_noise)
                Q=tf.matrix_diag(tf.exp(pred_Q_noise))
                R=tf.matrix_diag(tf.exp(pred_R_noise))

                #predict
                P = self._P
                self._P = P + Q

                #update
                P = self._P
                x = self._x
                self._y =z- x

                # S = HPH' + R
                # project system uncertainty into measurement space
                S = P + R
                # S = P

                # K = PH'inv(S)
                # map system uncertainty into kalman gain
                K = tf.matmul(P, tf.matrix_inverse(S))

                # x = x + Ky
                # predict new x with residual scaled by the kalman gain
                self._x = x +  tf.squeeze(tf.matmul(K, tf.expand_dims(self._y,2)))
                xpred_lst.append(x)
                # xres_lst.append(self._x)
                xres_lst.append(self._x)
                tres_lst.append(x)
                kres_lst.append(K)


                # P = (I-KH)P(I-KH)' + KRK'
                I_KH = self._I - K
                self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matmul(R, tf.matrix_transpose(K)))
                # self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matrix_transpose(K))

                self._S = S
                self._K = K
        # final_output=tf.transpose(tf.stack(xres_lst),[1,0,2])
        final_output=tf.transpose(tf.stack(xres_lst),[1,0,2])
        final_output_pred=tf.transpose(tf.stack(xpred_lst),[1,0,2])
        final_output = tf.reshape(final_output, [-1, params['n_output']])
        final_output_pred = tf.reshape(final_output_pred, [-1, params['n_output']])
        qnoise_lst = tf.reshape(tf.transpose(tf.stack(qnoise_lst),[1,0,2]), [-1, params['n_output']])
        final_pred_output = tf.reshape(tf.transpose(tf.stack(xpred_lst),[1,0,2]), [-1, params['n_output']])
        flt=tf.squeeze(tf.reshape(self.repeat_data,[-1,1]),[1])
        where_flt=tf.not_equal(flt,0)
        indices=tf.where(where_flt)

        y=tf.reshape(self.target_data,[-1,params["n_output"]])
        x=tf.reshape(self._z,[-1,params["n_output"]])
        self.final_output=tf.gather(final_output,tf.squeeze(indices,[1]))
        self.final_output_pred=tf.gather(final_output_pred,tf.squeeze(indices,[1]))
        self.full_final_output=final_output
        self.final_pred_output=final_pred_output
        self.y=tf.gather(y,tf.squeeze(indices,[1]))
        self.x=tf.gather(x,tf.squeeze(indices,[1]))
        self.qnoise_lst=qnoise_lst

        # tmp = self.final_output - self.y
        # loss = tf.nn.l2_loss(tmp)
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

            tmp_pred = self.final_output_pred - self.y
            loss_pred = tf.nn.l2_loss(tmp_pred)
            # pose_q=tf.stack(tf.unstack(self.y,axis=1)[3:7],axis=1)
            # predicted_q=tf.stack(tf.unstack(self.final_output,axis=1)[3:7],axis=1)
            # q1=tf.divide(pose_q,tf.expand_dims(tf.norm(pose_q,axis=1),1))
            # q2=tf.divide(predicted_q,tf.expand_dims(tf.norm(predicted_q,axis=1),1))
            # d = tf.abs(tf.reduce_sum(tf.multiply(q1,q2),axis=1))
            # theta = 2 * tf.acos(d) * 180/math.pi
            # # theta = d
            # # # loss=0.02*theta_loss+loss
            # pose_x=tf.stack(tf.unstack(self.y,axis=1)[0:3],axis=1)
            # predicted_x=tf.stack(tf.unstack(self.final_output,axis=1)[0:3],axis=1)
            # error_x = tf.norm(pose_x-predicted_x,axis=1)
            # loss=error_x+100*theta

        # tf.norm(tf.stack(tf.unpack(self.y,axis=1)[3:7],axis=1),axis=1)


        self.tvars = tf.trainable_variables()
        l2_reg=tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
        l2_reg=tf.multiply(l2_reg,1e-5)
        self.cost = tf.reduce_sum(loss)+l2_reg+0.8*tf.reduce_sum(loss_pred)
        # self.cost = tf.reduce_sum(loss)+l2_reg
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
        self.xres_lst=xres_lst
        self.pres_lst=pres_lst
        self.tres_lst=tres_lst
        self.kres_lst=kres_lst