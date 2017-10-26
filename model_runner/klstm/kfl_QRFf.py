import tensorflow as tf
from model_runner.common import rnn_cellv3 as rnncell

class Model(object):
    def __init__(self,params):

        num_layers = params['nlayer']
        rnn_size = params['n_hidden']
        grad_clip = params["grad_clip"]
        self.output_keep_prob = tf.placeholder(tf.float32)

        NOUT = params['n_output']

        # Transition LSTM
        cell = rnncell.ModifiedLSTMCell(rnn_size, forget_bias=1,initializer= tf.contrib.layers.xavier_initializer(),num_proj=None)
        cell = rnncell.MultiRNNCell([cell] * num_layers)
        cell = rnncell.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
        self.cell = cell

        # LSTM for Q noise
        cell_Q_noise =rnncell.ModifiedLSTMCell(params['Qn_hidden'], forget_bias=1,initializer= tf.contrib.layers.xavier_initializer(),num_proj=None)
        cell_Q_noise = rnncell.MultiRNNCell([cell_Q_noise] * params['Qnlayer'])
        cell_Q_noise = rnncell.DropoutWrapper(cell_Q_noise, output_keep_prob=self.output_keep_prob)
        self.cell_Q_noise = cell_Q_noise

        # LSTM for R noise
        cell_R_noise =rnncell.ModifiedLSTMCell(params['Rn_hidden'], forget_bias=1,initializer= tf.contrib.layers.xavier_initializer(),num_proj=None)
        cell_R_noise = rnncell.MultiRNNCell([cell_R_noise] * params['Rnlayer'])
        cell_R_noise = rnncell.DropoutWrapper(cell_R_noise, output_keep_prob=self.output_keep_prob)
        self.cell_R_noise = cell_R_noise

        self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_Q_noise = cell_Q_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_R_noise = cell_R_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"], params['seq_length']])

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
        pres_lst=[]
        tres_lst=[]
        kres_lst=[]
        with tf.variable_scope('rnnlm'):
          output_w1 = tf.get_variable("output_w", [rnn_size, NOUT],initializer= tf.contrib.layers.xavier_initializer())
          output_b1 = tf.get_variable("output_b", [NOUT])
          output_w1_Q_noise = tf.get_variable("output_w_Q_noise", [params['Qn_hidden'], NOUT],initializer= tf.contrib.layers.xavier_initializer())
          output_b1_Q_noise = tf.get_variable("output_b_Q_noise", [NOUT])
          output_w1_R_noise = tf.get_variable("output_w_R_noise", [params['Rn_hidden'], NOUT],initializer= tf.contrib.layers.xavier_initializer())
          output_b1_R_noise = tf.get_variable("output_b_R_noise", [NOUT])
          output_w1_A_mat = tf.get_variable("output_w1_A_mat", [params['Qn_hidden'], NOUT],initializer= tf.contrib.layers.xavier_initializer())
          output_b1_A_mat = tf.get_variable("output_b1_A_mat", [NOUT])
          # output_w1_H_mat = tf.get_variable("output_w1_H_mat", [rnn_size, NOUT])
          # output_b1_H_mat = tf.get_variable("output_b1_H_mat", [NOUT])

        state_F = self.initial_state
        state_Q = self.initial_state_Q_noise
        state_R = self.initial_state_R_noise
        with tf.variable_scope("rnnlm"):
            for time_step in range(params['seq_length']):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                z=self._z[:,time_step,:]
                if time_step == 0:
                    self._x= z
                    self._P=self._P_inp

                with tf.variable_scope("transitionF"):
                    (pred, state_F,ls_internals) = cell(self._x, state_F)
                    self._x  = tf.matmul(tf.squeeze(pred),output_w1)+output_b1

                with tf.variable_scope("noiseQ"):
                    (pred_val, state_Q,ls_internals) = cell_Q_noise(self._x, state_Q)
                    pred_Q_noise  = tf.matmul(tf.squeeze(pred_val),output_w1_Q_noise)+output_b1_Q_noise
                    pred_F_mat  = tf.matmul(tf.squeeze(pred_val),output_w1_A_mat)+output_b1_A_mat

                with tf.variable_scope("noiseR"):
                    (meas_val, state_R,ls_internals) = cell_R_noise(z, state_R)
                    pred_R_noise  = tf.matmul(tf.squeeze(meas_val),output_w1_R_noise)+output_b1_R_noise
                    # pred_H_mat  = tf.matmul(tf.squeeze(meas_val),output_w1_H_mat)+output_b1_H_mat

                # lst=tf.unpack(pred, axis=1)
                F=tf.matrix_diag(pred_F_mat)
                # H=tf.matrix_diag(tf.exp(pred_H_mat))

                Q=tf.matrix_diag(tf.exp(pred_Q_noise))
                R=tf.matrix_diag(tf.exp(pred_R_noise))

                #predict
                P = self._P
                self._P =tf.matmul(F,tf.matmul(P,tf.matrix_transpose(F)))+ Q

                #update
                P = self._P
                x = self._x
                self._y =z- x

                # S = HPH' + R
                # project system uncertainty into measurement space
                # S = tf.matmul(H,tf.matmul(P,tf.matrix_transpose(H))) + R
                S = P + R

                # K = PH'inv(S)
                # map system uncertainty into kalman gain
                # K = tf.matmul(P, tf.matmul(tf.matrix_transpose(H),tf.matrix_inverse(S)))
                K = tf.matmul(P, tf.matrix_inverse(S))

                # x = x + Ky
                # predict new x with residual scaled by the kalman gain
                self._x = x + tf.squeeze(tf.matmul(K, tf.expand_dims(self._y,2)),2)
                xres_lst.append(self._x)
                tres_lst.append(x)
                kres_lst.append(K)


                # P = (I-KH)P(I-KH)' + KRK'
                I_KH = self._I - K
                self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matmul(R, tf.matrix_transpose(K)))
                pres_lst.append(self._P)

                self._S = S
                self._K = K
        final_output = tf.reshape(tf.transpose(tf.pack(xres_lst), [1, 0, 2]), [-1, params['n_output']])
        flt = tf.squeeze(tf.reshape(self.repeat_data, [-1, 1]), [1])
        where_flt = tf.not_equal(flt, 0)
        indices = tf.where(where_flt)

        y = tf.reshape(self.target_data, [-1, params["n_output"]])
        self.final_output = tf.gather(final_output, tf.squeeze(indices, [1]))
        self.y = tf.gather(y, tf.squeeze(indices, [1]))

        tmp = self.final_output - self.y
        loss = tf.nn.l2_loss(tmp)
        self.tvars = tf.trainable_variables()
        l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
        l2_reg = tf.mul(l2_reg, 1e-4)
        self.cost = tf.reduce_mean(loss) + l2_reg
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

        self.states={}
        self.states["F_t"]=state_F
        self.states["Q_t"]=state_Q
        self.states["R_t"]=state_R
        self.xres_lst = xres_lst
        self.pres_lst = pres_lst
        self.tres_lst = tres_lst
        self.kres_lst = kres_lst
