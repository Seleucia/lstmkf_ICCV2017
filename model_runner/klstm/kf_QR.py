import tensorflow as tf
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from model_runner.common import rnn_cellv3 as rnncell

class Model(object):

    def __init__(self, params):
        def identity_matrix(bs,n):
          diag=tf.diag(tf.Variable(initial_value=[1]*n,dtype=tf.float32))
          lst=[]
          for i in range(bs):
              lst.append(diag)
          return tf.pack(lst)

        num_layers=params['nlayer']
        F_shape=params['F_shape']
        H_shape=params['H_shape']
        rnn_size=params['n_hidden']
        NOUT = params['n_output']
        batch_size=params["batch_size"]
        self.output_keep_prob = tf.placeholder(tf.float32)
        grad_clip=params["grad_clip"]

        self._x = tf.zeros((batch_size,F_shape[1])) # state
        # self._P_inp = tf.placeholder(dtype=tf.float32, shape=[None, F_shape[1], F_shape[1]],name='P')
        self._P = identity_matrix(batch_size,F_shape[1]) # uncertainty covariance
        # self._I = tf.placeholder(dtype=tf.float32, shape=[None, NOUT, NOUT],name='I')
        # self._P = identity_matrix(bs,dim_x)* 500. # uncertainty covariance
        # self._Q = identity_matrix(bs,dim_x) # process uncertainty
        B = 0.0                # control transition matrix
        u=0.0
        self._F = 0.0                # state transition matrix
        self._alpha_sq = 1.        # fading memory control
        self.M = 0.0                 # process-measurement cross correlation
        self._I = identity_matrix(batch_size,F_shape[1])


        # LSTM for Q noise
        cell_Q_noise = rnncell.ModifiedLSTMCell(params['Qn_hidden'], forget_bias=1,
                                                initializer=tf.contrib.layers.xavier_initializer(), num_proj=None)
        cell_Q_noise = rnncell.MultiRNNCell([cell_Q_noise] * params['Qnlayer'])
        cell_Q_noise = rnncell.DropoutWrapper(cell_Q_noise, output_keep_prob=self.output_keep_prob)
        self.cell_Q_noise = cell_Q_noise

        # LSTM for R noise
        cell_R_noise = rnncell.ModifiedLSTMCell(params['Rn_hidden'], forget_bias=1,
                                                initializer=tf.contrib.layers.xavier_initializer(), num_proj=None)
        cell_R_noise = rnncell.MultiRNNCell([cell_R_noise] * params['Rnlayer'])
        cell_R_noise = rnncell.DropoutWrapper(cell_R_noise, output_keep_prob=self.output_keep_prob)
        self.cell_R_noise = cell_R_noise

        self.initial_state_Q_noise = cell_Q_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_R_noise = cell_R_noise.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[params["batch_size"],params['seq_length'], NOUT],name="input_data")
        self.input_zero = tf.placeholder(dtype=tf.float32, shape=[params["batch_size"],params['seq_length'], params['n_input']],name="input_zero")
        self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"],params['seq_length']],name="repeat_data")
        self.target_data =tf.placeholder(tf.float32, [None,params['seq_length'],NOUT],name="target_data")


        self.F= tf.placeholder(dtype=tf.float32, shape=F_shape) #batch size, seqlength, feature
        self.H= tf.placeholder(dtype=tf.float32, shape=H_shape) #batch size, seqlength, feature
        # dt = 1.0   # time step
        # F = tf.Variable(initial_value=[[1, dt, 0,  0],
        #                                [0,  1, 0,  0],
        #                                [0,  0, 1, dt],
        #                                [0,  0, 0,  1]],dtype=tf.float32)
        #
        # self.u = 0.0
        # H = tf.Variable(initial_value=[[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]],dtype=tf.float32)              # Measurement function
        #
        # F=tf.pack([F]*bs)
        # H=tf.pack([H]*bs)

        with tf.variable_scope('rnnlm'):
          output_w1_Q_noise = tf.get_variable("output_w_Q_noise", [rnn_size, F_shape[1]],initializer=tf.contrib.layers.xavier_initializer() )
          output_b1_Q_noise = tf.get_variable("output_b_Q_noise", [F_shape[1]])
          output_w1_R_noise = tf.get_variable("output_w_R_noise", [rnn_size, NOUT],initializer=tf.contrib.layers.xavier_initializer() )
          output_b1_R_noise = tf.get_variable("output_b_R_noise", [NOUT])

        xres_lst=[]
        pres_lst=[]
        tres_lst=[]
        qres_lst=[]
        rres_lst=[]

        outputs = []
        state_Q = self.initial_state_Q_noise
        state_R = self.initial_state_R_noise
        for time_step in range(params['seq_length']):
            z=self.input_data[:,time_step,:] #bs,features
            if time_step > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("noiseQ"):
                (pred_Q_noise, state_Q, ls_internals) = cell_Q_noise(self._x, state_Q)
                pred_Q_noise  = tf.matmul(pred_Q_noise,output_w1_Q_noise)+output_b1_Q_noise

            with tf.variable_scope("noiseR"):
                (pred_R_noise, state_R, ls_internals) = cell_R_noise(z, state_R)
                pred_R_noise  = tf.matmul(pred_R_noise,output_w1_R_noise)+output_b1_R_noise

             # lst=tf.unpack(pred, axis=1)
            Q=tf.matrix_diag(tf.exp(pred_Q_noise))
            R=tf.matrix_diag(tf.exp(pred_R_noise))
            # R=tf.matmul(R,tf.matrix_transpose(R))
            # Q=tf.matmul(Q,tf.matrix_transpose(Q))
            qres_lst.append( self._P)
            #predict
            P = self._P
            # x = tf.expand_dims(tf.matmul(tf.squeeze(self._x),output_w1)+output_b1,2)
            # x = tf.expand_dims(self._x)
            x = self._x
            # x = Fx + Bu
            self._x = tf.matmul(self.F, tf.expand_dims(x,2))+ tf.mul(B, u)
            # P = FPF' + Q
            self._P = self._alpha_sq * tf.matmul(self.F,tf.matmul(P, tf.matrix_transpose(self.F))) + Q
            # self._P = self._alpha_sq * dot3(F, self._P, F.T) + Q



            #update
            P = self._P
            x = self._x

            # Hx = tf.matmul(H, x)
            Hx = tf.matmul(self.H, x)
            self._y =tf.expand_dims(z,2)- Hx

            # S = HPH' + R
            # project system uncertainty into measurement space
            S = tf.matmul(self.H, tf.matmul(P, tf.matrix_transpose(self.H))) + R

            # K = PH'inv(S)
            # map system uncertainty into kalman gain
            K = tf.matmul(P, tf.matmul(tf.matrix_transpose(self.H), tf.matrix_inverse(S)))

            # x = x + Ky
            # predict new x with residual scaled by the kalman gain
            self._x = tf.squeeze(x)+ tf.squeeze(tf.matmul(K, self._y))
            xres_lst.append(self._x)


            # P = (I-KH)P(I-KH)' + KRK'
            I_KH = self._I - tf.matmul(K, self.H)
            self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matmul(R, tf.matrix_transpose(K)))
            pres_lst.append(P)
            rres_lst.append(R)


            # self._S = S
            # self._K = K
        # rnn_output = tf.reshape(tf.concat(1, outputs), [-1, params['n_hidden']])
        test_mode=params['test_mode']
        if test_mode=='step2d':
            final_output=tf.pack([tf.transpose(tf.squeeze(xres_lst), (1, 0, 2))[:,:,0],tf.transpose(tf.squeeze(xres_lst), (1, 0, 2))[:,:,2]],axis=2)
        else:
            final_output=tf.transpose(tf.squeeze(xres_lst), (1, 0, 2))


        self.y=tf.reshape(self.target_data,[-1,params["n_output"]])
        self.final_output=tf.reshape(final_output,[-1,params["n_output"]])

        tmp = self.final_output - self.y
        loss=  tf.nn.l2_loss(tmp)
        self.tvars = tf.trainable_variables()
        l2_reg=tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
        l2_reg=tf.mul(l2_reg,1e-4)
        self.cost = tf.reduce_mean(loss)+l2_reg
        self.lr = tf.Variable(0.0, trainable=False)

        self.states = {}
        self.states["Q_t"] = state_Q
        self.states["R_t"] = state_R
        self.pres_lst=pres_lst
        self.qres_lst=qres_lst
        self.rres_lst=rres_lst
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
