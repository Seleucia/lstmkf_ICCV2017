import tensorflow as tf
from scipy.linalg import block_diag
import helper.kalman_libs as kl
import numpy as np

class Model(object):

    def __init__(self,params):
        def identity_matrix(bs,n):
          diag=tf.diag(tf.Variable(initial_value=[1]*n,dtype=tf.float32))
          lst=[]
          for i in range(bs):
              lst.append(diag)
          return tf.pack(lst)
        batch_size=params["batch_size"]
        num_layers=params['nlayer']
        rnn_size=params['n_hidden']
        grad_clip=10
        self.output_keep_prob=1.0
        NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr

        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        cell = cell_fn(rnn_size)#RNN size
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        self.cell = cell
        self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

        self.lr = tf.Variable(0.0, trainable=False)
        R_std = 0.5
        Q_std = 0.4
        self._z= tf.placeholder(dtype=tf.float32, shape=[batch_size,params['seq_length'],NOUT]) #batch size, seqlength, feature
        self.gt= tf.placeholder(dtype=tf.float32, shape=[batch_size,params['seq_length'],NOUT]) #batch size, seqlength, feature
        # self._x = tf.placeholder(dtype=tf.float32, shape=[batch_size,NOUT,1]) #batch size, seqlength, featuretf.zeros((batch_size,NOUT,1)) # state
        self._x = tf.zeros((batch_size,NOUT,1)) # state
        self._P = identity_matrix(batch_size,NOUT)* 100. # uncertainty covariance
        # G=np.identity(NOUT)
        # van_loan_discretization(tracker.F, G, dt)
        B = 0.0                # control transition matrix
        self._F = 0.0                # state transition matrix
        self._alpha_sq = 1.        # fading memory control
        self.M = 0.0                 # process-measurement cross correlation
        self._I = identity_matrix(batch_size,NOUT)


        dt = 1.0   # time step

        f_mat=np.identity(NOUT)
        F = tf.Variable(initial_value=f_mat,dtype=tf.float32)

        self.u = 0.0
        h_mat=np.identity(NOUT)
        H = tf.Variable(initial_value=h_mat,dtype=tf.float32)              # Measurement function


        # R = identity_matrix(batch_size,NOUT) * R_std**2
        R = (identity_matrix(batch_size,NOUT) * R_std**2)*2

        # Q = tf.Variable(initial_value=[[ 0.00025,  0.], [ 0.,  0.001]],dtype=tf.float32)
        G=np.identity(NOUT,dtype=np.float32)
        (sigma, Q) =kl.van_loan_discretization(f_mat, G, dt)
        Q=(Q/100.0)*18.0
        print np.max(Q)
        print R_std**2
        # Q=(identity_matrix(batch_size,NOUT) * R_std**2)
        # Q=Q*.001
        # q = kl.Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
        # vl = block_diag(q, q)
        # Q = tf.Variable(initial_value=q_mat,dtype=tf.float32)              # Measurement function
        # self.Q=tf.mul(q_mat,Q_std)
        # self.Q=q_mat
        #
        # self.Q=Q/10.
        # tmp=tf.Variable(initial_value=[[0, 0, 0, 0]],dtype=tf.float32)
        # x = tf.transpose(tmp)
        # P = identity_matrix(4) * 500.

        F=tf.pack([F]*batch_size)
        H=tf.pack([H]*batch_size)
        Q=tf.pack([Q]*batch_size)


        xres_lst=[]
        pres_lst=[]
        tres_lst=[]
        kres_lst=[]
        with tf.variable_scope('rnnlm'):
          output_w1 = tf.get_variable("output_w", [rnn_size, NOUT])
          output_b1 = tf.get_variable("output_b", [NOUT])

        state = self.initial_state
        with tf.variable_scope("rnnlm"):
            for time_step in range(params['seq_length']):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # else:
                #     self._x= tf.Variable(tf.expand_dims(self._z[:,time_step,:],2))

                if time_step == 0:
                    (self._x, state) = cell(tf.squeeze(self._z[:,time_step,:]), state)
                else:
                    (self._x, state) = cell(tf.squeeze(self._x), state)
                self._x  = tf.matmul(tf.squeeze(self._x),output_w1)+output_b1

                z=tf.expand_dims(self._z[:,time_step,:],2) #bs,features
                # z=self._z[:,i,:]
                # (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
                #predict
                P = self._P
                # self._x  = tf.expand_dims(tf.matmul(tf.squeeze(self._x),output_w1)+output_b1,2)
                # x = self._x
                # x = Fx + Bu
                # self._x = tf.matmul(F, x) + tf.mul(B, u)
                # P = FPF' + Q
                self._P = P + Q
                # self._P = self._alpha_sq * dot3(F, self._P, F.T) + Q


                #update
                P = self._P
                x = tf.expand_dims(self._x,2)

                # Hx = tf.matmul(H, x)
                self._y =z- x

                # S = HPH' + R
                # project system uncertainty into measurement space
                S = P + R

                # K = PH'inv(S)
                # map system uncertainty into kalman gain
                # K = tf.matmul(P, tf.matmul(tf.matrix_transpose(H), tf.matrix_inverse(S)))
                K = tf.matmul(P, tf.matrix_inverse(S))

                # x = x + Ky
                # predict new x with residual scaled by the kalman gain
                self._x = x + tf.matmul(K, self._y)
                xres_lst.append(self._x)
                tres_lst.append(x)
                kres_lst.append(K)


                # P = (I-KH)P(I-KH)' + KRK'
                # I_KH = self._I - tf.matmul(K, H)
                I_KH = self._I - K
                self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matmul(R, tf.matrix_transpose(K)))
                pres_lst.append(self._P)

                self._S = S
                self._K = K
        self.cost=tf.nn.l2_loss(self.gt-tf.transpose(tf.squeeze(xres_lst),(1,0,2)))
        self.xres_lst=xres_lst
        self.pres_lst=pres_lst
        self.tres_lst=tres_lst
        self.kres_lst=kres_lst
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def van_loan_discretization(F, G, dt):

        """ Discretizes a linear differential equation which includes white noise
        according to the method of C. F. van Loan [1]. Given the continuous
        model

            x' =  Fx + Gu

        where u is the unity white noise, we compute and return the sigma and Q_k
        that discretizes that equation.


        Examples
        --------

            Given y'' + y = 2u(t), we create the continuous state model of

            x' = [ 0 1] * x + [0]*u(t)
                 [-1 0]       [2]

            and a time step of 0.1:


            >>> F = np.array([[0,1],[-1,0]], dtype=float)
            >>> G = np.array([[0.],[2.]])
            >>> phi, Q = van_loan_discretization(F, G, 0.1)

            >>> phi
            array([[ 0.99500417,  0.09983342],
                   [-0.09983342,  0.99500417]])

            >>> Q
            array([[ 0.00133067,  0.01993342],
                   [ 0.01993342,  0.39866933]])

            (example taken from Brown[2])


        References
        ----------

        [1] C. F. van Loan. "Computing Integrals Involving the Matrix Exponential."
            IEEE Trans. Automomatic Control, AC-23 (3): 395-404 (June 1978)

        [2] Robert Grover Brown. "Introduction to Random Signals and Applied
            Kalman Filtering." Forth edition. John Wiley & Sons. p. 126-7. (2012)
        """


        n = F.shape[0]

        A = np.zeros((2*n, 2*n))

        # we assume u(t) is unity, and require that G incorporate the scaling term
        # for the noise. Hence W = 1, and GWG' reduces to GG"

        A[0:n,     0:n] = -F.dot(dt)
        A[0:n,   n:2*n] = G.dot(G.T).dot(dt)
        A[n:2*n, n:2*n] = F.T.dot(dt)

        B=np.expm(A)

        sigma = B[n:2*n, n:2*n].T

        Q = sigma.dot(B[0:n, n:2*n])

        return (sigma, Q)
