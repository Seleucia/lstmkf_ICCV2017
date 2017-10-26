import tensorflow as tf
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

matrix1 = tf.constant([[3., 3.]])


class kftf(object):

    def __init__(self, dim_x, dim_z,bs,N, R=None, H=None, u=0.0):
        def identity_matrix(bs,n):
          diag=tf.diag(tf.Variable(initial_value=[1]*n,dtype=tf.float32))
          lst=[]
          for i in range(bs):
              lst.append(diag)
          return tf.pack(lst)



        R_std = 0.35
        Q_std = 0.04
        self._z= tf.placeholder(dtype=tf.float32, shape=[bs,N,2]) #batch size, seqlength, feature
        self._x = tf.zeros((bs,dim_x,1)) # state
        self._P = identity_matrix(bs,dim_x)* 500. # uncertainty covariance
        self._Q = identity_matrix(bs,dim_x) # process uncertainty
        B = 0.0                # control transition matrix
        self._F = 0.0                # state transition matrix
        self._alpha_sq = 1.        # fading memory control
        self.M = 0.0                 # process-measurement cross correlation
        self._I = identity_matrix(bs,dim_x)


        dt = 1.0   # time step

        F = tf.Variable(initial_value=[[1, dt, 0,  0],
                                       [0,  1, 0,  0],
                                       [0,  0, 1, dt],
                                       [0,  0, 0,  1]],dtype=tf.float32)

        self.u = 0.0
        H = tf.Variable(initial_value=[[1/0.3048, 0, 0, 0],[0, 0, 1/0.3048, 0]],dtype=tf.float32)              # Measurement function


        R = identity_matrix(bs,2) * R_std**2

        # Q = tf.Variable(initial_value=[[ 0.00025,  0.0005,  0.0,  0.0],
        #                                [ 0.0005,  0.001,  0.0,  0.0],
        #                                [ 0.0,  0.0,  0.00025,  0.0005],
        #                                [ 0.0,  0.0,  0.0005,  0.001]],dtype=tf.float32)
        q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
        vl = block_diag(q, q)
        Q = tf.Variable(initial_value=vl,dtype=tf.float32)              # Measurement function

        # Q=tf.mul(Q,Q_std)

        self.Q=Q
        tmp=tf.Variable(initial_value=[[0, 0, 0, 0]],dtype=tf.float32)
        x = tf.transpose(tmp)
        # P = identity_matrix(4) * 500.

        F=tf.pack([F]*bs)
        H=tf.pack([H]*bs)
        Q=tf.pack([Q]*bs)

        xres_lst=[]
        pres_lst=[]
        tres_lst=[]

        for i in range(N):
            z=tf.expand_dims(self._z[:,i,:],2) #bs,features
            # (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
            #predict
            P = self._P
            x = self._x
            # x = Fx + Bu
            self._x = tf.matmul(F, x) + tf.mul(B, u)
            # P = FPF' + Q
            self._P = self._alpha_sq * tf.matmul(F,tf.matmul(P, tf.matrix_transpose(F))) + Q
            # self._P = self._alpha_sq * dot3(F, self._P, F.T) + Q


            #update
            P = self._P
            x = self._x

            Hx = tf.matmul(H, x)
            self._y =z- Hx

            # S = HPH' + R
            # project system uncertainty into measurement space
            S = tf.matmul(H, tf.matmul(P, tf.matrix_transpose(H))) + R

            # K = PH'inv(S)
            # map system uncertainty into kalman gain
            K = tf.matmul(P, tf.matmul(tf.matrix_transpose(H), tf.matrix_inverse(S)))

            # x = x + Ky
            # predict new x with residual scaled by the kalman gain
            self._x = x + tf.matmul(K, self._y)
            xres_lst.append(self._x)
            tres_lst.append(x)


            # P = (I-KH)P(I-KH)' + KRK'
            I_KH = self._I - tf.matmul(K, H)
            self._P = tf.matmul(I_KH, tf.matmul(P, tf.matrix_transpose(I_KH))) + tf.matmul(K, tf.matmul(R, tf.matrix_transpose(K)))
            pres_lst.append(self._P)

            self._S = S
            self._K = K
        self.xres_lst=xres_lst
        self.pres_lst=pres_lst
        self.tres_lst=tres_lst
