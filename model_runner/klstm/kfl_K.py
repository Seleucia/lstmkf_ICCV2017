import tensorflow as tf
from model_runner.common import rnn_cellv3 as rnncell

class Model(object):
    def __init__(self,params):
        def identity_matrix(bs,n):
          diag=tf.diag(tf.Variable(initial_value=[1]*n,dtype=tf.float32))
          lst=[]
          for i in range(bs):
              lst.append(diag)
          return tf.pack(lst)

        batch_size = params["batch_size"]
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

        # LSTM for Kalman gain
        cell_K =rnncell.ModifiedLSTMCell(params['Kn_hidden'], forget_bias=1,initializer= tf.contrib.layers.xavier_initializer(),num_proj=None)
        cell_K = rnncell.MultiRNNCell([cell_K] * params['nlayer'])
        cell_K = rnncell.DropoutWrapper(cell_K, output_keep_prob=self.output_keep_prob)
        self.cell_K = cell_K


        self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.initial_state_K = cell_K.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"], params['seq_length']])

        self._z = tf.placeholder(dtype=tf.float32,
                                 shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature
        self.target_data = tf.placeholder(dtype=tf.float32,
                                          shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature

        xres_lst=[]
        pres_lst=[]
        tres_lst=[]
        kres_lst=[]
        with tf.variable_scope('rnnlm'):
          output_w1 = tf.get_variable("output_w", [rnn_size, NOUT],initializer= tf.contrib.layers.xavier_initializer())
          output_b1 = tf.get_variable("output_b", [NOUT])
          output_w1_K = tf.get_variable("output_w_K", [params['Kn_hidden'], NOUT],initializer= tf.contrib.layers.xavier_initializer())
          output_b1_K = tf.get_variable("output_b_K", [NOUT])

          output_w1_K_inp = tf.get_variable("output_w_K_inp", [NOUT*2, params['K_inp']],initializer= tf.contrib.layers.xavier_initializer())
          output_b1_K_inp = tf.get_variable("output_b_K_inp", [params['K_inp']])


        state_F = self.initial_state
        state_K = self.initial_state_K
        with tf.variable_scope("rnnlm"):
            for time_step in range(params['seq_length']):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                z=self._z[:,time_step,:]
                if time_step == 0:
                    self._x= z

                with tf.variable_scope("transitionF"):
                    (pred, state_F,ls_internals) = cell(self._x, state_F)
                    self._x  = tf.matmul(pred,output_w1)+output_b1

                with tf.variable_scope("gainK"):
                    inp=tf.concat(1,[self._x,z])
                    emb  = tf.nn.relu(tf.matmul(inp,output_w1_K_inp)+output_b1_K_inp)
                    (pred_val, state_K,ls_internals) = cell_K(emb, state_K)
                    K  = tf.nn.tanh(tf.matmul(pred_val,output_w1_K)+output_b1_K)

                self._y =z- self._x

                # predict new x with residual scaled by the kalman gain
                self._x = self._x + tf.mul(K, self._y)
                xres_lst.append(self._x)

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

        self.states = {}
        self.states["F_t"] = state_F
        self.states["K_t"] = state_K
        self.xres_lst = xres_lst
        self.pres_lst = pres_lst
        self.tres_lst = tres_lst
        self.kres_lst = kres_lst
