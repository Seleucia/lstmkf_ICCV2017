import tensorflow as tf
from model_runner.common import rnn_cellv3 as rnncell

class Model(object):
    def __init__(self,params, is_training=True):
        batch_size=params["batch_size"]
        num_layers=params['nlayer']
        rnn_size=params['n_hidden']
        grad_clip=10
        self.output_keep_prob=1.0
        NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr

        # Transition LSTM
        cell_lst = []
        for i in range(num_layers):
            cell = rnncell.ModifiedLSTMCell(rnn_size, forget_bias=1, initializer=tf.contrib.layers.xavier_initializer(),
                                            num_proj=None, is_training=self.is_training)
            if i > 0 and is_training == True:
                cell_drop = rnncell.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
                cell = cell_drop
            cell_lst.append(cell)
        self.cell = rnncell.MultiRNNCell(cell_lst)
        #
        # cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        # cell = cell_fn(rnn_size)#RNN size
        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        # self.cell = cell
        self.initial_state = self.cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)
        self.repeat_data = tf.placeholder(dtype=tf.int32, shape=[params["batch_size"], params['seq_length']])

        self.lr = tf.Variable(0.0, trainable=False)
        self._z = tf.placeholder(dtype=tf.float32,
                                 shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature
        self.target_data = tf.placeholder(dtype=tf.float32,
                                          shape=[None, params['seq_length'], NOUT])  # batch size, seqlength, feature
        self._x = tf.zeros((batch_size,NOUT,1)) # state

        xres_lst=[]
        with tf.variable_scope('rnnlm'):
            output_w1 = tf.get_variable("output_w1", [rnn_size, rnn_size],initializer=tf.contrib.layers.xavier_initializer())
            output_b1 = tf.get_variable("output_b1", [rnn_size])

            output_w2 = tf.get_variable("output_w2", [rnn_size, rnn_size], initializer=tf.contrib.layers.xavier_initializer())
            output_b2 = tf.get_variable("output_b2", [rnn_size])

            output_w3 = tf.get_variable("output_w3", [rnn_size, NOUT], initializer=tf.contrib.layers.xavier_initializer())
            output_b3 = tf.get_variable("output_b3", [NOUT])

        state = self.initial_state
        with tf.variable_scope("rnnlm"):
            for time_step in range(params['seq_length']):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                r = self.repeat_data[:, time_step]
                where_flt = tf.not_equal(r, 0)
                (self._x, state) = cell(self._z[:,time_step,:], state)
                self._x  = tf.matmul(self._x,output_w1)+output_b1
                xres_lst.append(self._x)
                new_state = []
                for i in range(params['nlayer']):
                    s = []
                    s.append(tf.select(where_flt, state[i][0], pre_state[i][0]))
                    s.append(tf.select(where_flt, state[i][1], pre_state[i][1]))
                    new_state.append(tuple(s))
                state = new_state
                pre_state = state

        rnn_output=tf.reshape(tf.transpose(tf.pack(xres_lst), [1, 0, 2]), [-1, params['n_output']])

        hidden_1 = tf.nn.relu(tf.add(tf.matmul(rnn_output, output_w1), output_b1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, output_w2), output_b2))
        self.final_output = tf.add(tf.matmul(hidden_2, output_w3), output_b3)

        # final_output = tf.reshape(tf.transpose(tf.pack(xres_lst), [1, 0, 2]), [-1, params['n_output']])
        flt = tf.squeeze(tf.reshape(self.repeat_data, [-1, 1]), [1])
        where_flt = tf.not_equal(flt, 0)
        indices = tf.where(where_flt)

        y = tf.reshape(self.target_data, [-1, params["n_output"]])
        self.final_output = tf.gather(self.final_output, tf.squeeze(indices, [1]))
        self.y = tf.gather(y, tf.squeeze(indices, [1]))

        tmp = self.final_output - self.y
        loss = tf.nn.l2_loss(tmp)

        self.tvars = tf.trainable_variables()
        l2_reg=tf.reduce_sum([tf.nn.l2_loss(var) for var in self.tvars])
        l2_reg=tf.mul(l2_reg,1e-4)
        self.cost = tf.reduce_mean(loss)+l2_reg
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
        self.states["lstm_t"] = state
