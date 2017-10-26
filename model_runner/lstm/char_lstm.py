import tensorflow as tf

from model_runner.common import rnn_cellv2 as rnn_cell


class Model():
  def __init__(self,params, infer=False):

    self.is_training = tf.placeholder(tf.bool)
    self.output_keep_prob = tf.placeholder(tf.float32)

    num_layers=params['nlayer']
    rnn_size=params['n_hidden']
    grad_clip=10

    cell = rnn_cell.LSTMCell(rnn_size, initializer=tf.contrib.layers.xavier_initializer(uniform=False)
                                   , forget_bias=1.0)
    cell_lst=[]
    for i in range(num_layers):
        if i==num_layers-1:
          cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)
        cell_lst.append(cell)
    cell = rnn_cell.MultiRNNCell(cell_lst)
    self.cell = cell

    NOUT = params['n_output'] # end_of_stroke + prob + 2*(mu + sig) + corr
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, params['n_input']])
    self.input_zero = tf.placeholder(dtype=tf.float32, shape=[None, None, params['n_input']])
    self.target_data =tf.placeholder(tf.float32, [None, None,params["n_output"]])
    self.initial_state = cell.zero_state(batch_size=params['batch_size'], dtype=tf.float32)

    with tf.variable_scope('rnnlm'):
      output_w1 = tf.get_variable(name="output_w1", shape=[rnn_size, NOUT],initializer=tf.contrib.layers.xavier_initializer() )
      output_b1 = tf.get_variable(name="output_b1", shape=[NOUT])

    outputs = []
    new_c_lst = []
    state = self.initial_state
    with tf.variable_scope("rnnlm"):
      for time_step in range(params['seq_length']):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state,new_c) = cell(self.input_data[:,time_step,:], state)
        cell_output=tf.add(tf.matmul(cell_output, output_w1),output_b1)
        y_hat_softmax=tf.nn.softmax(cell_output)
        if time_step==0:
          cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(dtype=tf.float32,x=self.target_data[:,time_step,:]) * tf.log(y_hat_softmax), [1]))
        else:
          cost += tf.reduce_mean(-tf.reduce_sum(tf.cast(dtype=tf.float32,x=self.target_data[:,time_step,:]) * tf.log(y_hat_softmax), [1]))
        outputs.append(cell_output)
        new_c_lst.append(new_c)

    self.pred=tf.argmax(y_hat_softmax, 1)
    self.y_hat=y_hat_softmax
    self.final_state = state
    self.new_c_lst=tf.pack(axis=0,values= new_c_lst)

    # for time_step in range(params['seq_length']):
    #     batch_seq=outputs[time_step,:,:]
        # y_tf=tf.argmax(self.target_data[self:,time_step,:], 1)
        # y_hat_softmax=tf.nn.softmax(batch_seq)
        # self.pred=tf.argmax(y_hat_softmax, 1)
        # self.y_hat=y_hat_softmax
        # if time_step==0:
        #   cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(dtype=tf.float32,x=self.target_data[:,time_step,:]) * tf.log(y_hat_softmax), [1]))
        # else:
        #   cost += tf.reduce_mean(-tf.reduce_sum(tf.cast(dtype=tf.float32,x=self.target_data[:,time_step,:]) * tf.log(y_hat_softmax), [1]))


    self.cost = tf.reduce_mean(cost)

    tf.scalar_summary('losses/total_loss', cost)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    total_parameters = 0
    for variable in tvars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    self.total_parameters=total_parameters
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
    # for grad in grads:
    #   grad_values = grad
    #   logging_ops.histogram_summary(grad.op.name + ':gradient', grad_values)
    #   logging_ops.histogram_summary(grad.op.name + ':gradient_norm', clip_ops.global_norm([grad_values]))
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
  def sample(self, sess, intial_input, char_dict, num=200, prime='The ', sampling_type=1):
    print intial_input



