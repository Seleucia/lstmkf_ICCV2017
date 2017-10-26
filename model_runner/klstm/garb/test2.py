import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import helper.kalman_libs as kl
import model_runner.klstm.code2.book_plots as bp
import model_runner.klstm.garb.run_kalman as runner
from helper import config

params=config.get_params()
R_std = 2
Q_std = 0.04


def identity_matrix(n):
  """Returns nxn identity matrix."""
  # note, if n is a constant node, this assert node won't be executed,
  # this error will be caught during shape analysis
  assert_op = tf.Assert(tf.greater(n, 0), ["Matrix size must be positive"])
  with tf.control_dependencies([assert_op]):
    ones = tf.fill((n,1), 1.0)
    diag = tf.diag(ones)
  return diag



params['seq_length']=200
params["batch_size"]=2
params['n_hidden']= 4
params['nlayer']= 1 #LSTM
params['n_output']=4
seq_length=params['seq_length']
batch_size=params["batch_size"]


(measurements,groundtruth)=kl.get_data(params,R_std)
dt = 1.0   # time step
idx=1
measurements_idx=measurements[idx]
groundtruth_idx=groundtruth[idx]
measurements_kalman=np.expand_dims(measurements_idx, 2)

kalman_mu, kalman_cov=runner.test_filter(measurements_kalman,R_std)

kalman_mu=np.squeeze(kalman_mu)




#plot results
# measurements_idx *= R_std # convert to meters
# groundtruth_idx *= R_std # convert to meters
# kalman_mu *= R_std # convert to meters

plt.figure()
bp.plot_measurements(np.asarray(kalman_mu)[:, 0], np.asarray(kalman_mu)[:, 2],color='yellow', label='Kalman Filter')
bp.plot_measurements(np.asarray(groundtruth_idx)[:, 0], np.asarray(groundtruth_idx)[:, 1], color='red', label='Ground Truth')
bp.plot_measurements(np.asarray(measurements_idx)[:, 0], np.asarray(measurements_idx)[:, 1], color='blue', label='Measurements')
plt.legend(loc=2)
xmin=np.min(np.asarray(kalman_mu)[:, 0])
xmax=np.max(np.asarray(kalman_mu)[:, 0])
plt.xlim((xmin, xmax));
plt.show()
# ==> [[ 12.]]

