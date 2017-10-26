import matplotlib.pyplot as plt
import numpy as np

import helper.kalman_libs as kl
import model_runner.klstm.code2.book_plots as bp
import model_runner.klstm.garb.run_kalman as runner
from helper import config

params=config.get_params()
R_std = 0.5
Q_std = 0.04
params['lr']=0.001
number_of_batches=8
params['seq_length']=100
params["batch_size"]=50
params['n_hidden']= 16
params['nlayer']= 1 #LSTM
params['n_output']=2
seq_length=params['seq_length']
batch_size=params["batch_size"]
params["theta"]=0.3

idx=-1
max_epoch=10

consider=20
(measurements,groundtruth)=kl.get_dataV4(params,batch_size*number_of_batches,R_std)
(measurements_test,groundtruth_test)=kl.get_dataV4(params,batch_size*(number_of_batches/4),R_std)
n_train_batches = groundtruth.shape[0]
n_train_batches /= batch_size
n_test_batches = groundtruth_test.shape[0]
n_test_batches /= batch_size
measurements_idx=measurements_test[idx]
groundtruth_idx=groundtruth_test[idx]
measurements_kalman=np.expand_dims(measurements_idx, 2)
kalman_mu, kalman_cov=runner.test_filter3(measurements_kalman,R_std)

#plot results
# zs *= .3048 # convert to meters
# ys *= .3048 # convert to meters
# muu /= .3048 # convert to meters
# kalman_mu *= .3048 # convert to meters
# bp.plot_filter(np.asarray(muu)[:, 0], np.asarray(muu)[:, 2])
# bp.plot_measurements(np.asarray(zs)[:, 0], np.asarray(zs)[:, 1])
bp.plot_measurements(np.asarray(kalman_mu)[:, 0], np.asarray(kalman_mu)[:, 1],color='yellow', label='Kalman Filter',lw=1)
bp.plot_measurements(np.asarray(groundtruth_idx)[:, 0], np.asarray(groundtruth_idx)[:, 1],color='red', label='Ground Truth',lw=1)
bp.plot_measurements(np.asarray(measurements_idx)[:, 0], np.asarray(measurements_idx)[:, 1],color='blue', label='Measurements',lw=1)
plt.legend(loc=2)
xmin=np.min(np.asarray(groundtruth_idx)[:, 0])
xmax=np.max(np.asarray(groundtruth_idx)[:, 0])
plt.xlim((xmin, xmax));
plt.show()