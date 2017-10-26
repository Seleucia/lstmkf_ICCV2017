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
params['seq_length']=125
params["batch_size"]=50
params['n_hidden']= 16
params['nlayer']= 1 #LSTM

seq_length=params['seq_length']
batch_size=params["batch_size"]
params["theta"]=0.3

idx=-1
max_epoch=15

consider=5
test_mode='combine' #step, rotation,combine

if test_mode=='step':
    R_std = 1
    data_stream=kl.get_data
    params['n_output']=4
    filter=runner.test_filter
elif test_mode=='rotation':
    data_stream=kl.get_dataV3
    params['n_output']=2
    filter=runner.test_filter3
    R_std = 0.2
else:
    data_stream=kl.get_data_combine
    params['n_output']=2
    filter=runner.test_filter3
    R_std = 0.2


(measurements,groundtruth)=data_stream(params,batch_size*number_of_batches,R_std)
(measurements_test,groundtruth_test)=data_stream(params,batch_size*(number_of_batches/4),R_std)
n_train_batches = groundtruth.shape[0]
n_train_batches /= batch_size
n_test_batches = groundtruth_test.shape[0]
n_test_batches /= batch_size
measurements_idx=measurements_test[idx]
groundtruth_idx=groundtruth_test[idx]
measurements_kalman=np.expand_dims(measurements_idx, 2)
kalman_mu, kalman_cov=filter(measurements_kalman,R_std)


print("10number_of_batches: %f, theta: %f,R_std: %f, n_hidden: %f, lr: %f, test_mode: %s"%(number_of_batches,params["theta"],R_std,params['n_hidden'],params["lr"],test_mode))
# zs=zs[idx]
# ys=ys[idx]
plt.figure()

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
# ==> [[ 12.]]

# Close the Session when we're done.
