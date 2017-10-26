import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import helper.kalman_libs as kl
import model_runner.klstm.code2.book_plots as bp
import model_runner.klstm.garb.run_kalman as runner
from helper import config
from model_runner.klstm.kfl_f import Model as Kmodel
from model_runner.klstm.kfl_QRf import Model as Kmodel_mdn

params=config.get_params()
R_std = 0.5
Q_std = 0.04
params['lr']=0.001
number_of_batches=200
params['seq_length']=100
params["batch_size"]=50
params['n_hidden']= 16
params['nlayer']= 1 #LSTM

seq_length=params['seq_length']
batch_size=params["batch_size"]
params["theta"]=0.3

idx=-1
max_epoch=15

consider=5
test_mode='step' #step, rotation

if test_mode=='step':
    R_std = 1
    data_stream=kl.get_data
    params['n_output']=4
    filter=runner.test_filter
else:
    data_stream=kl.get_dataV3
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


decay_rate=0.8
LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden state
LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden sta
state_reset_counter_lst=[0 for i in range(batch_size)]
total_train_loss=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2



with tf.Graph().as_default():
    tracker_klstm_mdn = Kmodel_mdn(params=params)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'training Noise KLSTM'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_klstm_mdn.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_klstm_mdn._z: mes, tracker_klstm_mdn.target_data: gt, tracker_klstm_mdn.initial_state: LStateList_pre,}
                mu_klstm_mdn, cov_klstm_mdn, t, k, cost, _ = sess.run([tracker_klstm_mdn.xres_lst, tracker_klstm_mdn.pres_lst,
                                                               tracker_klstm_mdn.tres_lst, tracker_klstm_mdn.kres_lst, tracker_klstm_mdn.cost,
                                                               tracker_klstm_mdn.train_op], feed)
                # print cost

            print 'Testing Noise KLSTM model...'
            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_klstm_mdn._z: mes, tracker_klstm_mdn.target_data: gt, tracker_klstm_mdn.initial_state: LStateList_pre,}
                mu_klstm_mdn, cov_klstm_mdn, t, k = sess.run([tracker_klstm_mdn.xres_lst, tracker_klstm_mdn.pres_lst,
                                                      tracker_klstm_mdn.tres_lst, tracker_klstm_mdn.kres_lst], feed)
                mu_klstm_mdn=np.transpose(np.squeeze(np.asarray(mu_klstm_mdn)),axes=(1,0,2))
                loss=kl.compute_loss(gt=gt[:,consider:,:],est=mu_klstm_mdn[:,consider:,:])
                batch_loss.append(loss)
            print "Epoch Noise KLSTM loss %i/%f"%(e,np.mean(batch_loss))

with tf.Graph().as_default():
    tracker_klstm = Kmodel(params=params)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'training  KLSTM'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_klstm.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_klstm._z: mes, tracker_klstm.gt: gt, tracker_klstm.initial_state: LStateList_pre,}
                mu_klstm, cov_klstm, t, k, cost, _ = sess.run([tracker_klstm.xres_lst, tracker_klstm.pres_lst, tracker_klstm.tres_lst, tracker_klstm.kres_lst, tracker_klstm.cost, tracker_klstm.train_op], feed)
                # print cost

            print 'Testing KLSTM model...'
            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_klstm._z: mes, tracker_klstm.gt: gt, tracker_klstm.initial_state: LStateList_pre,}
                mu_klstm, cov_klstm, t, k = sess.run([tracker_klstm.xres_lst, tracker_klstm.pres_lst, tracker_klstm.tres_lst, tracker_klstm.kres_lst], feed)
                mu_klstm=np.transpose(np.squeeze(np.asarray(mu_klstm)),axes=(1,0,2))
                loss=kl.compute_loss(gt=gt[:,consider:,:],est=mu_klstm[:,consider:,:])
                batch_loss.append(loss)
            print "Epoch KLSTM loss %i/%f"%(e,np.mean(batch_loss))

# np.save('mu_klstm_mdn',mu_klstm_mdn)
# np.save('mu_klstm',mu_klstm)
# np.save('mu_lstm',mu_lstm)
# np.save('kalman_mu',kalman_mu)
muu_klstm=mu_klstm[idx]
muu_klstm_mdn=mu_klstm_mdn[idx]

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
bp.plot_measurements(np.asarray(muu_klstm_mdn)[:, 0], np.asarray(muu_klstm_mdn)[:, 1], color='cyan', label='Noise Kalman LSTM',lw=1)
bp.plot_measurements(np.asarray(muu_klstm)[:, 0], np.asarray(muu_klstm)[:, 1], color='green', label='Kalman LSTM',lw=1)
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
