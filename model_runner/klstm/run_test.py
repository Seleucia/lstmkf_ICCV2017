import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import code2.book_plots as bp
import helper.kalman_libs as kl
from helper import gpu_config
from kfl_f import Model as kfl_f
from kfl_QRFf import Model as kfl_QRFf
from kfl_QRf import Model as kfl_QRf
from kf_QR import Model as kf_QR
from lstm import Model as Lmodel
from model_runner.klstm.garb import run_kalman as runner

params=gpu_config.get_params()
params['R_std'] = 0.5
Q_std = 0.04
params['lr']=0.001
number_of_batches=20
params['seq_length']=100
params["batch_size"]=50
params['n_hidden']= 16
params['nlayer']= 1 #LSTM

seq_length=params['seq_length']
batch_size=params["batch_size"]
params["theta"]=0.3

idx=-1
max_epoch=20

consider=5
test_mode='combine' #step, rotation,combine
test_kalman=1
params['test_mode']=test_mode

if test_mode=='step2d':
    params['R_std'] = 0.2
    data_stream=kl.get_data
    params['n_output']=4
    filter=runner.test_filter
if test_mode=='step2d':
    params['R_std'] = 0.2
    data_stream=kl.get_data2d
    params['n_output']=2
    filter=runner.test_filter2d
    test_kalman=0
elif test_mode=='rotation':
    data_stream=kl.get_dataV3
    params['n_output']=2
    filter=runner.test_filter3
    params['R_std'] = 0.2
    test_kalman=1
    dt=1
else:
    data_stream=kl.get_data_combine
    params['n_output']=2
    filter=runner.test_filter3
    params['R_std'] = 0.2
    params['seq_length']=125
    test_kalman=0

F,H=kl.build_matrices(params)
params['F_shape']=F.shape
params['F']=F[0]
params['F_mat']=F
params['H_mat']=H
params['H_shape']=H.shape
params['H']=H[0]
(measurements,groundtruth)=data_stream(params,batch_size*number_of_batches)
(measurements_test,groundtruth_test)=data_stream(params,batch_size*(number_of_batches/4))
n_train_batches = groundtruth.shape[0]
n_train_batches /= batch_size
n_test_batches = groundtruth_test.shape[0]
n_test_batches /= batch_size


if test_kalman==1:
    print 'Kalman Testing:'
    lst_kalman_err=[]
    for i in range(groundtruth_test.shape[0]):
        measurements_idx=measurements_test[i]
        groundtruth_idx=groundtruth_test[i]
        measurements_kalman=np.expand_dims(measurements_idx, 2)
        kalman_mu, kalman_cov=filter(params,measurements_kalman)
        kalman_mu=np.squeeze(kalman_mu)
        loss=kl.compute_loss(gt=groundtruth_idx[consider:], est=kalman_mu[consider:])
        lst_kalman_err.append(loss)
    print np.mean(lst_kalman_err)
else:
    print 'Kalman NOT tested'

measurements_idx=measurements_test[idx]
groundtruth_idx=groundtruth_test[idx]
measurements_kalman=np.expand_dims(measurements_idx, 2)
kalman_mu, kalman_cov=filter(params,measurements_kalman)

NoiseKalmanlstm_err=[]
Kalmanlstm_err=[]
lstm_err=[]
decay_rate=0.8
LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden state
LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden sta
P= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)

state_reset_counter_lst=[0 for i in range(batch_size)]
total_train_loss=0
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.2


with tf.Graph().as_default():
    tracker_kf_QR = kf_QR(params=params)
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'Training: (kf_QR)'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_kf_QR.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024


                feed = {tracker_kf_QR._z: mes, tracker_kf_QR.target_data: gt, tracker_kf_QR.initial_state_Q_noise: LStateList_pre,
                        tracker_kf_QR.initial_state_R_noise: LStateList_pre, tracker_kf_QR.H: H, tracker_kf_QR.F: F}

                cost,_ = sess.run([tracker_kf_QR.cost, tracker_kf_QR.train_op], feed)

            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
                feed = {tracker_kf_QR._z: mes, tracker_kf_QR.target_data: gt, tracker_kf_QR.initial_state_Q_noise: LStateList_pre,
                        tracker_kf_QR.initial_state_R_noise: LStateList_pre, tracker_kf_QR.H: H, tracker_kf_QR.F: F}
                mu_kf_QR= sess.run([tracker_kf_QR.xres_lst], feed)
                if test_mode=='rotation':
                    mu_kf_QR=np.squeeze(np.asarray(mu_kf_QR[0]))
                else:
                    mu_kf_QR=np.squeeze(np.asarray(mu_kf_QR[0]))
                loss=kl.compute_loss(gt=gt[:, consider:, :], est=mu_kf_QR[:, consider:, :])
                batch_loss.append(loss)
            print "Epoch: (kf_QR) loss %i/%f"%(e,np.mean(batch_loss))
            NoiseKalmanlstm_err.append(np.mean(batch_loss))

with tf.Graph().as_default():
    tracker_kfl_QRFf = kfl_QRFf(params=params)
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'Training: (kfl_QRFf)'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_kfl_QRFf.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024

                feed = {tracker_kfl_QRFf._z: mes, tracker_kfl_QRFf.target_data: gt, tracker_kfl_QRFf.initial_state: LStateList_pre
                    , tracker_kfl_QRFf.initial_state_Q_noise: LStateList_pre, tracker_kfl_QRFf.initial_state_R_noise:
                            LStateList_pre, tracker_kfl_QRFf._P_inp: P, tracker_kfl_QRFf._I: I}

                cost, _ = sess.run([tracker_kfl_QRFf.cost, tracker_kfl_QRFf.train_op], feed)
                # print cost

            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
                feed = {tracker_kfl_QRFf._z: mes, tracker_kfl_QRFf.target_data: gt, tracker_kfl_QRFf.initial_state: LStateList_pre
                    , tracker_kfl_QRFf.initial_state_Q_noise: LStateList_pre, tracker_kfl_QRFf.initial_state_R_noise:
                            LStateList_pre, tracker_kfl_QRFf._P_inp: P, tracker_kfl_QRFf._I: I}
                mu_kfl_QRFf, cov_klstm_mdn, t, k = sess.run([tracker_kfl_QRFf.xres_lst, tracker_kfl_QRFf.pres_lst,
                                                             tracker_kfl_QRFf.tres_lst, tracker_kfl_QRFf.kres_lst], feed)
                mu_kfl_QRFf=np.transpose(np.squeeze(np.asarray(mu_kfl_QRFf)), axes=(1, 0, 2))
                loss=kl.compute_loss(gt=gt[:, consider:, :], est=mu_kfl_QRFf[:, consider:, :])
                batch_loss.append(loss)
            print "Epoch: (kfl_QRFf) loss %i/%f"%(e,np.mean(batch_loss))
            NoiseKalmanlstm_err.append(np.mean(batch_loss))

with tf.Graph().as_default():
    tracker_kfl_QRf = kfl_QRf(params=params)
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'Training: (kfl_QRf)'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_kfl_QRf.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])])
                feed = {tracker_kfl_QRf._z: mes, tracker_kfl_QRf.target_data: gt, tracker_kfl_QRf.initial_state: LStateList_pre
                       , tracker_kfl_QRf._P_inp: P, tracker_kfl_QRf._I: I}
                cost, _ = sess.run([tracker_kfl_QRf.cost, tracker_kfl_QRf.train_op], feed)
                # print cost

            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_kfl_QRf._z: mes, tracker_kfl_QRf.target_data: gt, tracker_kfl_QRf.initial_state: LStateList_pre
                       , tracker_kfl_QRf._P_inp: P, tracker_kfl_QRf._I: I}
                mu_kfl_QRf = sess.run([tracker_kfl_QRf.xres_lst], feed)
                mu_kfl_QRf=np.transpose(np.squeeze(np.asarray(mu_kfl_QRf)), axes=(1, 0, 2))
                loss=kl.compute_loss(gt=gt[:, consider:, :], est=mu_kfl_QRf[:, consider:, :])
                batch_loss.append(loss)
            print "Epoch: (kfl_QRf) loss %i/%f"%(e,np.mean(batch_loss))
            NoiseKalmanlstm_err.append(np.mean(batch_loss))



with tf.Graph().as_default():
    tracker_lstm = Lmodel(params=params)
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        print 'Training: (lstm)'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_lstm.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_lstm._z: mes, tracker_lstm.gt: gt, tracker_lstm.initial_state: LStateList_pre,}
                mu_lstm, cost, _ = sess.run([tracker_lstm.xres_lst, tracker_lstm.cost, tracker_lstm.train_op], feed)
                # print cost

            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_lstm._z: mes, tracker_lstm.gt: gt, tracker_lstm.initial_state: LStateList_pre,}
                mu_lstm = sess.run([tracker_lstm.xres_lst], feed)
                mu_lstm=np.transpose(np.squeeze(np.asarray(mu_lstm)),axes=(1,0,2))
                loss=kl.compute_loss(gt=gt[:, consider:, :], est=mu_lstm[:, consider:, :])
                batch_loss.append(loss)
            print "Epoch: (lstm) loss %i/%f"%(e,np.mean(batch_loss))
            lstm_err.append(np.mean(batch_loss))

with tf.Graph().as_default():
    tracker_kfl_f = kfl_f(params=params)
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'Training: (kfl_f)'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_kfl_f.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_kfl_f._z: mes, tracker_kfl_f.gt: gt, tracker_kfl_f.initial_state: LStateList_pre,}
                mu_kfl_f, cov_klstm, t, k, cost, _ = sess.run([tracker_kfl_f.xres_lst, tracker_kfl_f.pres_lst, tracker_kfl_f.tres_lst, tracker_kfl_f.kres_lst, tracker_kfl_f.cost, tracker_kfl_f.train_op], feed)
                # print cost

            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_kfl_f._z: mes, tracker_kfl_f.gt: gt, tracker_kfl_f.initial_state: LStateList_pre,}
                mu_kfl_f, cov_klstm, t, k = sess.run([tracker_kfl_f.xres_lst, tracker_kfl_f.pres_lst, tracker_kfl_f.tres_lst, tracker_kfl_f.kres_lst], feed)
                mu_kfl_f=np.transpose(np.squeeze(np.asarray(mu_kfl_f)), axes=(1, 0, 2))
                loss=kl.compute_loss(gt=gt[:, consider:, :], est=mu_kfl_f[:, consider:, :])
                batch_loss.append(loss)
            print "Epoch: (kfl_f) loss %i/%f"%(e,np.mean(batch_loss))
            Kalmanlstm_err.append(np.mean(batch_loss))

# np.save('mu_klstm_mdn',mu_klstm_mdn)
# np.save('mu_klstm',mu_klstm)
# np.save('mu_lstm',mu_lstm)
# np.save('kalman_mu',kalman_mu)
muu_kfl_QRFf=mu_kfl_QRFf[idx]
muu_kf_QR=mu_kf_QR[idx]
muu_kfl_f=mu_kfl_f[idx]
muu_kfl_QRf=mu_kfl_QRf[idx]
muu_lstm=mu_lstm[idx]

print("number_of_batches: %f, theta: %f,R_std: %f, n_hidden: %f, lr: %f, test_mode: %s"%(number_of_batches,params["theta"],params["R_std"],params['n_hidden'],params["lr"],test_mode))
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
bp.plot_measurements(np.asarray(muu_kfl_QRFf)[:, 0], np.asarray(muu_kfl_QRFf)[:, 1], color='#008B8B', label='kfl_QRFf', lw=1)
bp.plot_measurements(np.asarray(muu_kf_QR)[:, 0], np.asarray(muu_kf_QR)[:, 1], color='#9400D3', label='kf_QR', lw=1)
bp.plot_measurements(np.asarray(muu_kfl_QRf)[:, 0], np.asarray(muu_kfl_QRf)[:, 1], color='#654321', label='kfl_QRf', lw=1)
bp.plot_measurements(np.asarray(muu_kfl_f)[:, 0], np.asarray(muu_kfl_f)[:, 1], color='#056608', label='kfl_f', lw=1)
bp.plot_measurements(np.asarray(muu_lstm)[:, 0], np.asarray(muu_lstm)[:, 1], color='#000000', label='lstm',lw=1)
if test_kalman==0:
    bp.plot_measurements(np.asarray(kalman_mu)[:, 0], np.asarray(kalman_mu)[:, 1],color='#FFF600', label='Kalman Filter',lw=1)

bp.plot_measurements(np.asarray(groundtruth_idx)[:, 0], np.asarray(groundtruth_idx)[:, 1],color='#CC0000', label='Ground Truth',lw=1)
bp.plot_measurements(np.asarray(measurements_idx)[:, 0], np.asarray(measurements_idx)[:, 1],color='#0247FE', label='Measurements',lw=1)
plt.legend(loc=2)
xmin=np.min(np.asarray(groundtruth_idx)[:, 0])
xmax=np.max(np.asarray(groundtruth_idx)[:, 0])
plt.xlim((xmin, xmax));
plt.show()
# ==> [[ 12.]]

# Close the Session when we're done.
