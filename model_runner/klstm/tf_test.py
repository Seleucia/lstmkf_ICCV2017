import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import code2.book_plots as bp
import helper.kalman_libs as kl
from helper import config
from kfl_f import Model as Kmodel
from lstm import Model as Lmodel
from model_runner.klstm.garb import run_kalman as runner

params=config.get_params()
R_std = 0.5
Q_std = 0.04
params['lr']=0.001
params['seq_length']=100
params["batch_size"]=50
params['n_hidden']= 16
params['nlayer']= 1 #LSTM
params['n_output']=2
seq_length=params['seq_length']
batch_size=params["batch_size"]
params["theta"]=0.3

idx=-1
max_epoch=20

consider=20
(measurements,groundtruth)=kl.get_dataV3(params,batch_size*1000,R_std)
(measurements_test,groundtruth_test)=kl.get_dataV3(params,batch_size*200,R_std)
n_train_batches = groundtruth.shape[0]
n_train_batches /= batch_size
n_test_batches = groundtruth_test.shape[0]
n_test_batches /= batch_size


print 'Kalman Testing:'
lst_kalman_err=[]
for i in range(groundtruth_test.shape[0]):
    measurements_idx=measurements_test[i]
    groundtruth_idx=groundtruth_test[i]
    measurements_kalman=np.expand_dims(measurements_idx, 2)

    kalman_mu, kalman_cov=runner.test_filter3(measurements_kalman,R_std)
    kalman_mu=np.squeeze(kalman_mu)
    loss=kl.compute_loss(gt=groundtruth_idx[consider:],est=kalman_mu[consider:])
    lst_kalman_err.append(loss)
print np.mean(lst_kalman_err)

measurements_idx=measurements_test[idx]
groundtruth_idx=groundtruth_test[idx]
measurements_kalman=np.expand_dims(measurements_idx, 2)
kalman_mu, kalman_cov=runner.test_filter3(measurements_kalman,R_std)


decay_rate=0.7
LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden state
LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden sta
state_reset_counter_lst=[0 for i in range(batch_size)]
total_train_loss=0
with tf.Graph().as_default():
    tracker_klstm = Kmodel(params=params)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'training KLSTM'
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

with tf.Graph().as_default():
    tracker_lstm = Lmodel(params=params)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print 'Training LSTM'
        for e in range(max_epoch):
            sess.run(tf.assign(tracker_lstm.lr, params['lr'] * (decay_rate ** e)))
            for minibatch_index in xrange(n_train_batches):
                gt=groundtruth[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_lstm._z: mes, tracker_lstm.gt: gt, tracker_lstm.initial_state: LStateList_pre,}
                mu_lstm, cost, _ = sess.run([tracker_lstm.xres_lst, tracker_lstm.cost, tracker_lstm.train_op], feed)
                # print cost

            print 'Testing LSTM model...'
            batch_loss=[]
            for minibatch_index in xrange(n_test_batches):
                gt=groundtruth_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                mes=measurements_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {tracker_lstm._z: mes, tracker_lstm.gt: gt, tracker_lstm.initial_state: LStateList_pre,}
                mu_lstm = sess.run([tracker_lstm.xres_lst], feed)
                mu_lstm=np.transpose(np.squeeze(np.asarray(mu_lstm)),axes=(1,0,2))
                loss=kl.compute_loss(gt=gt[:,consider:,:],est=mu_lstm[:,consider:,:])
                batch_loss.append(loss)
            print "Epoch LSTM loss %i/%f"%(e,np.mean(batch_loss))



np.save('mu_klstm',mu_klstm)
np.save('mu_lstm',mu_lstm)
np.save('kalman_mu',kalman_mu)
muu_klstm=mu_klstm[idx]
muu_lstm=mu_lstm[idx]

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
bp.plot_measurements(np.asarray(muu_klstm)[:, 0], np.asarray(muu_klstm)[:, 1], color='green', label='Kalman LSTM',lw=1)
bp.plot_measurements(np.asarray(muu_lstm)[:, 0], np.asarray(muu_lstm)[:, 1], color='black', label='LSTM',lw=1)
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
sess.close()