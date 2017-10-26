import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import helper.kalman_libs as kl
from helper import config
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_K import Model as kfl_K
from helper import dt_utils as dut
from helper import utils as ut

params=config.get_params()
params["reload_data"]=0
params["model"]="lstm"
params=config.update_params(params)
params['is_forcasting']=0 #Current frame prediction
params["normalise_data"]=0 #Data will not be normalised
tracker = kfl_K(params=params)
global params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list,index_train_list,S_Train_list

(params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,
 X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)=\
    dut.prepare_training_set(params)


P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])])*100
I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.9

show_every=1

(index_train_list,S_Train_list)=dut.get_seq_indexes(params,S_Train_list)
(index_test_list,S_Test_list)=dut.get_seq_indexes(params,S_Test_list)

batch_size=params['batch_size']
n_train_batches = len(index_train_list)
n_train_batches /= batch_size

n_test_batches = len(index_test_list)
n_test_batches /= batch_size

params['training_size']=len(X_train)*params['seq_length']
params['test_size']=len(X_test)*params['seq_length']
ut.start_log(params)

ut.log_write("Model training started",params)

def test_data(sess,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    LStateList_F_t=ut.get_zero_state(params)
    LStateList_F_pre=ut.get_zero_state(params)
    LStateList_K_t = ut.get_zero_state(params, t='K')
    LStateList_K_pre = ut.get_zero_state(params, t='K')
    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_n_count=0.0
    for minibatch_index in xrange(n_batches):
        state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
        (LStateList_F_pre,LStateList_K_pre,_,x,y,r,f,state_reset_counter_lst)=\
            dut.prepare_kfl_QRFf_batch(index_list, minibatch_index, batch_size,
                                       S_list, LStateList_F_t, LStateList_F_pre, LStateList_K_t, LStateList_K_pre,
                                       None, None, params, Y, X, R_L_list,F_list,state_reset_counter_lst)
        gt= y
        mes=x
        # print(r)
        feed = {tracker._z: mes, tracker.target_data: gt,tracker.repeat_data: r, tracker.initial_state: LStateList_F_pre
        , tracker.initial_state_Q_noise: LStateList_K_pre,tracker.output_keep_prob:1}
        # feed = {tracker._z: mes, tracker.target_data: gt, tracker.initial_state: LStateList_F_pre
        #        , tracker._P_inp: P, tracker._I: I}
        LStateList_F_t,LStateList_K_t,final_output,y = \
            sess.run([tracker.final_state_F,tracker.final_state_K,
                      tracker.final_output,tracker.y], feed)

        tmp_lst=[]
        for item in  LStateList_F_t:
            tmp_lst.append(item.c)
            tmp_lst.append(item.h)
        LStateList_F_t=tmp_lst

        tmp_lst=[]
        for item in  LStateList_K_t:
            tmp_lst.append(item.c)
            tmp_lst.append(item.h)
        LStateList_K_t=tmp_lst

        # print(y)
        # print(y.shape)
        # print(final_output.shape)
        if params["normalise_data"]==3 or params["normalise_data"]==2:
            final_output=ut.unNormalizeData(final_output,params["y_men"],params["y_std"])
            y=ut.unNormalizeData(y,params["y_men"],params["y_std"])
        test_loss,n_count=ut.get_loss(params,gt=y,est=final_output)
        total_loss+=test_loss*n_count
        total_n_count+=n_count
        # if (minibatch_index%show_every==0):
        #     print pre_test+" test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,test_loss)
    total_loss=total_loss/total_n_count
    s =pre_test+' Loss --> epoch %i | error %f'%(e,total_loss)
    ut.log_write(s,params)
    return total_loss

consider=0
def train():
    batch_size=params["batch_size"]
    num_epochs=1000
    decay_rate=0.5
    show_every=100
    deca_start=2
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tracker.predict())
        print 'Training Noise KLSTM'
        noise_std = params['noise_std']
        new_noise_std=0.0
        for e in range(num_epochs):
            if e>(deca_start-1):
                sess.run(tf.assign(tracker.lr, params['lr'] * (decay_rate ** ((e-deca_start)/2))))
            else:
                sess.run(tf.assign(tracker.lr, params['lr']))
            total_train_loss=0
            LStateList_F_t=ut.get_zero_state(params)
            LStateList_F_pre=ut.get_zero_state(params)
            LStateList_K_t=ut.get_zero_state(params,t='K')
            LStateList_K_pre=ut.get_zero_state(params,t='K')
            state_reset_counter_lst=[0 for i in range(batch_size)]
            index_train_list_s=index_train_list
            if params["shufle_data"]==1 and params['reset_state']==1:
                index_train_list_s = ut.shufle_data(index_train_list)

            for minibatch_index in xrange(n_train_batches):
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                (LStateList_F_pre,LStateList_K_pre,_,x,y,r,f,state_reset_counter_lst)=\
                    dut.prepare_kfl_QRFf_batch(index_train_list_s, minibatch_index, batch_size,
                                               S_Train_list, LStateList_F_t, LStateList_F_pre, LStateList_K_t, LStateList_K_pre,
                                               None, None, params, Y_train, X_train, R_L_Train_list,F_list_train,state_reset_counter_lst)
                if noise_std >0.0:
                   u_cnt= e*n_train_batches + minibatch_index
                   if u_cnt in params['noise_schedule']:
                       new_noise_std=noise_std*(u_cnt/(params['noise_schedule'][0]))
                       s = 'NOISE --> u_cnt %i | error %f' % (u_cnt, new_noise_std)
                       ut.log_write(s, params)
                   if new_noise_std>0.0:
                       noise=np.random.normal(0.0,new_noise_std,x.shape)
                       x=noise+x

                gt= y
                mes=x
                feed = {tracker._z: mes, tracker.target_data: gt,tracker.repeat_data: r, tracker.initial_state: LStateList_F_pre
                , tracker.initial_state_K: LStateList_K_pre, tracker.output_keep_prob:params['rnn_keep_prob']}
                # feed = {tracker._z: mes, tracker.target_data: gt, tracker.initial_state: LStateList_F_pre
                #        , tracker._P_inp: P, tracker._I: I}
                train_loss,LStateList_F_t,LStateList_K_t,_ = \
                    sess.run([tracker.cost,tracker.final_state_F,tracker.final_state_Q,
                              tracker.train_op], feed)

                tmp_lst=[]
                for item in  LStateList_F_t:
                    tmp_lst.append(item.c)
                    tmp_lst.append(item.h)
                LStateList_F_t=tmp_lst

                tmp_lst=[]
                for item in  LStateList_K_t:
                    tmp_lst.append(item.c)
                    tmp_lst.append(item.h)
                LStateList_K_t=tmp_lst

                total_train_loss+=train_loss
                if (minibatch_index%show_every==0):
                    print "Training batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,
                                                                 train_loss)

            total_train_loss=total_train_loss/n_train_batches
            s='TRAIN --> epoch %i | error %f'%(e, total_train_loss)
            ut.log_write(s,params)
            pre_test="TEST_Data"
            total_loss= test_data(sess,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,e, pre_test,n_test_batches)
train()