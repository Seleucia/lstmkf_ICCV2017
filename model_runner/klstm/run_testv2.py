import numpy as np
import tensorflow as tf

import helper.kalman_libs as kl
from model_runner.klstm.garb import run_kalman as runner
from helper import train_helper as th
from helper import config
from model_runner.klstm.kf_QR import Model as kf_QR
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.lstm.tf_lstm import Model as lstm
from helper import dt_utils as dut
from helper import utils as ut

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4

num_epochs=10
params=config.get_params()

params['rnn_keep_prob']=1
params['R_std'] = 0.5
Q_std = 0.04
number_of_batches=400
params["batch_size"]=50

params['nlayer']= 1 #LSTM
params['Qnlayer'] = 1  # LSTM
params['Rnlayer'] = 1  # LSTM
params['Knlayer'] = 1  # LSTM
params['Flayer'] = 1  # LSTM
params['n_hidden']= 24
params['Qn_hidden']= 24
params['Rn_hidden']= 24
params['Kn_hidden']= 24
params['K_inp']=2
params["normalise_data"] = 0

seq_length=params['seq_length']
batch_size=params["batch_size"]
params["theta"]=0.3

idx=-1
max_epoch=20

consider=5
test_mode='robot' #step, rotation,combine
test_kalman=1
params['test_mode']=test_mode


if test_mode=='step4d':
    params['R_std'] = 0.2
    data_stream=kl.get_step2d_data
    params['n_output']=4
    filter=runner.test_filter
elif test_mode=='step2d':
    params['R_std'] = 0.4
    data_stream=kl.get_step2d_data
    params['n_output']=2
    params['n_input']=2
    filter=runner.test_filter2d
    test_kalman=0
elif test_mode=='ball':
    params['R_std'] = 0.4
    data_stream=kl.get_ball_motion
    params['n_output']=2
    params['n_input']=2
    filter=runner.test_filter2d
    test_kalman=0
elif test_mode=='robot':
    params['R_std'] = 0.4
    data_stream=kl.get_robot_motion
    params['n_output']=2
    params['n_input']=2
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


#
# if test_kalman==1:
#     print 'Kalman Testing:'
#     lst_kalman_err=[]
#     for i in range(Y_test.shape[0]):
#         measurements_idx=X_test[i]
#         groundtruth_idx=Y_test[i]
#         measurements_kalman=np.expand_dims(measurements_idx, 2)
#         kalman_mu, kalman_cov=filter(params,measurements_kalman)
#         kalman_mu=np.squeeze(kalman_mu)
#         loss=kl.compute_loss(gt=groundtruth_idx[consider:], est=kalman_mu[consider:])
#         lst_kalman_err.append(loss)
#     print "Kalman Loss: %f"% np.mean(lst_kalman_err)
# else:
#     print 'Kalman NOT tested'


def test_data(sess,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    dic_state=ut.get_state_list(params)
    P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])])*100
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)

    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_n_count=0.0
    prediction=[]
    actual_val=[]
    input_val=[]
    for minibatch_index in xrange(n_batches):
        state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
        (dic_state,x,y,r,f,state_reset_counter_lst)= \
            th.prepare_batch(index_list, minibatch_index, batch_size,
                                       S_list, dic_state, params, Y, X, R_L_list,F_list,state_reset_counter_lst)
        feed=th.get_feed(tracker,params,r,x,y,P,I,dic_state, is_training=0)

        states,final_output,y = \
            sess.run([tracker.states,
                      tracker.final_output,tracker.y], feed)

        for k in states.keys():
            tmp_lst=[]
            for item in  states[k]:
                tmp_lst.append(item[0])
                tmp_lst.append(item[1])
            dic_state[k]=tmp_lst

        if params["normalise_data"]==3 or params["normalise_data"]==2:
            final_output=ut.unNormalizeData(final_output,params["y_men"],params["y_std"])
            y=ut.unNormalizeData(y,params["y_men"],params["y_std"])
        prediction.append(final_output)
        actual_val.append(y)
        test_loss=kl.compute_loss(gt=y,est=final_output)
        total_loss+=test_loss*batch_size
        total_n_count+=batch_size
        # if (minibatch_index%show_every==0):
        #     print pre_test+" test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,test_loss)
    total_loss=total_loss/total_n_count
    s =pre_test+' Loss --> %s update %i | error %f'%(params["model"],e,total_loss)
    ut.log_write(s,params)
    return total_loss,prediction,actual_val

def train(tracker,params):
    P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])])*100
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
    batch_size=params["batch_size"]
    base_cp_path=params["cp_file"]+"/"

    decay_rate=0.3
    deca_start=2
    test_loss=[]
    pre_best_loss=10000.0
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        for e in range(num_epochs):
            if e>(deca_start-1):
                sess.run(tf.assign(tracker.lr, params['lr'] * (decay_rate ** ((e-deca_start)/2))))
            else:
                sess.run(tf.assign(tracker.lr, params['lr']))
            total_train_loss=0

            state_reset_counter_lst=[0 for i in range(batch_size)]
            index_train_list_s=index_train_list
            dic_state = ut.get_state_list(params)
            if params["shufle_data"]==1 and params['reset_state']==1:
                index_train_list_s = ut.shufle_data(index_train_list)

            for minibatch_index in xrange(n_train_batches):
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                (dic_state,x,y,r,f,state_reset_counter_lst)= \
                    th.prepare_batch(index_train_list_s, minibatch_index, batch_size,
                                       S_Train_list, dic_state, params, Y_train, X_train, R_L_Train_list,F_list_train,state_reset_counter_lst)

                feed = th.get_feed(tracker, params, r, x, y, P, I, dic_state, is_training=1)
                train_loss,states,_ = sess.run([tracker.cost,tracker.states,tracker.train_op], feed)

                for k in states.keys():
                    tmp_lst=[]
                    for item in  states[k]:
                        tmp_lst.append(item[0])
                        tmp_lst.append(item[1])
                    dic_state[k]=tmp_lst

                total_train_loss+=train_loss
            total_train_loss=total_train_loss/n_train_batches
            s='TRAIN --> %s epoch %i | error %f'%(params["model"],e*n_train_batches+minibatch_index, total_train_loss)
            ut.log_write(s,params)
            pre_test="TEST_Data"
            total_loss,ret,actual_val= test_data(sess,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,e*n_train_batches+minibatch_index, pre_test,n_test_batches)
            lss_str='%.5f' % total_loss
            if pre_best_loss>total_loss:
                pre_best_loss=total_loss
                model_name=lss_str+"_"+str(e)+"_"+str(params['process_uncertainty'])+params["model"]+"_vals"
                save_path=base_cp_path+model_name
                saved_path= np.save(save_path,ret)
                model_name=lss_str+"_"+str(e)+"_"+str(params['process_uncertainty'])+params["model"]+"_actual_vals"
                save_path=base_cp_path+model_name
                saved_path= np.save(save_path,actual_val)
            if saved_path != "":
                s='MODEL_Saved --> epoch %i | error %f path %s'%(e, total_loss,saved_path)
                ut.log_write(s,params)

            test_loss.append(total_loss)
    return test_loss

model_list=["kfl_QRf","kfl_QRFf","kfl_K","lstm"]
# model_list=["kfl_QRf","lstm"]
# model_list=["kfl_QRf"]
params['training_size']=0
params['test_size']=0
params['fseq_length']=200
params['seq_length']=50

ut.start_log(params)
ut.log_write("Models training started",params)
dict_loss={}

F,H=kl.build_matrices(params)
params['F_shape']=F.shape
params['F']=F[0]
params['F_mat']=F
params['H_mat']=H
params['H_shape']=H.shape
params['H']=H[0]
print "%s motion model testing"%test_mode
params["batch_size"]=10
batch_size=params["batch_size"]
number_of_seq=100
seq_lst=range(10,100,20)
R_lst=[0.2,0.4,0.6,1.0,1.5,2.0]
Proc_noise_lst=[5,7, 9,13,17,22]
params['lr']=0.001
for r in Proc_noise_lst:
    params['process_uncertainty'] = r
    (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,
     X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)= data_stream(params,number_of_seq)
    base_cp_path=params["cp_file"]+"/"
    # params["batch_size"] = 100

    (index_train_list,S_Train_list)=dut.get_seq_indexes(params,S_Train_list)
    (index_test_list,S_Test_list)=dut.get_seq_indexes(params,S_Test_list)
    np.save(base_cp_path+'X_test',X_test)
    np.save(base_cp_path+'Y_test',Y_test)
    np.save(base_cp_path+'S_Test_list',S_Test_list)
    np.save(base_cp_path+'index_test_list',index_test_list)
    np.save(base_cp_path+'R_L_Test_list',R_L_Test_list)
    np.save(base_cp_path+'S_Test_list',S_Test_list)
    np.save(base_cp_path+'F_list_test',F_list_test)

    batch_size=params['batch_size']
    n_train_batches = len(index_train_list)
    n_train_batches /= batch_size

    n_test_batches = len(index_test_list)
    n_test_batches /= batch_size

    print "=========================================="
    MeassErr=np.mean(np.sqrt(np.sum(np.square(X_test.reshape((X_test.shape[0]*X_test.shape[1],2))- Y_test.reshape((Y_test.shape[0]*Y_test.shape[1],2))),axis=1)))
    print "Measurment Error %f"%MeassErr

    print "-------------------------------------------"
    print r
    print "-------------------------------------------"
    for m in model_list:
        with tf.Graph().as_default():
            num_epochs = 10
            params["model"]=m
            print 'Training model:'+params["model"]
            if params["model"]=="kf_QR":
                tracker = kf_QR(params=params)
            elif params["model"]=="lstm":
                num_epochs = 15
                tracker = lstm(params=params)
            elif params["model"]=="kfl_QRf":
                tracker = kfl_QRf(params=params)
            elif params["model"]=="kfl_QRFf":
                tracker = kfl_QRFf(params=params)
            elif params["model"]=="kfl_K":
                tracker = kfl_K(params=params)

            dict_loss[m]=train(tracker,params)
