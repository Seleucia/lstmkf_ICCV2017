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


model_list=["kfl_QRf","kfl_QRFf","kfl_K","lstm"]
# model_list=["kfl_QRf","lstm"]
# model_list=["kfl_QRf"]
params['training_size']=0
params['test_size']=0
params['fseq_length']=200
params['seq_length']=20

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
m="kfl_QRf"

mfile="0.07584_0_5kfl_QRf_best_model.ckpt"

def test_data(sess,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    dic_state=ut.get_state_list(params)
    P= np.asarray([np.diag([1]*params['n_output']) for i in range(params["batch_size"])])*100
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)

    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_n_count=0.0
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
        test_loss=kl.compute_loss(gt=y,est=final_output)
        total_loss+=test_loss*batch_size
        total_n_count+=batch_size
        # if (minibatch_index%show_every==0):
        #     print pre_test+" test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,test_loss)
    total_loss=total_loss/total_n_count
    s =pre_test+' Loss --> %s update %i | error %f'%(params["model"],e,total_loss)
    ut.log_write(s,params)
    return total_loss

with tf.Graph().as_default():
    num_epochs = 20
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
    base_cp_path=params["cp_file"]+"/"
    X_test=np.load(base_cp_path+'X_test.npy')
    Y_test=np.load(base_cp_path+'Y_test.npy')
    S_Test_list=np.load(base_cp_path+'S_Test_list.npy')
    index_test_list=np.load(base_cp_path+'index_test_list.npy')
    R_L_Test_list=np.load(base_cp_path+'R_L_Test_list.npy')
    S_Test_list=np.load(base_cp_path+'S_Test_list.npy')
    n_test_batches = len(index_test_list)
    n_test_batches /= batch_size
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        save_path=base_cp_path+mfile
        saver.restore(sess,save_path)
        total_loss= test_data(sess,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,0, "TEST",n_test_batches)



