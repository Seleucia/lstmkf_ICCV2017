import numpy as np
import tensorflow as tf
import os

from helper import train_helper as th
from helper import config
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.lstm.tf_lstm import Model as lstm
from helper import dt_utils as dut
from helper import utils as ut
#
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.95
# gpu_config = tf.ConfigProto(
#     device_count={'GPU':0}
# )
def test_data(sess,params,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    dic_state=ut.get_state_list(params)
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
    is_test=1

    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_n_count=0.0
    for minibatch_index in xrange(n_batches):
        state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
        (dic_state,x,y,r,f,_,state_reset_counter_lst,_)= \
            th.prepare_batch(is_test,index_list, minibatch_index, batch_size,
                                       S_list, dic_state, params, Y, X, R_L_list,F_list,state_reset_counter_lst)
        feed=th.get_feed(tracker,params,r,x,y,I,dic_state, is_training=0)

        if mode == 'klstm':
            states,final_output,final_pred_output,final_meas_output,q_mat,r_mat,k_mat,y =\
                sess.run([tracker.states,tracker.final_output,tracker.final_pred_output,tracker.final_meas_output,
                      tracker.final_q_output,tracker.final_r_output,tracker.final_k_output,tracker.y], feed)
        else:
            states, final_output, y = \
                sess.run([tracker.states, tracker.final_output, tracker.y], feed)

        for k in states.keys():
            dic_state[k] = states[k]

        if params["normalise_data"]==3 or params["normalise_data"]==2:
            final_output=ut.unNormalizeData(final_output,params["y_men"],params["y_std"])
            y=ut.unNormalizeData(y,params["y_men"],params["y_std"])

        if params["normalise_data"]==4:
            final_output=ut.unNormalizeData(final_output,params["x_men"],params["x_std"])
            y = ut.unNormalizeData(y, params["x_men"], params["x_std"])
            if mode == 'klstm':
                final_pred_output=ut.unNormalizeData(final_pred_output,params["x_men"],params["x_std"])
                final_meas_output=ut.unNormalizeData(final_meas_output,params["x_men"],params["x_std"])


        test_loss,n_count=ut.get_loss(params,gt=y,est=final_output,r=None)
        f=f.reshape((-1, 2))
        y_f=y.reshape(final_output.shape)
        r=r.flatten()
        fnames=f[np.nonzero(r)]
        # e=final_output[np.nonzero(r)]
        if mode == 'klstm':
            ut.write_est(est_file=params["est_file"]+"/kal_est/",est=final_output,file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/kal_est_dif/",est=np.abs(final_output-y_f),file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/kal_pred/",est=final_pred_output,file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/kal_pred_dif/",est=np.abs(final_pred_output-y_f),file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/meas/",est=final_meas_output,file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/q_mat/",est=q_mat,file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/r_mat/",est=r_mat,file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/k_mat/",est=k_mat,file_names=fnames)
            ut.write_est(est_file=params["est_file"]+"/y_f/",est=y_f,file_names=fnames)
        else:
            ut.write_est(est_file=params["est_file"], est=final_output, file_names=fnames)
        # print test/_loss
        total_loss+=test_loss*n_count

        total_n_count+=n_count
        print total_loss / total_n_count
        # if (minibatch_index%show_every==0):
        #     print pre_test+" test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,test_loss)
    total_loss=total_loss/total_n_count
    s =pre_test+' Loss --> epoch %i | error %f'%(e,total_loss)
    ut.log_write(s,params)
    return total_loss
mode='lstm'


params = config.get_params()
if mode=='klstm':
    params["data_bin"] = '/mnt/Data1/hc/tt/training_temp/cp-1stepoch-ntu-' + str(params['n_input']) + '.h5'
    params["mfile"] = "/mnt/Data1/hc/tt/cp/ires/klstm_sel"

    # params["data_bin"] = '/mnt/Data1/hc/tt/training_temp/full2-48.h5'
    # params["mfile"] = "/mnt/Data1/hc/tt/cp/lstm3/cp_sel"
else:
    params["data_bin"]='/mnt/Data1/hc/tt/training_temp/full2-'+str(params['n_input'])+'.h5'
    params["data_bin"]='/mnt/Data1/hc/tt/training_temp/full2-48.h5'
    # params["data_bin"]='/mnt/Data1/hc/tt/training_temp/fulliv4_bb-48.h5'
    params["mfile"] = "/mnt/Data1/hc/tt/cp/lstm2/"

params["est_file"]="/mnt/Data1/hc/est/lstm_supp/"
# params["mfile"]="/home/coskun/PycharmProjects/poseftv4/files/cp_mp/0.07181_8_iv2_seq100_res1lstm_best_model.ckpt"

params['rnn_keep_prob']=0.7
params['reset_state']=1000
params['normalise_data'] = 3
params['seq_length'] = 50
params["reload_data"] = 0
params["model"] = "lstm"
params['batch_size']=10
params['input_keep_prob']=1

with tf.Graph().as_default():
    if params["model"] == "lstm":
        tracker = lstm(params=params)
    elif params["model"] == "kfl_QRf":
        tracker = kfl_QRf(params=params)
    elif params["model"] == "kfl_QRFf":
        tracker = kfl_QRFf(params=params)
    elif params["model"] == "kfl_K":
        tracker = kfl_K(params=params)
    params=config.update_params(params)
    (params, X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list,
             X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list) = \
            dut.prepare_training_set(params)
    show_every = 1
    (index_train_list, S_Train_list) = dut.get_seq_indexes(params, S_Train_list)
    (index_test_list, S_Test_list) = dut.get_seq_indexes(params, S_Test_list)
    batch_size = params['batch_size']
    n_train_batches = len(index_train_list)
    n_train_batches /= batch_size

    n_test_batches = len(index_test_list)
    n_test_batches /= batch_size
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(params["mfile"])
        if ckpt and ckpt.model_checkpoint_path:
            print "Model found:%s, %s" % (params["model"],ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            mfile=ckpt.model_checkpoint_path
            params["est_file"]=params["est_file"]+mfile.split('/')[-1].replace('.ckpt','')+'_v3/'
            print "Loaded Model: %s"% ckpt.model_checkpoint_path
            # for var in tracker.tvars:
            #     print 'Variables saved %s' % var.name
            #     val=sess.run(var)
            #     path = '/mnt/Data1/hc/tt/cp/weights/' + var.name
            #     dir_name=os.path.dirname(path)
            #     print dir_name
            #     if not os.path.exists(dir_name):
            #         os.makedirs(dir_name)
            #     np.save(path, val)
            #     # sv=tf.train.Saver({var.name:var})
            #     # sv.save('/mnt/Data1/hc/tt/cp/weights/')


        else:
            print "No file found...."

        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess, params["mfile"])
        pre_test="TEST_Data"
        total_loss= test_data(sess,params,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,0, pre_test,n_test_batches)