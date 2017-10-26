import numpy as np
import os
import tensorflow as tf

from helper import train_helper as th
from helper import config
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_Rf_mdn import Model as kfl_Rf
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.lstm.tf_lstm import Model as lstm
from helper import dt_utils as dut
from helper import utils as ut

# gpu_config = tf.ConfigProto(
#     device_count={'GPU':0}
# )
gpu_config = tf.ConfigProto()
# gpu_config.de
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.45

def test_data(sess,params,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    is_test=1
    dic_state=ut.get_state_list(params)
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
    params["reset_state"]=-1 #Never reset

    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_pred_loss=0.0
    total_meas_loss=0.0
    total_n_count=0.0
    for minibatch_index in xrange(n_batches):
        state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
        # print state_reset_counter_lst
        (dic_state,x,y,r,f,_,state_reset_counter_lst,_)= \
            th.prepare_batch(is_test,index_list, minibatch_index, batch_size,
                                       S_list, dic_state, params, Y, X, R_L_list,F_list,state_reset_counter_lst)
        feed=th.get_feed(tracker,params,r,x,y,I,dic_state, is_training=0)

        states,final_output,final_pred_output,final_meas_output,y =sess.run([tracker.states,tracker.final_output,tracker.final_pred_output,tracker.final_meas_output,tracker.y], feed)

        for k in states.keys():
            dic_state[k] = states[k]

        if params["normalise_data"]==3 or params["normalise_data"]==2:
            final_output=ut.unNormalizeData(final_output,params["y_men"],params["y_std"])
            final_pred_output=ut.unNormalizeData(final_pred_output,params["y_men"],params["y_std"])
            final_meas_output=ut.unNormalizeData(final_meas_output,params["x_men"],params["x_std"])
            y=ut.unNormalizeData(y,params["y_men"],params["y_std"])
        if params["normalise_data"]==4:
            final_output=ut.unNormalizeData(final_output,params["x_men"],params["x_std"])
            final_pred_output=ut.unNormalizeData(final_pred_output,params["x_men"],params["x_std"])
            final_meas_output=ut.unNormalizeData(final_meas_output,params["x_men"],params["x_std"])
            y=ut.unNormalizeData(y,params["x_men"],params["x_std"])

        test_loss,n_count=ut.get_loss(params,gt=y,est=final_output,r=r)
        test_pred_loss,n_count=ut.get_loss(params,gt=y,est=final_pred_output,r=r)
        test_meas_loss,n_count=ut.get_loss(params,gt=y,est=final_meas_output,r=r)
        total_loss+=test_loss*n_count
        total_pred_loss+=test_pred_loss*n_count
        total_meas_loss+=test_meas_loss*n_count
        total_n_count+=n_count
        # if (minibatch_index%show_every==0):
        #     print pre_test+" test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,test_loss)
    total_loss=total_loss/total_n_count
    total_pred_loss=total_pred_loss/total_n_count
    total_meas_loss=total_meas_loss/total_n_count
    s =pre_test+' Loss --> epoch %i | error %f, %f, %f'%(e,total_loss,total_pred_loss,total_meas_loss)
    ut.log_write(s,params)
    return total_loss

def train(tracker,params):
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)

    batch_size=params["batch_size"]
    num_epochs=1000
    decay_rate=0.9
    show_every=100
    deca_start=3
    pre_best_loss=10000
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        # if params["model"] == "kfl_QRf":
            # ckpt = tf.train.get_checkpoint_state(params["mfile"])
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            #     mfile = ckpt.model_checkpoint_path
            #     params["est_file"] = params["est_file"] + mfile.split('/')[-1].replace('.ckpt', '') + '/'
            #     print "Loaded Model: %s" % ckpt.model_checkpoint_path
        # if params["model"] == "kfl_QRf":
        #     for var in tracker.tvars:
        #         path = '/mnt/Data1/hc/tt/cp/weights/' + var.name.replace('transitionF/','')
        #         if os.path.exists(path+'.npy'):
        #             val=np.load(path+'.npy')
        #             sess.run(tf.assign(var, val))
        #     print 'PreTrained LSTM model loaded...'


        # sess.run(tracker.predict())
        print 'Training model:'+params["model"]
        noise_std = params['noise_std']
        new_noise_std=0.0
        for e in range(num_epochs):
            if e>(deca_start-1):
                sess.run(tf.assign(tracker.lr, params['lr'] * (decay_rate ** (e))))
            else:
                sess.run(tf.assign(tracker.lr, params['lr']))
            total_train_loss=0

            state_reset_counter_lst=[0 for i in range(batch_size)]
            index_train_list_s=index_train_list
            dic_state = ut.get_state_list(params)
            # total_loss = test_data(sess, params, X_test, Y_test, index_test_list, S_Test_list, R_L_Test_list,
            #                        F_list_test, e, 'Test Check', n_test_batches)
            if params["shufle_data"]==1 and params['reset_state']==1:
                index_train_list_s = ut.shufle_data(index_train_list)

            for minibatch_index in xrange(n_train_batches):
                is_test = 0
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                (dic_state,x,y,r,f,_,state_reset_counter_lst,_)= \
                    th.prepare_batch(is_test,index_train_list_s, minibatch_index, batch_size,
                                       S_Train_list, dic_state, params, Y_train, X_train, R_L_Train_list,F_list_train,state_reset_counter_lst)
                if noise_std >0.0:
                   u_cnt= e*n_train_batches + minibatch_index
                   if u_cnt in params['noise_schedule']:
                       if u_cnt==params['noise_schedule'][0]:
                         new_noise_std=noise_std
                       else:
                           new_noise_std = noise_std * (u_cnt / (params['noise_schedule'][1]))

                       s = 'NOISE --> u_cnt %i | error %f' % (u_cnt, new_noise_std)
                       ut.log_write(s, params)
                   if new_noise_std>0.0:
                       noise=np.random.normal(0.0,new_noise_std,x.shape)
                       x=noise+x

                feed = th.get_feed(tracker, params, r, x, y, I, dic_state, is_training=1)
                train_loss,states,_ = sess.run([tracker.cost,tracker.states,tracker.train_op], feed)

                for k in states.keys():
                    dic_state[k] = states[k]

                total_train_loss+=train_loss
                if (minibatch_index%show_every==0):
                    print "Training batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,
                                                                 train_loss)

            total_train_loss=total_train_loss/n_train_batches
            s='TRAIN --> epoch %i | error %f'%(e, total_train_loss)
            ut.log_write(s,params)

            pre_test = "TRAINING_Data"
            total_loss = test_data(sess, params, X_train, Y_train, index_train_list, S_Train_list, R_L_Train_list,
                                   F_list_train, e, pre_test, n_train_batches)

            pre_test="TEST_Data"
            total_loss= test_data(sess,params,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,e, pre_test,n_test_batches)
            base_cp_path = params["cp_file"] + "/"

            lss_str = '%.5f' % total_loss
            model_name = lss_str + "_" + str(e) + "_" + str(params["rn_id"]) + params["model"] + "_model.ckpt"
            save_path = base_cp_path + model_name
            saved_path = False
            if pre_best_loss > total_loss:
                pre_best_loss = total_loss
                model_name = lss_str + "_" + str(e) + "_" + str(params["rn_id"]) + params["model"] + "_best_model.ckpt"
                save_path = base_cp_path + model_name
                saved_path = saver.save(sess, save_path)
            else:
                if e % 3.0 == 0:
                    saved_path = saver.save(sess, save_path)
            if saved_path != "":
                s = 'MODEL_Saved --> epoch %i | error %f path %s' % (e, total_loss, saved_path)
                ut.log_write(s, params)

rnn_keep_prob_lst=[0.8]
rnn_input_prob_lst=[1.0]
seq_lst=[50]
reset_state=[5,100,20]
normalise_data_lst=[3]
params = config.get_params()
params["mfile"]='/mnt/Data1/hc/tt/cp/lstm_nostate1/cp/'
rnn_keep_prob=0.8
input_keep_prob=1.0
params['rnn_keep_prob']=rnn_keep_prob
params['input_keep_prob']=input_keep_prob
seq=50
res=5
with tf.Graph().as_default():
    print "seq: ============== %s  ============" % seq
    print "reset_state: ============== %s  ============" % res
    print "rnn_keep_prob: ============== %s  ============" % rnn_keep_prob
    params['normalise_data'] = 4
    params['reset_state']=res
    params['seq_length']=seq
    params["reload_data"] = 0
    params = config.update_params(params)
    params["model"] = "kfl_QRf"
    if params["model"] == "lstm":
        tracker = lstm(params=params)
    elif params["model"] == "kfl_QRf":
        tracker = kfl_QRf(params=params)
    elif params["model"] == "kfl_Rf":
        tracker = kfl_Rf(params=params)
    elif params["model"] == "kfl_QRFf":
        tracker = kfl_QRFf(params=params)
    elif params["model"] == "kfl_K":
        tracker = kfl_K(params=params)
    params["rn_id"]="dobuleloss081500_nrm4_seq%i_res%i_keep%f_lr%f"%(seq,res,rnn_keep_prob,params["lr"])
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
    params['training_size'] = len(X_train) * params['seq_length']
    params['test_size'] = len(X_test) * params['seq_length']
    ut.start_log(params)
    ut.log_write("Model training started", params)
    train(tracker,params)