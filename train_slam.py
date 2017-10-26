import numpy as np
import tensorflow as tf
import collections

from helper import train_helper as th
from helper import config
from model_runner.klstm.kfl_QRf_slam import Model as kfl_QRf
from model_runner.klstm.kfl_Rf_slam_mdn import Model as kfl_Rf
from model_runner.klstm.kfl_f_slam_mdn import Model as kfl_f
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.klstm.lstm_slam import Model as lstm
from helper import slam_helper as sh
from helper import utils as ut

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.70

def test_data(sess,params,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    dic_state=ut.get_state_list(params)
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)
    dict_err={}
    dict_name={}
    uniq_lst=[item for item in collections.Counter(S_list)]
    is_test=1

    file_lst=[]

    for u in uniq_lst:
        idx=np.where(S_list==u)
        sname=F_list[idx][0][0][0].split('/')[-2]
        dict_name[u]=sname
        dict_err[u]=[]

    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_n_count=0.0
    full_curr_id_lst=[]
    full_noise_lst=[]
    full_r_lst=[]
    full_y_lst=[]
    full_final_output_lst=[]
    for minibatch_index in xrange(n_batches):
        state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
        (dic_state,x_sel,y,r,f,curr_sid,state_reset_counter_lst,curr_id_lst)= \
            th.prepare_batch(is_test,index_list, minibatch_index, batch_size,
                                       S_list, dic_state, params, Y, X, R_L_list,F_list,state_reset_counter_lst)
        feed=th.get_feed(tracker,params,r,x_sel,y,I,dic_state, is_training=0)

        if params["model"] == "lstm":
            states,final_output,sel_y =sess.run([tracker.states,tracker.final_output,tracker.y], feed)
        else:
            states,final_output,full_final_output,sel_y,x,qnoise_lst =\
                sess.run([tracker.states,tracker.final_output,tracker.full_final_output,tracker.y,tracker.x,tracker.qnoise_lst], feed)
        full_final_output=np.asarray(full_final_output).reshape((batch_size,params['seq_length'],params['n_output']))
        for k in states.keys():
            dic_state[k]=states[k]

        full_curr_id_lst.extend(curr_id_lst)
        full_r_lst.extend(r)
        file_lst.extend(f)
        full_final_output_lst.extend(full_final_output)
        full_y_lst.extend(y)

        if params["model"] != "lstm":
            full_noise_lst.extend(qnoise_lst)

    # total_loss=total_loss/total_n_count

    index_lst=sh.get_nondublicate_lst(full_curr_id_lst)
    full_r_lst=np.asarray(full_r_lst)[index_lst]

    # if params["model"] != "lstm":
    #     full_noise_lst=np.asarray(full_noise_lst)[index_lst]
    #     full_noise_lst=full_noise_lst[full_r_lst==1]



    full_final_output_lst=np.asarray(full_final_output_lst)[index_lst]
    full_y_lst=np.asarray(full_y_lst)[index_lst]
    file_lst=np.asarray(file_lst)[index_lst]

    file_lst=file_lst[full_r_lst==1]
    full_final_output_lst=full_final_output_lst[full_r_lst==1]
    full_y_lst=full_y_lst[full_r_lst==1]
    dict_err={}


    if params["normalise_data"]==3 or params["normalise_data"]==2:
        full_final_output_lst=ut.unNormalizeData(full_final_output_lst,params["y_men"],params["y_std"])
        full_y_lst=ut.unNormalizeData(full_y_lst,params["y_men"],params["y_std"])

    if params["normalise_data"]==4:
        full_final_output_lst=ut.unNormalizeData(full_final_output_lst,params["x_men"],params["x_std"])
        full_y_lst=ut.unNormalizeData(full_y_lst,params["x_men"],params["x_std"])

    full_loss,dict_err=sh.get_loss(file_lst,gt=full_y_lst,est=full_final_output_lst)
    # np.savetxt('trials/garb/x',np.asarray(x_lst))
    if params["sequence"]=="David":
        for  u in dict_err.keys():
            seq_err=dict_err[u]
            median_result= np.median(seq_err,axis=0)
            mean_result= np.mean(seq_err,axis=0)
            print 'Epoch:',e,' full ',u ,' median/mean error ', median_result[0],'/', mean_result[0], 'm  and ', median_result[1],'/', mean_result[1], 'degrees.'

    else:
        median_result= np.median(full_loss,axis=0)
        mean_result= np.mean(full_loss,axis=0)
        if params["data_mode"]=="xyx":
            print 'Epoch:',e,' full sequence median/mean error ', median_result[0],'/', mean_result[0],''
        elif params["data_mode"]=="q":
            print 'Epoch:',e,' full sequence median/mean error ', median_result[0],'/', mean_result[0], 'degrees.'
        else:
            print 'Epoch:',e,' full sequence median/mean error ', median_result[0],'/', mean_result[0], 'm  and ', median_result[1],'/', mean_result[1], 'degrees.'


    # s =pre_test+' Loss --> epoch %i | error %f'%(e,total_loss)
    # ut.log_write(s,params)
    return total_loss,median_result,mean_result,full_final_output_lst,file_lst,full_noise_lst

def train(tracker,params):
    I= np.asarray([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])],dtype=np.float32)

    batch_size=params["batch_size"]

    decay_rate=0.95
    # show_every=100
    deca_start=10
    # pre_best_loss=10000
    with tf.Session(config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        # sess.run(tracker.predict())
        print 'Training model:'+params["model"]
        noise_std = params['noise_std']
        new_noise_std=0.0
        median_result_lst=[]
        mean_result_lst=[]
        for e in range(num_epochs):
            if e==2:
                params['lr']=params['lr']
            if e>(deca_start-1):
                sess.run(tf.assign(tracker.lr, params['lr'] * (decay_rate ** (e))))
            else:
                sess.run(tf.assign(tracker.lr, params['lr']))
            total_train_loss=0

            state_reset_counter_lst=[0 for i in range(batch_size)]
            index_train_list_s=index_train_list
            dic_state = ut.get_state_list(params)
            if params["shufle_data"]==1 and params['reset_state']==1:
                index_train_list_s = ut.shufle_data(index_train_list)

            for minibatch_index in xrange(n_train_batches):
                is_test=0
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                (dic_state,x,y,r,f,_,state_reset_counter_lst,_)= \
                    th.prepare_batch(is_test,index_train_list_s, minibatch_index, batch_size,
                                       S_Train_list, dic_state, params, Y_train, X_train, R_L_Train_list,F_list_train,state_reset_counter_lst)
                if noise_std >0.0:
                   u_cnt= e*n_train_batches + minibatch_index
                   if u_cnt in params['noise_schedule']:
                       new_noise_std=noise_std*(u_cnt/(params['noise_schedule'][0]))
                       s = 'NOISE --> u_cnt %i | error %f' % (u_cnt, new_noise_std)
                       ut.log_write(s, params)
                   if new_noise_std>0.0:
                       noise=np.random.normal(0.0,new_noise_std,x.shape)
                       x=noise+x

                feed = th.get_feed(tracker, params, r, x, y, I, dic_state, is_training=1)
                train_loss,states,_ = sess.run([tracker.cost,tracker.states,tracker.train_op], feed)
                # print last_pred.shape
                # print states.shape


                for k in states.keys():
                    dic_state[k]=states[k]

                total_train_loss+=train_loss
            # if e%5==0:
            #         print total_train_loss
            pre_test="TEST_Data"
            total_loss,median_result,mean_result,final_output_lst,file_lst,noise_lst= test_data(sess,params,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,e, pre_test,n_test_batches)
            if len(full_median_result_lst)>1:
                if median_result[0]<np.min(full_median_result_lst,axis=0)[0]:
                    # ut.write_slam_est(est_file=params["est_file"],est=final_output_lst,file_names=file_lst)
                #     ut.write_slam_est(est_file=params["noise_file"],est=noise_lst,file_names=file_lst)
                #     save_path=params["cp_file"]+params['msg']
                    # saver.save(sess,save_path)
                    print 'Writing estimations....'

            full_median_result_lst.append(median_result)
            median_result_lst.append(median_result)
            mean_result_lst.append(mean_result)
            # base_cp_path = params["cp_file"] + "/"
            #
            # lss_str = '%.5f' % total_loss
            # model_name = lss_str + "_" + str(e) + "_" + str(params["rn_id"]) + params["model"] + "_model.ckpt"
            # save_path = base_cp_path + model_name
            # saved_path = False
            # if pre_best_loss > total_loss:
            #     pre_best_loss = total_loss
            #     model_name = lss_str + "_" + str(e) + "_" + str(params["rn_id"]) + params["model"] + "_best_model.ckpt"
            #     save_path = base_cp_path + model_name
            #     saved_path = saver.save(sess, save_path)
            # else:
            #     if e % 3.0 == 0:
            #         saved_path = saver.save(sess, save_path)
            # if saved_path != "":
            #     s = 'MODEL_Saved --> epoch %i | error %f path %s' % (e, total_loss, saved_path)
            #     ut.log_write(s, params)
    return median_result_lst,mean_result_lst

# rnn_keep_prob_lst=[0.5]
# seq_lst=[5,10,15,20]
# reset_state=[100]
# hidden_units=[2,4,8,12,16]
# normalise_data_lst=[0]
# nrun=10


params = config.get_params()
params["sequence"]="OldHospital"#"ShopFacade",KingsCollege,Street,OldHospital,David,StMarysChurch
#KingsCollege-,ShopFacade,Street-,OldHospital-,StMarysChurch, David-
#KingsCollege-,ShopFacade,Street-,OldHospital-,StMarysChurch, David
# params['lr'] = 0.0005
params['lr'] = 0.001
params["grad_clip"]=100
seq_lst=[10]
hidden_units=[16]
reset_state=[200]
nrun=10
num_epochs=100

params["model"] = "kfl_QRf"
params["train_mode"]="partial"

if params["sequence"]=="David":
    params["train_mode"]="partial"

params["est_file"]='/home/coskun/PycharmProjects/poseft/trials/garb/'+params["model"]+'/'+params["train_mode"]+'/'+params["sequence"]+'/'
params["noise_file"]='/home/coskun/PycharmProjects/poseft/trials/noise/'+params["sequence"]+'/'
params["cp_file"]='/home/coskun/PycharmProjects/poseft/trials/models/'
params["data_dir"]="/home/coskun/PycharmProjects/PoseNet/slam/"
params["data_mode"]='xyzq'

beta_lst= [1000]
layers=[1]
res=reset_state[0]
dic_res={}



# params["data_mode"]="xyzq"#"q","xyzq"
params['input_keep_prob']=1.0

params,db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test=\
    sh.load_flat_data(params)

full_median_result_lst=[]
for s in seq_lst:
    params['seq_length']=s
    X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list=sh.prepare_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
    X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list=sh.prepare_sequences(params,db_values_x_test,db_values_y_test,db_names_test)
    (index_train_list, S_Train_list) = sh.get_seq_indexes(params, S_Train_list)
    (index_test_list, S_Test_list) = sh.get_seq_indexes(params, S_Test_list)
    batch_size = params['batch_size']
    n_train_batches = len(index_train_list)
    n_train_batches /= batch_size
    n_test_batches = len(index_test_list)
    n_test_batches /= batch_size
    params['training_size'] = len(X_train) * params['seq_length']
    params['test_size'] = len(X_test) * params['seq_length']
    for b in beta_lst:
        for h in hidden_units:
            for r in range(nrun):
                with tf.Graph().as_default():
                    msg='r'+str(r)+'_h'+str(h)+'_s'+str(s)
                    params['msg']=msg
                    print "msg: ============== %s  ============" % msg
                    params["rn_id"]="l2loss_seq%i_res%i"%(h,s)
                    params['reset_state']=res
                    params['beta']= b #LSTM
                    params['nlayer']= 1 #LSTM

                    params['Qnlayer'] = 1  # LSTM
                    params['Rnlayer'] = 1  # LSTM
                    params['Knlayer'] = 1  # LSTM
                    params['Flayer'] =1  # LSTM
                    params['P_mul']= 1000
                    if params["data_mode"]=="xyz":
                        params['n_output']= 3
                        params['n_input']= 3
                    elif params["data_mode"]=="q":
                        params['n_output']= 4
                        params['n_input']= 4
                    # elif params["data_mode"]=="David":
                    #     params['n_output']= 12
                    #     params['n_input']= 12
                    else:
                        params['n_output']= 7
                        params['n_input']= 7

                    params['n_hidden']= h
                    params['Qn_hidden']= h
                    params['Rn_hidden']= h
                    params['Kn_hidden']= h

                    if params["model"] == "lstm":
                        tracker = lstm(params=params)
                    elif params["model"] == "kfl_QRf":
                        tracker = kfl_QRf(params=params)
                    elif params["model"] == "kfl_Rf":
                        tracker = kfl_Rf(params=params)
                    elif params["model"] == "kfl_f":
                        tracker = kfl_f(params=params)
                    elif params["model"] == "kfl_QRFf":
                        tracker = kfl_QRFf(params=params)
                    elif params["model"] == "kfl_K":
                        tracker = kfl_K(params=params)
                    params=config.update_params(params)

                    show_every = 1

                    ut.start_log(params)
                    ut.log_write("Model training started", params)
                    median_result_lst,mean_result_lst= train(tracker,params)
                    # if params["data_mode"]=="David":
                    #     np.savetxt('/home/coskun/PycharmProjects/poseft/trials/res/'+params["sequence"]+'/'+msg,mean_result_lst)
                    # else:
                    # np.savetxt('/home/coskun/PycharmProjects/poseft/trials/res/klstm/'+params["sequence"]+'/'+msg,median_result_lst)
                    # np.min(median_result_lst,axis=1)
                    # print median_result_lst
                    # print mean_result_lst
                    # print 'Min...'
