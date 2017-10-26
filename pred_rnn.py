import os
import numpy as np
import tensorflow as tf

from helper import dt_utils as dut
from helper import config
from helper import dt_utils as dt
from helper import utils as ut
from model_runner.lstm.noisy_lstm import  Model

params=config.get_params()

params["notes"]="inception_resnet_v2 testing on training and test dataset..." #running id
params['write_est']=True
params['n_input']= 48
params["data_bin"]='/mnt/Data1/hc/tt/training_temp/full-'+str(params['n_input'])+'.h5'
params["est_file"]="/mnt/Data1/hc/est/iv4/lstm/"
params["model_file"]='/home/coskun/PycharmProjects/poseftv4/files/cp-3thepoch/'
params["model"]='0.07120_4_pretrainedlstm102402081536_best_model.ckpt'
params["model"]='0.0.07443_5_iv2_seq50_res5lstm_best_model.ckpt'

params["cp_file"]='/home/coskun/PycharmProjects/poseftv4/files/cp/'+params["model"]
est_file=params["est_file"]+params["model"]

if not os.path.exists(est_file):
    os.makedirs(est_file)

est_file=est_file+"/"
# params['scope']='InceptionResnetV2'
# params['model_file']=params["model_file"]+'/'+"inception_resnet_v2_2016_08_30.ckpt"
# params['checkpoint_exclude_scopes']=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
params['run_mode']=2
params["test_lst_act"]=['S11','S9']
params["train_lst_act"]=['S1','S5','S6','S7','S8']
params['training_size']=0
params['test_size']=0
params['batch_size']=1
params['run_mode']=2 #Load previuesly trained model
is_training=False
# params['training_files']=dt.load_files(params,is_training=True)
# params['training_files']=([],[])
#Action List=
lst_bool=[False,True]
lst_action=['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting','SittingDown','Smoking','Waiting','Walking','WalkDog','WalkTogether']
# lst_action=['Posing']
(X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)=dut.prepare_training_set(params)
model = Model(params,is_training=is_training)

config = tf.ConfigProto(device_count = {'GPU': 0})
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
with tf.Session(config=config) as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver.restore(sess,params["cp_file"])
    print("Model loaded:%s"%params["model"])
    loss=0.
    total_cnt=0.
    test_write=True
    for action in lst_action:
        params["action"]=action
        if test_write==True:
            X,Y,F_list,G_list,S_list,R_L_list=dut.get_action_dataset(params,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)
        else:
            X,Y,F_list,G_list,S_list,R_L_list=dut.get_action_dataset(params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list)

        index_list,S_list=dut.get_seq_indexes(params,S_list)

        batch_size=params['batch_size']
        n_batches = len(index_list)
        n_batches /= batch_size

        LStateList_t=ut.get_zero_state(params)
        LStateList_pre=ut.get_zero_state(params)
        state_reset_counter_lst=[0 for i in range(batch_size)]
        total_loss=0.0
        total_n_count=0.0
        for minibatch_index in xrange(n_batches):
            state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
            # (LStateList_b,x,y,r,f,state_reset_counter_lst)=dut.prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,LStateList_t,LStateList_pre, params, Y, X,R_L_list,F_list,state_reset_counter_lst)
            (LStateList_b,x,y,r,f,state_reset_counter_lst)=\
            dut.prepare_lstm_batch(index_list, minibatch_index, batch_size,
                                   S_list,LStateList_t,LStateList_pre, params,
                                   Y, X,R_L_list,F_list,state_reset_counter_lst)
            LStateList_pre=LStateList_b
            # y=y.reshape(batch_size*params["seq_length"],params["n_output"])
            # feed = {model.input_data: x, model.target_data: y, model.initial_state: LStateList_b, model.repeat_data: r ,model.is_training:False,model.output_keep_prob:1.0}
            feed = {model.input_data: x, model.target_data: y, model.initial_state: LStateList_b, model.repeat_data: r ,model.is_training:False,model.output_keep_prob:1.0}
            LStateList_t,final_output,y = sess.run([model.final_state,model.final_output,model.y])
            test_loss,n_count=ut.get_loss(params,gt=y,est=final_output)
            # ut.write_rnn_est(est_file=est_file,est=final_output,file_names=f)
            # tmp_lst=[]
            # for item in  LStateList_t:
            #     tmp_lst.append(item.c)
            #     tmp_lst.append(item.h)
            # LStateList_t=tmp_lst
            total_loss+=test_loss*n_count
            total_n_count+=n_count
        total_loss=total_loss/total_n_count
        s ='%s Loss --> %f'%(action,total_loss)
        print(s)
        # ut.log_write(s,params)






        # test_loss= inception_output.eval(params)
        # cnt=len(params['test_files'][0])
        # loss=loss+test_loss*cnt
        # total_cnt=total_cnt+cnt
        # if tr==False:
        #     s ='TEST Set --> Action: %s, Frame Count: %i Final error %f'%(action,cnt,test_loss)
        # else:
        #     s ='Train Set --> Action: %s, Frame Count: %i Final error %f'%(action,cnt,test_loss)
        # ut.log_write(s,params)

    # loss=loss/total_cnt
    # if tr==False:
    #     s ='Total Test Frame Count: %i Final error %f'%(total_cnt,loss)
    # else:
    #     s ='Total Training Frame Count: %i Final error %f'%(total_cnt,loss)
    # ut.log_write(s,params)


    # ut.start_log(params)
    # ut.log_write("Model testing started",params)
    # test_loss=inception_eval.eval(params)
    # s ='VAL --> Final error %f'%(test_loss)
    # ut.log_write(s,params)
