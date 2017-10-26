import time

import numpy as np
import tensorflow as tf

from helper import config
from helper import dt_utils as dut
from helper import utils as ut

# from model_runner.rnn_lstm import  Model
from model_runner.mlp.mlp import  Model

params=config.get_params()
params['batch_size']=200
params["model"]="mlp"
params=config.update_params(params)
#load model.....
model = Model(params)
params['n_param']=model.total_parameters
print('Model loaded....%i'%params['n_param'])
(X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)=dut.prepare_training_set(params)


X_train=X_train.reshape((-1,params['n_input']))
X_test=X_test.reshape((-1,params['n_input']))
Y_train=Y_train.reshape((-1,params['n_output']))
Y_test=Y_test.reshape((-1,params['n_output']))
# men=np.mean(X_train)
# s=np.std(X_train)
# X_train=X_train-men
# X_train=X_train/s

# X_test=X_test-men
# X_test=X_test/s

index_train_list,S_Train_list=dut.get_seq_indexes(params,S_Train_list)
index_test_list,S_Test_list=dut.get_seq_indexes(params,S_Test_list)

batch_size=params['batch_size']
n_train_batches = X_train.shape[0]
n_train_batches /= batch_size

n_test_batches =  X_test.shape[0]
n_test_batches /= batch_size

params['training_size']= X_train.shape[0]
params['test_size']= X_test.shape[0]
ut.start_log(params)
ut.log_write("Train mean:%f/%f, max:%f/%f, min:%f/%f ; Test mean:%f/%f, max:%f/%f, min:%f/%f "
             ";"%(np.mean(X_train),np.mean(Y_train),np.max(X_train),np.max(Y_train),
                  np.min(X_train),np.min(Y_train),np.mean(X_test),np.mean(Y_test),
                  np.max(X_test),np.max(Y_test),np.min(X_test),np.min(Y_test)),params)
ut.log_write("Model training started",params)
# summary_writer = tf.train.SummaryWriter(params["sm"])
show_every=100000.0

def test_data(sess,X,Y,index_list,S_list,R_L_list,F_list,e, pre_test,n_batches):
    state_reset_counter_lst=[0 for i in range(batch_size)]
    total_loss=0.0
    total_losss=0.0
    total_n_count=0.0
    total_n_countt=0.0
    for minibatch_index in xrange(n_batches):
        state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
        x=X[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
        y=Y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
        feed = {model.input_data: x, model.target_data: y, model.is_training:False,model.output_keep_prob:1.0}
        final_output = sess.run([model.final_output], feed)
        final_output=final_output[0]
        test_loss,n_count=ut.get_loss(params,gt=y,est=final_output)
        test_losss,n_countt=ut.get_loss(params,gt=y,est=x)
        total_loss+=test_loss*n_count
        total_losss+=test_losss*n_countt
        total_n_count+=n_count
        total_n_countt+=n_countt
        if (minibatch_index%show_every==0):
            print pre_test+" test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,test_loss)
    total_loss=total_loss/total_n_count
    total_losss=total_losss/total_n_countt
    s =pre_test+' Loss --> epoch %i | error %f, %f'%(e,total_loss,total_losss)
    ut.log_write(s,params)
    return total_loss

def train(X_train,Y_train,X_test,Y_test):
    num_epochs=1000
    decay_rate=0.5
    pre_best_loss=10000.0

    # config = tf.ConfigProto(device_count = {'GPU': 0})
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(params["sm"], sess.graph)
        for e in xrange(num_epochs):
            X_train,Y_train=ut.unison_shuffled_copies(X_train,Y_train)
            if e>1:
                sess.run(tf.assign(model.lr, params['lr'] * (decay_rate ** (e-1))))
            else:
                sess.run(tf.assign(model.lr, params['lr']))
            state_reset_counter_lst=[0 for i in range(batch_size)]
            total_train_loss=0
            for minibatch_index in xrange(n_train_batches):
                start = time.time()
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                x=X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                y=Y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
                feed = {model.input_data: x, model.target_data: y,model.is_training:True, model.output_keep_prob:0.8}
                summary,train_loss,_ =\
                    sess.run([merged,model.cost, model.train_op], feed)
                summary_writer.add_summary(summary, minibatch_index)
                total_train_loss+=train_loss
                if (minibatch_index%show_every==0):
                    print "Training batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,train_loss)

            total_train_loss=total_train_loss/n_train_batches
            s='TRAIN --> epoch %i | error %f'%(e, total_train_loss)
            ut.log_write(s,params)

            pre_test="TRAINING_Data"
            test_data(sess,X_train,Y_train,index_train_list,S_Train_list,R_L_Train_list,F_list_train,e, pre_test,n_train_batches)

            pre_test="TEST_Data"
            total_loss= test_data(sess,X_test,Y_test,index_test_list,S_Test_list,R_L_Test_list,F_list_test,e, pre_test,n_test_batches)
            base_cp_path=params["cp_file"]+"/"

            lss_str='%.5f' % total_loss
            model_name=lss_str+"_"+str(e)+"_"+str(params["rn_id"])+params["model"]+"_model.ckpt"
            save_path=base_cp_path+model_name
            saved_path=False
            if pre_best_loss>total_loss:
                pre_best_loss=total_loss
                model_name=lss_str+"_"+str(e)+"_"+str(params["rn_id"])+params["model"]+"_best_model.ckpt"
                save_path=base_cp_path+model_name
                saved_path= saver.save(sess, save_path)
            else:
                if e%3.0==0:
                 saved_path= saver.save(sess, save_path)
            if saved_path != False:
                s='MODEL_Saved --> epoch %i | error %f path %s'%(e, total_loss,saved_path)
                ut.log_write(s,params)


train(X_train,Y_train,X_test,Y_test)
