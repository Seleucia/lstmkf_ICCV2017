import time

import numpy as np
import tensorflow as tf

from helper import config
from helper import dt_utils as dut
from helper import utils as ut

# from model_runner.rnn_lstm import  Model
from model_runner.lstm.rnn_lstm_2layer import  Model

params=config.get_params()
params["model"]="lstmv2_2layer"
params=config.update_params(params)
(F_names_training,S_Train_list,F_names_test,S_Test_list)=dut.prepare_training_set_fnames(params)
index_train_list,S_Train_list=dut.get_seq_indexes(params,S_Train_list)
index_test_list,S_Test_list=dut.get_seq_indexes(params,S_Test_list)

batch_size=params['batch_size']
n_train_batches = len(index_train_list)
n_train_batches /= batch_size

n_test_batches = len(index_test_list)
n_test_batches /= batch_size

params['training_size']=len(F_names_training)*params['seq_length']
params['test_size']=len(F_names_test)*params['seq_length']
ut.start_log(params)
ut.log_write("Model training started",params)
# summary_writer = tf.train.SummaryWriter(params["sm"])

def train():
    model = Model(params)
    num_epochs=1000
    decay_rate=0.4
    show_every=100

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = params['per_process_gpu_memory_fraction']
    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        merged = tf.summary.merge_all()
        summary_writer = tf.train.SummaryWriter(params["sm"], sess.graph)

        for e in xrange(num_epochs):
            sess.run(tf.assign(model.lr, params['lr'] * (decay_rate ** e)))
            LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden state
            LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden sta
            state_reset_counter_lst=[0 for i in range(batch_size)]
            total_train_loss=0
            for minibatch_index in xrange(n_train_batches):
                start = time.time()
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                (LStateList_b,x,y,state_reset_counter_lst)=dut.prepare_lstm_batch_joints(index_train_list, minibatch_index, batch_size, S_Train_list,LStateList_t,LStateList_pre, params, F_names_training,state_reset_counter_lst)
                LStateList_pre=LStateList_b

                y=y.reshape(batch_size*params["seq_length"],params["n_output"])
                feed = {model.input_data: x, model.input_zero:np.ceil(x), model.target_data: y, model.initial_state: LStateList_b,model.is_training:True,model.output_keep_prob:0.5}
                summary,train_loss, LStateList_t,_ =\
                    sess.run([merged,model.cost, model.final_state, model.train_op], feed)
                summary_writer.add_summary(summary, minibatch_index)
                tmp_lst=[]
                for item in  LStateList_t:
                    tmp_lst.append(item.c)
                    tmp_lst.append(item.h)
                LStateList_t=tmp_lst
                total_train_loss+=train_loss
                if (minibatch_index%show_every==0):
                    print "Training batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,train_loss)

            total_train_loss=total_train_loss/n_train_batches
            s='TRAIN --> epoch %i | error %f'%(e, total_train_loss)
            ut.log_write(s,params)

            LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden state
            LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden sta
            state_reset_counter_lst=[0 for i in range(batch_size)]
            total_test_loss=0
            for minibatch_index in xrange(n_test_batches):
                state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
                (LStateList_b,x,y,state_reset_counter_lst)=dut.prepare_lstm_batch(index_test_list, minibatch_index, batch_size, S_Test_list,LStateList_t,LStateList_pre, params, Y_test, X_test,state_reset_counter_lst)
                LStateList_pre=LStateList_b
                y=y.reshape(batch_size*params["seq_length"],params["n_output"])
                feed = {model.input_data: x, model.target_data: y, model.initial_state: LStateList_b,model.is_training:False,model.output_keep_prob:1.0}
                LStateList_t,final_output = sess.run([model.final_state,model.final_output], feed)
                test_loss=ut.get_loss(params,gt=y,est=final_output)
                tmp_lst=[]
                for item in  LStateList_t:
                    tmp_lst.append(item.c)
                    tmp_lst.append(item.h)
                LStateList_t=tmp_lst
                total_test_loss+=test_loss
                if (minibatch_index%show_every==0):
                    print "Test batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_test_batches,test_loss)
            total_test_loss=total_test_loss/n_test_batches
            print "Total test loss %f"%total_test_loss
            s ='VAL --> epoch %i | error %f'%(e,total_test_loss)
            ut.log_write(s,params)


train()
