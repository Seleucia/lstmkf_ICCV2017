import time

import numpy as np
import tensorflow as tf

from helper import config
from helper import dt_char_utils as dut
from helper import utils as ut
from model_runner.lstm.char_lstm import  Model

params=config.get_params()
params["data_bin"]="/mnt/Data1/hc/tt/training_temp/"
params["reload_data"]=0#0=load from the bin, 1 =reload form the local
params['max_count']=5000000000
params['lr']=0.001
params["model"]="char-lstm"
params=config.update_params(params)

X_train,Y_train,char_dict=dut.prepare_training_set(params)
params['n_output']=len(char_dict)
params['n_input']=len(char_dict)

#load model.....
model = Model(params)
params['n_param']=model.total_parameters
print('Model loaded....%i'%params['n_param'])




batch_size=params['batch_size']
n_train_batches = len(X_train)
n_train_batches /= batch_size


params['training_size']=len(X_train)*params['seq_length']
params['test_size']=0
ut.start_log(params)
ut.log_write("Model training started",params)
# summary_writer = tf.train.SummaryWriter(params["sm"])

def train():
    num_epochs=1000
    decay_rate=0.4
    show_every=100

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.40
    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(params["sm"], sess.graph)

        for e in xrange(num_epochs):
            sess.run(tf.assign(model.lr, params['lr'] * (decay_rate ** e)))
            LStateLis=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(params['nlayer']*2)] # initial hidden state
            total_train_loss=0
            for minibatch_index in xrange(n_train_batches):
                start = time.time()
                x=X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                y=Y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

                feed = {model.input_data: x, model.target_data: y, model.initial_state: LStateLis,model.is_training:True,model.output_keep_prob:0.5}
                summary,train_loss, LStateLis,_ =\
                    sess.run([merged,model.cost, model.final_state, model.train_op], feed)
                summary_writer.add_summary(summary, minibatch_index)
                total_train_loss+=train_loss
                if (minibatch_index%show_every==0):
                    print "Training batch loss: (%i / %i / %i)  %f"%(e,minibatch_index,n_train_batches,train_loss)

            total_train_loss=total_train_loss/n_train_batches
            s='TRAIN --> epoch %i | error %f'%(e, total_train_loss)
            ut.log_write(s,params)
            save_path = saver.save(sess, params["cp_file"]+ "/char/model"+str(e)+'_'+str(minibatch_index)+".ckpt")


train()
