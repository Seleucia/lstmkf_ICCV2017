import math
import numpy as np
import tensorflow as tf
from nets import vgg
import time
from datetime import datetime
from helper import config
from helper import utils as ut
from helper import dt_utils as dut

slim = tf.contrib.slim
num_examples=100
subset='validation'
is_training=False
def eval(params):
    batch_size = params['batch_size']
    num_examples = len(params['test_files'][0])
    with tf.Graph().as_default():
        batch = dut.distorted_inputs(params,is_training=is_training)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_19(batch[0], num_classes=params['n_output'], is_training=is_training)

        init_fn=ut.get_init_fn(slim,params)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = params['per_process_gpu_memory_fraction']

        with tf.Session(config=config) as sess:
            # sess.run(tf.initialize_all_variables())
            sess.run(tf.initialize_local_variables())
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            init_fn(sess)
            num_iter = int(math.ceil(num_examples / batch_size))
            print('%s: Testing started.' % (datetime.now()))

            step = 0
            loss_lst=[]
            run_lst=[]
            run_lst.append(logits)
            [run_lst.append(lst) for lst in batch[1:len(batch)]]

            while step < num_iter and not coord.should_stop():
                try:
                    batch_res= sess.run(run_lst)
                except tf.errors.OutOfRangeError:
                    print ('Testing finished....%d'%step)
                    break
                if(params['write_est']==True):
                    ut.write_est(params,batch_res)
                est=batch_res[0]
                gt=batch_res[1]
                loss= ut.get_loss(params,gt,est)
                loss_lst.append(loss)
                s ='VAL --> batch %i/%i | error %f'%(step,num_iter,loss)
                ut.log_write(s,params)
                # joint_list=['/'.join(p1.split('/')[0:-1]).replace('joints','img').replace('.cdf','')+'/frame_'+(p1.split('/')[-1].replace('.txt','')).zfill(5)+'.png' for p1 in image_names]
                # print ('List equality check:')
                # print len(label_names) == len(set(label_names))
                # print sum(joint_list==label_names)==(len(est))
                # print(len(label_names))
                step += 1
            coord.request_stop()
            coord.join(threads)
            return np.mean(loss_lst)
