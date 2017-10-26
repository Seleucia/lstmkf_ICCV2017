import math
import numpy as np
import tensorflow as tf
from nets import inception_resnet_v2
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
    params['write_est']=False
    num_examples = len(params['test_files'][0])
    with tf.Graph().as_default():
        batch = dut.distorted_inputs(params,is_training=is_training)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits,aux, end_points = inception_resnet_v2.inception_resnet_v2(batch[0],
                                                                         num_classes=params['n_output'],
                                                                         is_training=is_training)

        # Obtain the trainable variables and a saver
        # variables_to_restore = slim.get_variables_to_restore()

        init_fn=ut.get_init_fn(slim,params)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = params['per_process_gpu_memory_fraction']
        with tf.Session() as sess:
            init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            # init_op = tf.global_variables_initializer()
            sess.run(init_op)
            init_fn(sess)
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))


            num_iter = int(math.ceil(num_examples / batch_size))
            print('%s: Testing started.' % (datetime.now()))

            step = 0
            loss_lst=[]
            run_lst=[]
            run_lst.append(logits)
            # run_lst.append(end_points['PreLogitsFlatten'])
            # run_lst.append(end_points['PrePool'])
            [run_lst.append(lst) for lst in batch[1:len(batch)]]

            while step < num_iter and not coord.should_stop():
                try:
                    batch_res= sess.run(run_lst)
                except tf.errors.OutOfRangeError:
                    print ('Testing finished....%d'%step)
                    break
                if(params['write_est']==True):
                    ut.write_mid_est(params,batch_res)
                est=batch_res[0]
                gt=batch_res[1]
                # print(est.shape)
                # print(gt.shape)
                prepool=batch_res[-1]
                loss,_= ut.get_loss(params,gt,est)
                loss_lst.append(loss)
                s ='VAL --> batch %i/%i | error %f'%(step,num_iter,loss)
                if step%10==0:
                    ut.log_write(s,params,screen_print=True)
                    print "Current number of examples / mean err: %i / %f"%(step*gt.shape[0],np.mean(loss_lst))
                else:
                    ut.log_write(s, params, screen_print=False)
                # joint_list=['/'.join(p1.split('/')[0:-1]).replace('joints','img').replace('.cdf','')+'/frame_'+(p1.split('/')[-1].replace('.txt','')).zfill(5)+'.png' for p1 in image_names]
                # print ('List equality check:')
                # print len(label_names) == len(set(label_names))
                # print sum(joint_list==label_names)==(len(est))
                # print(len(label_names))
                step += 1
            coord.request_stop()
            coord.join(threads)
            return np.mean(loss_lst)
