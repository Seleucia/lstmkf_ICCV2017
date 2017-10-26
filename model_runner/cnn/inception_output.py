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
num_examples=400
subset='validation'
# params=config.get_params()
#
# params['write_est']=False

def eval(params):
    batch_size = params['batch_size']
    num_examples = len(params['test_files'][0])
    with tf.Graph().as_default():
        batch = dut.distorted_inputs(params,is_training=False)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits,aux, end_points = inception_resnet_v2.inception_resnet_v2(batch[0],
                                                                         num_classes=params['n_output'],
                                                                         is_training=False)
        # with slim.arg_scope(inception.inception_v3_arg_scope()):
        #     logits, end_points = inception.inception_v3(batch[0], num_classes=params['n_output'], is_training=False)

        init_fn=ut.get_init_fn(slim,params)
        # config_prot = tf.ConfigProto()
        # config_prot.gpu_options.per_process_gpu_memory_fraction = 0.45
        config_prot = tf.ConfigProto(device_count = {'GPU': 0})
        with tf.Session(config=config_prot) as sess:
            # sess.run(tf.initialize_all_variables())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            init_fn(sess)
            num_iter = int(math.ceil(num_examples / batch_size))+1
            print('%s: Testing started.' % (datetime.now()))

            step = 0
            loss_lst=[]
            run_lst=[]
            run_lst.append(logits)
            run_lst.append(end_points['PreLogitsFlatten'])
            run_lst.append(end_points['Mixed_6a_repeat'])
            [run_lst.append(lst) for lst in batch]

            while step < num_iter and not coord.should_stop():
                try:
                    # start_time=time.time()
                    batch_res= sess.run(run_lst)
                    # print(time.time()-start_time)
                except tf.errors.OutOfRangeError:
                    print ('Testing finished....%d'%step)
                    break
                est=batch_res[0]
                gt=batch_res[4]
                # mid_layer=batch_res[0]
                # file_names=batch_res[-1]
                if(params['write_est']==True):
                    ut.write_mid_est(params,batch_res)
                    # ut.write_est(est_file,est,file_names)

                diff=gt-est
                err=np.square(diff)
                err=np.sum(err)

                loss,_= ut.get_loss(params,gt,est)
                loss_lst.append(loss)
                s ='VAL --> batch %i/%i | error %f, %f'%(step,num_iter,loss,err)
                print(s)
                # ut.log_write(s,params)
                # joint_list=['/'.join(p1.split('/')[0:-1]).replace('joints','img').replace('.cdf','')+'/frame_'+(p1.split('/')[-1].replace('.txt','')).zfill(5)+'.png' for p1 in image_names]
                # print ('List equality check:')
                # print len(label_names) == len(set(label_names))
                # print sum(joint_list==label_names)==(len(est))
                # print(len(label_names))
                step += 1
            coord.request_stop()
            coord.join(threads)
            return np.mean(loss_lst)
