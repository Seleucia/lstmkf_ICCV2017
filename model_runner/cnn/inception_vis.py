import math
import numpy as np
import tensorflow as tf
from nets import inception
import urllib2
from datetime import datetime
from helper import config
from helper import utils as ut
from helper import dt_utils as dut
from helper.preprocessing import human36m_preprocessing
from PIL import Image
import numpy as np

params=config.get_params()

slim = tf.contrib.slim

num_examples=100
subset='validation'
is_training=False

def eval(params):
    # batch_size = params['batch_size']
    # num_examples = len(params['test_files'][0])
    with tf.Graph().as_default() as g:
        url = '/home/coskun/PycharmProjects/data/pose/mv_val/img/S9/Discussion 1.54138969/frame_00010.png'
        filename_queue = tf.train.string_input_producer([url]) #  list of files to read

        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)

        # image_raw = tf.image.decode_png(value) # use png or jpg decoder based on your files.
        image_raw = tf.image.decode_png(value, channels=3)

        processed_image =  human36m_preprocessing.preprocess_image(image_raw, 224, 224, is_training=is_training)
        processed_images  = tf.expand_dims(processed_image, 0)
        # image, label = dut.distorted_inputs(params,is_training=is_training)

        with slim.arg_scope(inception.inception_v2_arg_scope()):
            logits, end_points = inception.inception_v2(processed_images, num_classes=params['n_output'], is_training=is_training)

        init_fn=ut.get_init_fn(slim,params,load_previus_cp=True)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = params['per_process_gpu_memory_fraction']
        # operations = g.get_operations()
        # for operation in operations:
        #     print "Operation:",operation.name

        features = g.get_tensor_by_name('InceptionV2/InceptionV2/Mixed_3b/concat:0')
        # features = g.get_tensor_by_name('InceptionV2/InceptionV2/MaxPool_3a_3x3/MaxPool:0')


        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_local_variables())
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            init_fn(sess)
            num_iter = 1
            print('%s: Model reading started.' % (datetime.now()))

            step = 0
            loss_lst=[]
            # while step < num_iter and not coord.should_stop():
            while step < num_iter:
                try:
                    features_values = sess.run(features)
                except tf.errors.OutOfRangeError:
                    print ('Testing finished....%d'%step)
                    break
                print features_values.shape
                img_arr=np.squeeze(features_values[:,:,:,1])
                print img_arr.shape

                img = Image.fromarray(img_arr).convert('RGB')
                img.save('/home/coskun/PycharmProjects/poseft/files/temp/my.png')
                img.show()


                # joint_list=['/'.join(p1.split('/')[0:-1]).replace('joints','img').replace('.cdf','')+'/frame_'+(p1.split('/')[-1].replace('.txt','')).zfill(5)+'.png' for p1 in image_names]
                # print ('List equality check:')
                # print len(label_names) == len(set(label_names))
                # print sum(joint_list==label_names)==(len(est))
                # print(len(label_names))init_fn=ut.get_init_fn(slim,params,load_previus_cp=True)
                step += 1
            coord.request_stop()
            coord.join(threads)

eval(params)