import os
import numpy as np
import tensorflow as tf
from nets import inception
from helper import utils as ut
from helper.preprocessing import human36m_preprocessing
import urllib2

from helper import config
params=config.get_params()

slim = tf.contrib.slim
num_examples=100
subset='validation'
is_training=False
params['run_mode']=2
params['scope']='InceptionV3'
params['model_file']=params["model_file"]+'/'+"inception_v3.ckpt"
params['checkpoint_exclude_scopes']=["InceptionV3/Logits", "InceptionV3/AuxLogits"]
params["cp_file"]='/home/coskun/PycharmProjects/poseftv3/files/cp'

slim = tf.contrib.slim

batch_size = 3

with tf.Graph().as_default():
    url = '/home/coskun/Downloads/11.png'
    filename_queue = tf.train.string_input_producer([url]) #  list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    # image_raw = tf.image.decode_png(value) # use png or jpg decoder based on your files.
    if('.jpg' in os.path.basename(url)):
        image_raw = tf.image.decode_jpeg(value, channels=3)
    else:
        image_raw = tf.image.decode_png(value, channels=3)

    # image = tf.image.resize_bilinear(image_raw, [256, 256,3],    align_corners=False)
    processed_image =  human36m_preprocessing.preprocess_image(image_raw, 299, 299, is_training=is_training)
    processed_images  = tf.expand_dims(processed_image, 0)


    # image, label = dut.distorted_inputs(params,is_training=is_training)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits,aux, end_points = inception.inception_v3(processed_images, num_classes=params['n_output'], is_training=is_training)

    init_fn=ut.get_init_fn(slim,params)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = params['per_process_gpu_memory_fraction']
    loss_lst=[]
    run_lst=[]
    run_lst.append(logits)

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        init_fn(sess)

        batch_res= sess.run(run_lst)
        e=batch_res[0][0].tolist()
        vec_str = ' '.join(['%.6f' % num for num in e])
        est_file="/home/coskun/PycharmProjects/poseftv3/files/temp"
        p_file=est_file+'/'+os.path.basename(url).replace('.png','.txt').replace('.jpg','.txt')
        if os.path.exists(p_file):
            os.remove(p_file)
        with open(p_file, "a") as p:
            p.write(vec_str)
            p.close()
        coord.request_stop()
        coord.join(threads)




