import math

import tensorflow as tf

from helper import dt_utils as dut
from helper import utils as ut
from model_runner.cnn import learn
from nets import inception_resnet_v2

slim = tf.contrib.slim
# This might take a few minutes.

is_training=True
# mfile=ut.get_last_modelname(params)
_USE_DEFAULT = 0

def run_steps(params,epoch_counter):
    with tf.Graph().as_default():
        num_examples=len(params['training_files'][0])
        number_of_steps = int(math.ceil(num_examples / params['batch_size']))-1
        print "Number of steps: %i" % number_of_steps
        number_of_steps=number_of_steps*(epoch_counter+1)

        tf.logging.set_verbosity(tf.logging.INFO)
        batch = dut.distorted_inputs(params,is_training=is_training)

        # with tf.Session() as sess:
        #     tf.initialize_all_variables().run()
        #     print('Variables init.......')
        #     retur=sess.run(batch)
        #     print(retur)
        #
        # Create the model:
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits,aux, endpoint = inception_resnet_v2.inception_resnet_v2(batch[0], num_classes=params['n_output'], is_training=is_training)

        err=tf.subtract(logits, batch[1])

        losses = tf.reduce_sum(tf.reduce_sum(tf.square(err)))
        err_tr = tf.reshape(err, shape=[-1, 3])
        err_tr = tf.square(err_tr)
        err_tr = tf.reduce_sum(err_tr,axis=1)
        err_tr = tf.sqrt(err_tr)
        err_tr = tf.reduce_mean(err_tr)


        # err_aux=tf.subtract(aux, batch[1])
        # losses_aux = tf.reduce_sum(tf.reduce_sum(tf.square(err_aux)))
        reg_loss=slim.losses.get_total_loss()
        # total_loss = losses+0.4*losses_aux+reg_loss
        total_loss = losses+reg_loss
        #Compute cross entropy

        # with tf.Session() as sess:
        #     tf.initialize_all_variables().run()
        #     print('Variables init.......')
        #     r1,r2=sess.run([losses,reg_loss])
        #     print(r1)
        #


        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.scalar('losses/losses', losses)
        tf.summary.scalar('losses/training_loss', err_tr)
        tf.summary.scalar('losses/reg_loss', reg_loss)
        summary_writer = tf.summary.FileWriter(params["sm"])
        #
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
        #
        # optimizer.compute_gradients()
        train_op = slim.learning.create_train_op(total_loss, optimizer,summarize_gradients=False)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
           # Run the training:
        # final_loss = learn.train(
        #     loss=losses,
        #     logits=logits,
        #     batch=batch,
        #     endpoint=endpoint,
        #     train_op=train_op,
        #     logdir=params["cp_file"],
        #     init_fn=ut.get_init_fn(slim,params),
        #     number_of_steps=number_of_steps,
        #     summary_writer=summary_writer,
        #     session_config=config,
        # )
        final_loss = tf.contrib.slim.learning.train(
            train_op=train_op,
            logdir=params["cp_file"],
            init_fn=ut.get_init_fn(slim,params),
            number_of_steps=number_of_steps,
            summary_writer=summary_writer,
            session_config=config,
        )
        # final_loss=0
    return final_loss