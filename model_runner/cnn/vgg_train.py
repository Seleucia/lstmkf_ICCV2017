import math

import tensorflow as tf

from helper import dt_utils as dut
from helper import utils as ut
from model_runner.cnn import learn
from nets import vgg

slim = tf.contrib.slim
# This might take a few minutes.

is_training=True
# mfile=ut.get_last_modelname(params)
_USE_DEFAULT = 0

def run_steps(params,epoch_counter):
    with tf.Graph().as_default():
        num_examples=len(params['training_files'][0])
        number_of_steps = int(math.ceil(num_examples / params['batch_size']))-1
        number_of_steps=number_of_steps*(epoch_counter+1)
        tf.logging.set_verbosity(tf.logging.INFO)
        batch = dut.distorted_inputs(params,is_training=is_training)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, endpoint = vgg.vgg_19(batch[0], num_classes=params['n_output'], is_training=is_training)

        # # Create the model:
        # with slim.arg_scope(inception.inception_v2_arg_scope()):
        #     logits, _ = inception.inception_v2(batch[0], num_classes=params['n_output'], is_training=is_training)

        err=tf.sub(logits, batch[1])
        losses = tf.reduce_mean(tf.reduce_sum(tf.square(err),1))
        reg_loss=slim.losses.get_total_loss()
        total_loss = losses+reg_loss
        tf.scalar_summary('losses/total_loss', total_loss)
        tf.scalar_summary('losses/losses', losses)
        tf.scalar_summary('losses/reg_loss', reg_loss)
        summary_writer = tf.train.SummaryWriter(params["sm"])

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

        train_op = slim.learning.create_train_op(total_loss, optimizer,summarize_gradients=True)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = params['per_process_gpu_memory_fraction']
           # Run the training:
        final_loss = learn.train(
            loss=losses,
            logits=logits,
            batch=batch,
            endpoint=endpoint,
            train_op=train_op,
            logdir=params["cp_file"],
            init_fn=ut.get_init_fn(slim,params),
            number_of_steps=number_of_steps,
            summary_writer=summary_writer,
            session_config=config,
        )
    return final_loss