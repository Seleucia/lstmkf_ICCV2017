import tensorflow as tf
from model_runner.klstm.kf_humanmotion import  kftf
import helper.dt_utils as dut
from helper import config
import helper.slam_helper as sh

print "Kalman started..."
params = config.get_params()
params["data_mode"]="slam" #human
params['n_output']= 7
params['n_input']= 7
params["reload_data"] = 0
params["data_dir"]="/home/coskun/PycharmProjects/PoseNet/Street/"

if params["data_mode"]=="human":
    _, _, _, db_values_x_test, db_values_y_test, _ = \
        dut.load_from_bin_db(params)
else:
    (db_values_x_test,db_values_y_test,_,_)=sh.load_dataset(params,is_training=False)

# with tf.Graph().as_default():
#     kfm=kftf()
# kfm.run(params,db_values_x_test,db_values_y_test)
print 'Loadedd.'