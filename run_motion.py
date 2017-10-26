import tensorflow as tf
from model_runner.klstm.kf_humanmotion import  kftf as kf
from model_runner.klstm.kf_humanmotion_ac import  kftf as kf_ac
from model_runner.klstm.ema import  Ema as ema
import helper.dt_utils as dut
from helper import config
import helper.slam_helper as sh

print "Kalman started..."
params = config.get_params()
params["sequence"]='David'
params["est_file"]='/mnt/Data1/hc/est/vel/'
params["data_mode"]="human" #human
params['avg_mode']='exp'
n=params['n']=3
# params['n_output']= 48
# params['n_input']= 48
# params["reload_data"] = 0
# params["data_bin"] = '/mnt/Data1/hc/tt/training_temp/full2-' + str(params['n_input']) + '.h5'
# params["data_dir"]="/home/coskun/PycharmProjects/PoseNet/Street/"

seq_id_names_test=[]
if params["data_mode"]=="slam":
    params['n_output'] = 7
    params['n_input'] = 7
    params["reload_data"] = 0
    params["data_dir"] = "/home/coskun/PycharmProjects/PoseNet/"+params["sequence"]+"/"
    db_values_x_test, db_values_y_test,seq_id_names_test=sh.load_kalman_data(params)

    # print db_values_x_test.shape
else:
    params['n_output'] = 48
    params['n_input'] = 48
    params["reload_data"] = 0
    params["data_bin"] = '/mnt/Data1/hc/tt/training_temp/full2-' + str(params['n_input']) + '.h5'
    # params["data_bin"] = '/mnt/Data1/hc/tt/training_temp/fullivb-' + str(params['n_input']) + '.h5'
    db_values_x_training, db_values_y_training, db_names_training, db_values_x_test, \
    db_values_y_test, db_names_test = dut.load_from_bin_db(params)
    params['db_names_test']=db_names_test
    # db_values_x_test, db_values_y_test = sh.load_kalman_data(params)

# print seq_id_names_test
# model=kf_ac()
model=kf()
# model=ema()


model.run(params,db_values_x_test,db_values_y_test,seq_id_names_test)
print 'Loadedd.'