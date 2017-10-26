from helper import utils as ut
import argparse
from helper import dt_utils as dt
from model_runner import model_provider
from helper import config

params=config.get_params()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode',type=int,default=1) #1= train, test,2, test, train 3= only test given model.
parser.add_argument('--run_mode',type=int,default=2) #
parser.add_argument('--model_file',default="/home/coskunh/PycharmProjects/data/36m/cp_tr/model.ckpt-10199")
parser.add_argument('--epoch_counter_start',type=int,default=0) #1= train, test,2, test, train 3= only test given model.

args = parser.parse_args()

params['write_est']=True
params["ds_training"]="crop350"
params["ds_test"]="crop350"
(trainer,evaller,params)=model_provider.get_model(params)
#Number of steps per epoch
params['training_files']=dt.load_files(params,is_training=True)
# params['training_files']=([],[])
params['test_files']=dt.load_files(params,is_training=False)


# params['run_mode']=0 #Load previuesly trained model
if args.mode==3:
    assert params['model_file']!=""
    params['model_file']=args.model_file
    params['run_mode'] = 3
    ut.start_log(params)
    ut.log_write("Testing given model", params)
    params["batch_size"] = 100
    params["est_file"] = params["est_file"] + str(args.model_file.split('/')[-1]) + '/'
    test_loss = evaller.eval(params)
    s = 'VAL --> Model %s | error %f' % (args.model_file.split('/')[-1], test_loss)
    ut.log_write(s, params)

elif args.mode==1:
    ut.start_log(params)
    for epoch_counter in range(args.epoch_counter_start,100):
        params["sm"]=params["sm"]+'/'+ut.get_time()
        # if epoch_counter> 0:.
        params['run_mode'] = 2
        params["batch_size"] = 30
        training_loss=trainer.run_steps(params,epoch_counter)
        s='TRAIN --> epoch %i | error %f'%(epoch_counter, training_loss)
        ut.log_write(s,params)
        # params['run_mode'] = 3 # 3, loading given model
        params['run_mode'] = 2
        params['lr'] = params['lr'] / (5**epoch_counter)
        params["batch_size"]=100
        params["est_file"] = params["est_file"]+str(epoch_counter)+'/'
        test_loss = evaller.eval(params)
        s = 'VAL --> epoch %i | error %f' % (epoch_counter, test_loss)
        ut.log_write(s, params)
