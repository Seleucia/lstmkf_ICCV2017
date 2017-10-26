import os
import utils
import platform
import getpass
import locale
locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

def get_params():
   global params
   params={}
   params['run_mode']=1 #from scratch: 0,from trained model: 1, resume: 2,
   params["rn_id"]="pretrained" #running id, model
   params["load_mode"]=0#0=full training,1=only test set,2=full dataset, 3=hyper param searc
   params["notes"]="we traing LSTM with with [n_input] input" #running id
   params["model"]="inception_resnet_v2"#kccnr,dccnr
   params["optimizer"]="Adam" #1=classic kcnnr, 2=patch, 3=conv, 4 =single channcel
   params['write_est']=False

   params['lr']=0.001
   params["grad_clip"]=100.0
   params['mtype']="seq"
   params['shufle_data']=1
   params['noise_std'] = 0.0000
   params['noise_schedule'] = [100.0, 1000.0, 2000.0, 1000.0]
   params["normalise_data"] = 4

   params['nlayer']= 3 #LSTM
   params['Qnlayer'] = 1  # LSTM
   params['Rnlayer'] = 1  # LSTM
   params['Knlayer'] = 3  # LSTM
   params['Flayer'] = 1  # LSTM
   params['n_output']= 48
   params['n_hidden']= 512
   params['Qn_hidden']= 256
   params['Rn_hidden']= 256
   params['Kn_hidden']= 128
   params['P_mul']= 0.1
   params['K_inp']=48
   params['n_input']= 48#1536#48#2048

   params['KMIX']=10
   params['n_epochs']=5
   params['batch_size']=5
   params['seq_length']= 100
   params['reset_state']= 5#-1=Never, n=every n batch
   params["corruption_level"]=0.5

   params['per_process_gpu_memory_fraction']=1
   params['hidden_size']=5
   params['rnn_keep_prob']=0.9
   params['num_epochs_per_decay']=30

   #Data loading params.
   params["test_lst_act"]=['S9','S11']
   params["train_lst_act"]=['S1','S6','S7','S5','S8']
   params["action"]=''
   params["data_bin"]='/home/coskun/PycharmProjects/poseft/files/temp/full-2048.h5'
   params["reload_data"]=0#0=load from the hdf, 1 =reload form the local

   params['frame_shift']=0
   params['is_forcasting']=0


   # learning parameters
   params['forcast_sequence_count']= 10
   params['seed_length']=50
   params['forcast_length']=100
   params["subsample"]=0
   # params['is_forcasting']=1 Call this with training file
   params['forcast_distance']=1

   #image sizes
   params['height']=299
   params['width']=299
   params["7scene"]=['chess','fire','office','pumpkin','redkitchen','stairs','heads']
   params["cambridge"]=['KingsCollege','OldHospital','ShopFacade','StMarysChurch','Street']



   if(platform.node()=="coskunh"):
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params["data_dir"]='/home/coskun/PycharmProjects/data/pose/mv_val'
       params["data_bin"]='/home/coskun/PycharmProjects/poseft/files/temp/full-11474-11328.h5-51547-50668.h5'
       params['batch_size']=2
       params['max_count']= 5000000000
       params["test_lst_act"]=['S11']
       params["train_lst_act"]=['S9']

   if(platform.node()=="milletari-workstation"):
       params["data_dir_y"]="/mnt/Data1/hc/tt/" #joints with 16, cnn+lstm and autoencder training
       params["data_dir_x"]="/mnt/Data1/hc/est/iv4-1/" #joints with 16, cnn+lstm and autoencder training
       #/mnt/Data1/hc/est/iv4-1/fl_1536
       # params["data_dir"]="/mnt/Data1/hc/est/iv4/" #joints with 16, cnn+lstm and autoencder training
       params["data_dir"]="/mnt/Data1/hc/tt/" #inception training path
       # params["data_bin"]='/mnt/Data1/hc/tt/training_temp/fullivb-'+str(params['n_input'])+'.h5'
       # params["data_bin"]='/mnt/Data1/hc/tt/training_temp/fulliv4_bb-'+str(params['n_input'])+'.h5'
       params["data_bin"]='/mnt/Data1/hc/tt/training_temp/cp-1stepoch-ntu-'+str(params['n_input'])+'.h5'
       # params["data_bin"]='/mnt/Data1/hc/tt/training_temp/full2-'+str(params['n_input'])+'.h5'
       params['batch_size']=20
       params['max_count']=40000000

   if (platform.node() == "hc"):
       if(getpass.getuser()=="coskunh"):
           # params["model_file"] = '/home/coskunh/PycharmProjects/poseft/files/models/inception-resnet-old/'
           # params["model_file"] = '/home/coskunh/PycharmProjects/poseft/files/models/trained/cp/'
           params["model_file"] = '/home/coskunh/PycharmProjects/poseft/files/models/imagenet/inception_resnet_v2_2016_08_30.ckpt'
           params['data_dir'] = '/home/coskunh/PycharmProjects/data/36m'
           params["est_file"] = "/home/coskunh/PycharmProjects/poseft/files/est/"
           params["cp_file"] = "/home/coskunh/PycharmProjects/data/36m/cp_tr/fresh/"
           params["data_bin"]='/home/huseyin/projects/data/full-2048.h5'
           params['batch_size']=1
           params['max_count']=1000000000
           params["test_lst_act"] = ['S9', 'S11']
           params["train_lst_act"] = ['S9', 'S11']


       if(getpass.getuser()=="coskun"):
           params["model_file"] = '/home/coskun/PycharmProjects/poseft/files/models/inception_resnet_v2_2016_08_30.ckpt'
           params['data_dir'] = '/mnt/2tb/datasets/36m/'
           params["est_file"] = "/mnt/2tb/datasets/36m/training/est/"
           params["cp_file"] = "/mnt/2tb/datasets/36m/training/cp/"
           params["data_bin"]='/home/huseyin/projects/data/full-2048.h5'
           params['batch_size']=30
           params['max_count']=1000000000


   if (platform.node() == "hm36"):
       params["model_file"] = '/home/coskunh/PycharmProjects/poseft/files/models/inception_resnet_v2_2016_08_30.ckpt'
       params['data_dir'] = '/mnt/disks/data/datasets/36m'
       params['est_file'] = '/mnt/disks/data/datasets/36m/training/est/'
       params['cp_file'] = '/mnt/disks/data/datasets/36m/training/cp/'
       params["data_bin"]='/home/huseyin/projects/data/full-2048.h5'
       params['batch_size']=30
       params['max_count']=1000000000

   if(platform.node()=="titanx2"):
       params["data_dir"]="/home/users/achilles/human36/joints16/" #joints with 16, cnn+lstm and autoencder training
       params['max_count']=-1

   if(platform.node()=="FedeWSLinux"):
       params["caffe"]="/usr/local/caffe/python"
       params["data_dir"]="/mnt/hc/joints16/"
       params['max_count']=1000

   if(platform.node()=="cmp-comp"):
       params['batch_size']=60
       params["n_procc"]=1
       params["WITH_GPU"]=True
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params["data_dir"]="/mnt/Data1/hc/joints/"
       params["data_dir"]="/home/huseyin/data/joints16/"
       params['n_hidden']= 128
       params['max_count']= 100

   params["data_bin"]=params["data_bin"]

   if params["reload_data"]==1:
       params['data_dir']=params["data_bin"]

      #system settings
   wd=os.path.dirname(os.path.realpath(__file__))
   wd=os.path.dirname(wd)
   params['wd']=wd
   params['log_file']=wd+"/files/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   # params["cp_file"]="/mnt/Data1/hc/tt/cp/lstm/"
   # params["cp_file"]="/mnt/Data1/hc/tt/cp/ires_new/"
   # params["model_file"]=wd+"/files/models"
   # params["est_file"]="/mnt/Data1/hc/est/iv3"
   # params["est_file"]="/mnt/Data1/hc/est/iv4xxx"
   params["sm"]=wd+"/files/sm"



   #params['step_size']=[10]
   params['test_size']=0.20 #Test size
   params['val_size']=0.20 #val size
   params['test_freq']=100 #Test frequency
   params["notes"]=params["notes"].replace('n_input',str(params['n_input']))
   return params

def update_params(params):
   params['log_file']=params['wd']+"/files/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   return params
