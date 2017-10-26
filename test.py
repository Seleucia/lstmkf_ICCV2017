from helper import config
from helper import dt_utils as dt
from helper import utils as ut
from model_runner.cnn import inception_output

params=config.get_params()
params["model"]="inception_resnet_v2"#kccnr,dccnr
params["notes"]="inception_resnet_v2 testing on training and test dataset..." #running id
params['write_est']=False
params["data_dir"]="/mnt/Data1/hc/tt/" #inception training path
params["est_file"]="/mnt/Data1/hc/est/46882/"
params["model_file"]='/home/coskun/PycharmProjects/poseftv4/files/cp/'
# params["cp_file"]='/home/coskun/PycharmProjects/poseftv4/files/cp-1stepoch-ntu/'
params["cp_file"]='/mnt/Data1/hc/tt/cp/ires/cp_sel/'
#
# params["cp_file"]='/mnt/Data1/hc/tt/cp/ires/cp_sel/'
params['scope']='InceptionResnetV2'
# params['model_file']=params["model_file"]+'/'+"inception_resnet_v2_2016_08_30.ckpt"
params['checkpoint_exclude_scopes']=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
params['run_mode']=1
# params["test_lst_act"]=['S11','S9']

#params["test_lst_act"]=['S9','S5','S6','S7','S8','S9','S11']
#params["test_lst_act"]=['S9','S5','S6','S7','S8','S9','S11']

# params["test_lst_act"]=['S1','S6','S7']
# params["test_lst_act"]=['S9','S11','S5','S8']

params["test_lst_act"]=['S9','S11']
params["train_lst_act"]=['S1','S6','S7','S5','S8']

#params["train_lst_act"]=['S1','S6','S7']
# params["train_lst_act"]=['S1']
params['training_size']=0
params['test_size']=0
params['batch_size']=100
params['run_mode']=2 #Load previuesly trained model
is_training=False
# params['training_files']=dt.load_files(params,is_training=True)
# params['training_files']=([],[])
#Action List=
ut.start_log(params)
lst_bool=[True,False]
lst_bool=[False,True]
lst_action=['Walking','Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing',
            'Purchases','Sitting','SittingDown','Smoking','Waiting','WalkDog','WalkTogether']

#Test walking regenerate....
# lst_action=['Smoking','Waiting','Walking','WalkDog','WalkTogether']
for tr in lst_bool:
    loss=0.
    total_cnt=0.
    for action in lst_action:
        params["action"]=action
        params['test_files']=dt.load_files(params,is_training=tr)
        test_loss= inception_output.eval(params)
        cnt=len(params['test_files'][0])
        loss=loss+test_loss*cnt
        total_cnt=total_cnt+cnt
        if tr==False:
            s ='TEST Set --> Action: %s, Frame Count: %i Final error %f'%(action,cnt,test_loss)
        else:
            s ='Train Set --> Action: %s, Frame Count: %i Final error %f'%(action,cnt,test_loss)
        ut.log_write(s,params)

    loss=loss/total_cnt
    if tr==False:
        s ='Total Test Frame Count: %i Final error %f'%(total_cnt,loss)
    else:
        s ='Total Training Frame Count: %i Final error %f'%(total_cnt,loss)
    ut.log_write(s,params)


# ut.start_log(params)
# ut.log_write("Model testing started",params)
# test_loss=inception_eval.eval(params)
# s ='VAL --> Final error %f'%(test_loss)
# ut.log_write(s,params)
