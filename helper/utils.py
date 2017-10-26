import numpy as np
import h5py
import datetime
from random import randint
import os
import time
import tensorflow as tf
import locale
locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

def get_loss(params,gt,est,r=None):
    gt= np.asarray(gt)
    est=est
    batch_size=gt.shape[0]
    loss=[]
    if(len(gt.shape)==2):
        for b in range(batch_size):
            diff_vec=np.abs(gt[b].reshape(params['n_output']/3,3) - est[b].reshape(params['n_output']/3,3)) #13*3
            diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss.append(np.mean(sq_m))
        loss=np.nanmean(loss),batch_size
    else:
        g=gt[np.nonzero(r)]
        e=est[np.nonzero(r)]
        n=e.shape[0]
        g=g.reshape(n,params['n_output']/3,3)
        e=e.reshape(n,params['n_output']/3,3)
        diff_vec=np.abs(g - e) #13*3
        # diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
        sq_m=np.sqrt(np.sum(diff_vec**2,axis=2))
        return np.mean(sq_m),n

        # seq_length=gt.shape[1]
        # for b in range(batch_size):
        #     for s in range(seq_length):
        #         g=gt[b][s].reshape(params['n_output']/3,3)
        #         e=est[b][s].reshape(params['n_output']/3,3)
        #         diff_vec=np.abs(g - e) #13*3
        #         diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
        #         sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
        #         loss.append(np.mean(sq_m))
        # loss=np.nanmean(loss)

    return loss


def start_log(params):
    log_file=params["log_file"]
    create_file(log_file)
    ds= get_time()
    log_write("Run Id: %s"%(params['rn_id']),params)
    log_write("Deployment Notes: %s"%(params['notes']),params)
    log_write("Running Mode: %s"%(params['run_mode']),params)
    log_write("Running Model: %s"%(params['model']),params)
    log_write("Batch Size: %s"%(params['batch_size']),params)
    log_write("Load Mode: %s"%(params['load_mode']),params)
    log_write("Sequence size: %s"%(params['seq_length']),params)
    log_write("Learnig rate: %s"%(params['lr']),params)
    log_write("Data Dir: %s"%(params['data_dir']),params)
    log_write("Data Bin: %s"%(params['data_bin']),params)
    log_write("Train list actors: %s"%( ', '.join(params["train_lst_act"])),params)
    log_write("Test list actors: %s"%(', '.join(params['test_lst_act'])),params)
    log_write("Spesific action: %s"%(params['action']),params)
    if("n_param" in params.keys()):
        log_write("Number of parameters: %s"%(params['n_param']),params)
    log_write("Number of hidden units for RNN: %s"%(params['n_hidden']),params)
    log_write("Number of layer for RNN: %s"%(params['nlayer']),params)

    if ("Qn_hidden" in params.keys()):
        log_write("Number of hidden units for Qn_hidden: %s" % (params['Qn_hidden']), params)
    if ("Qnlayer" in params.keys()):
        log_write("Number of layer for Qnlayer: %s" % (params['Qnlayer']), params)
    if ("Rn_hidden" in params.keys()):
        log_write("Number of hidden units for Rn_hidden: %s" % (params['Rn_hidden']), params)
    if ("Rnlayer" in params.keys()):
        log_write("Number of layer for Rnlayer: %s" % (params['Rnlayer']), params)
    if ("Kn_hidden" in params.keys()):
        log_write("Number of hidden units for Kn_hidden: %s" % (params['Kn_hidden']), params)
    if ("Knlayer" in params.keys()):
        log_write("Number of layer for Knlayer: %s" % (params['Knlayer']), params)


    log_write("Number of output (1Xn) : %s"%(params['n_output']),params)
    log_write("CNN image height, width : %s,%s"%(params['height'],params['width']),params)

    if "noise_std" in  params:
        log_write("Training noise:%f"%(params["noise_std"]),params)
    if "normalise_data" in  params:
        log_write("Data Normalsation Mode:%f"%(params["normalise_data"]),params)
    if "frame_shift" in  params:
        log_write("Frame Shift:%f"%(params["frame_shift"]),params)
    if "is_forcasting" in  params:
        log_write("Is Forcas:%f"%(params["is_forcasting"]),params)
    if "rnn_keep_prob" in  params:
        log_write("RNN Dropout %f"%(params["rnn_keep_prob"]),params)
    log_write("RNN Reset State %f"%(params["reset_state"]),params)
    log_write("Shuffle dataset: %f"%(params["shufle_data"]),params)

    if("training_files" in params.keys()):
        log_write("size of training data:%f"%(len(params['training_files'][0])),params)
    # else:
    #     log_write("size of training data:%f"%(params['training_size']),params)

    if("test_files" in params.keys()):
        log_write("size of test data:%f"%(len(params['test_files'][0])),params)
    else:
        log_write("size of test data:%f"%(params['test_size']),params)


    if("training_files" in params.keys()):
        rnd_train=randint(0,len(params['training_files'][0]))
        log_write("Example training file: %s"%(params['training_files'][0][rnd_train]),params)
        log_write("Example training file: %s"%(params['training_files'][1][rnd_train]),params)
    if("test_files" in params.keys()):
        rnd_train=randint(0,len(params['test_files'][0]))
        log_write("Example test file: %s"%(params['test_files'][0][rnd_train]),params)
        log_write("Example test file: %s"%(params['test_files'][1][rnd_train]),params)
    log_write("Starting Time:%s"%(ds),params)

def prep_pred_file(params):
    f_dir=params["est_file"];
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    f_dir=params["est_file"]+'/'+params["model"];
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    map( os.unlink, (os.path.join( f_dir,f) for f in os.listdir(f_dir)) )


def write_forcast(params,mpath,gt,forcast,file_names):
    gt=np.squeeze(gt)
    forcast=np.squeeze(forcast)
    file_names=np.squeeze(file_names)
    est_file=params["est_file"]
    mpath=est_file+mpath
    if not os.path.exists(mpath):
        os.makedirs(mpath)
    est_file=mpath+'/'
    fl_forcast="forcast"
    fl_gt="gt"
    # start_time=time.time()
    for dr,dt in zip([fl_forcast,fl_gt],[gt,forcast]):
        for i in range(file_names.shape[0]):
            e=dt[i]
            fl=file_names[i]
            vec_str = ' '.join(['%.6f' % num for num in e])
            actor= est_file+dr+'/'+fl.split('/')[-3]
            action= est_file+dr+'/'+fl.split('/')[-3]+'/'+fl.split('/')[-2]
            if not os.path.exists(actor):
                os.makedirs(actor)
            if not os.path.exists(action):
                os.makedirs(action)
            p_file=action+'/'+os.path.basename(fl)

            if os.path.exists(p_file):
                os.remove(p_file)
            with open(p_file, "a") as p:
                p.write(vec_str)
                p.close()
    # print(time.time()-start_time)

def write_mid_est(params,batch_res):
    est=batch_res[0]
    mid=batch_res[1]
    file_names=batch_res[-1]
    est=np.squeeze(est)
    mid=np.squeeze(mid)
    est_file=params["est_file"]
    fl_2048="fl_2048"
    fl_48="fl_48"
    mid_file="est"
    # start_time=time.time()
    i=0
    for dt in est:
        fl=file_names[i]
        vec_str = ' '.join(['%.6f' % num for num in dt])
        actor= est_file+fl_48+'/'+fl.split('/')[-3]
        action= est_file+fl_48+'/'+fl.split('/')[-3]+'/'+fl.split('/')[-2]
        if not os.path.exists(actor):
            os.makedirs(actor)
        if not os.path.exists(action):
            os.makedirs(action)
        p_file=action+'/'+os.path.basename(fl)

        if os.path.exists(p_file):
            os.remove(p_file)
        with open(p_file, "a") as p:
            p.write(vec_str)
            p.close()
        i+=1
    # print(time.time()-start_time)

def write_mid_est_np(params,batch_res):
    prepool=batch_res[0]
    file_names=batch_res[-1]
    # est=np.squeeze(est)
    prepool=np.squeeze(prepool)
    print(prepool.shape)
    est_file=params["est_file"]
    fl_prepool="fl_48"
    # start_time=time.time()
    for i in range(len(file_names)):
        fl=file_names[i]
        vec_str = prepool[i]
        actor= est_file+fl_prepool+'/'+fl.split('/')[-3]
        action= est_file+fl_prepool+'/'+fl.split('/')[-3]+'/'+fl.split('/')[-2]
        if not os.path.exists(actor):
            os.makedirs(actor)
        if not os.path.exists(action):
            os.makedirs(action)
        p_file=action+'/'+os.path.basename(fl).replace('.txt','')

        if os.path.exists(p_file):
            os.remove(p_file)
        np.save(p_file,vec_str)
        # with open(p_file, "a") as p:
        #     p.write(vec_str)
        #     p.close()
# print(time.time()-start_time)


def write_rnn_est(est_file,est,file_names):
    # est=np.squeeze(est)
    bs=est.shape[0]
    sq=est.shape[1]
    for i in range(bs):
        for j in range(sq):
            e=est[i][j].tolist()
            fl=file_names[i][j][1]
            vec_str = ' '.join(['%.6f' % num for num in e])
            actor= est_file+'/'+fl.split('/')[-3]
            action= est_file+'/'+fl.split('/')[-3]+'/'+fl.split('/')[-2]
            if not os.path.exists(actor):
                os.makedirs(actor)
            if not os.path.exists(action):
                os.makedirs(action)
            p_file=action+'/'+os.path.basename(fl)
            if os.path.exists(p_file):
                os.remove(p_file)
            with open(p_file, "a") as p:
                p.write(vec_str)



def write_slam_est(est_file,est,file_names):
    est=np.squeeze(est)
    # start_time=time.time()
    for i in range(len(file_names)):
        e=est[i]
        fl=file_names[i][0]
        vec_str = ' '.join(['%.6f' % num for num in e])
        action= est_file+'/'+'/'+fl.split('/')[-2]
        if not os.path.exists(action):
            os.makedirs(action)
        p_file=action+'/'+os.path.basename(fl)

        if os.path.exists(p_file):
            os.remove(p_file)
        with open(p_file, "a") as p:
            p.write(vec_str)
            p.close()

def write_est(est_file,est,file_names):
    est=np.squeeze(est)
    # start_time=time.time()
    for i in range(len(file_names)):
        e=est[i]
        fl=file_names[i][0]
        vec_str = ' '.join(['%.6f' % num for num in e])
        actor= est_file+'/'+fl.split('/')[-3]
        action= est_file+'/'+fl.split('/')[-3]+'/'+fl.split('/')[-2]
        if not os.path.exists(actor):
            os.makedirs(actor)
        if not os.path.exists(action):
            os.makedirs(action)
        p_file=action+'/'+os.path.basename(fl)

        if os.path.exists(p_file):
            os.remove(p_file)
        with open(p_file, "a") as p:
            p.write(vec_str)
            p.close()
    # print(time.time()-start_time)

def single_write(est_file,e,fl):
    vec_str = ' '.join(['%.6f' % num for num in e])
    actor= est_file+'/'+fl.split('/')[-3]
    action= est_file+'/'+fl.split('/')[-3]+'/'+fl.split('/')[-2]
    if not os.path.exists(actor):
        os.makedirs(actor)
    if not os.path.exists(action):
        os.makedirs(action)
    p_file=action+'/'+os.path.basename(fl)

    if os.path.exists(p_file):
        os.remove(p_file)
    with open(p_file, "a") as p:
        p.write(vec_str)
        p.close()

def get_char(char_dict,id):
    for key, value in char_dict.iteritems():
            if value == id:
                return key

def get_time():
    return str(datetime.datetime.now()).replace(":","-").replace(".","-").replace(" ","-")

def get_zero_state(params,t='L'):
    LStateList=[]
    if t == 'L':
        n_hidden = params['n_hidden']
        nlayer=params['nlayer']
    if t == 'Q':
        n_hidden = params['Qn_hidden']
        nlayer = params['Qnlayer']
    if t == 'R':
        n_hidden = params['Rn_hidden']
        nlayer = params['Rnlayer']
    if t == 'K':
        n_hidden = params['Kn_hidden']
        nlayer = params['Knlayer']
    for i in range(nlayer):
        s = []
        s.append(np.zeros(shape=(params["batch_size"],n_hidden), dtype=np.float32))
        s.append(np.zeros(shape=(params["batch_size"],n_hidden), dtype=np.float32))
        LStateList.append(tuple(s))
    return LStateList


def get_state_list(params):
    dic_state={}
    if params["model"]=="lstm":
        dic_state["lstm_t"]=get_zero_state(params)
        dic_state["lstm_pre"]=get_zero_state(params)
    else:
        dic_state["F_t"]=get_zero_state(params)
        dic_state["F_pre"]=get_zero_state(params)
        if params["model"]=="kfl_K":
            dic_state["K_t "]=get_zero_state(params, t='K')
            dic_state["K_pre "]=get_zero_state(params, t='K')

        if params["model"]=="kfl_QRFf":
            dic_state["Q_t "]=get_zero_state(params, t='Q')
            dic_state["Q_pre"]=get_zero_state(params, t='Q')
            dic_state["R_t"]=get_zero_state(params, t='R')
            dic_state["R_pre"]=get_zero_state(params, t='R')


    return dic_state

def get_random_state(params):
    LStateList=[]
    for i in range(params['nlayer']):
        s=[]
        s.append(np.random.uniform(size=(params["batch_size"],params['n_hidden'])))
        s.append(np.random.uniform(size=(params["batch_size"],params['n_hidden'])))
        LStateList.append(tuple(s))
    return LStateList


def save_state(seq_ls_internal,f,r):
    f=np.squeeze(f)
    r=np.squeeze(r)
    # print r.shape
    bpath="/mnt/Data1/hc/est/iv4/rnn/state/"
    print("New writing.....")
    # print(f.shape)
    for idx in range(len(f)):
        # print(f[idx].shape)
        # print("idxxxxxxxxxxxxxx")
        # print(f[idx])
        fname=bpath+f[idx][0].split('/')[-1].replace('.txt','')
        print(fname)
        isrepeat=r[idx]
        internal_state=seq_ls_internal[idx]
        if(isrepeat==1):
            np_state=np.asarray(internal_state)
            np.save(fname,np_state)


    print("Writing")
def create_file(log_file):
    log_dir= os.path.dirname(log_file)
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if(os.path.isfile(log_file)):
        with open(log_file, "w"):
            pass
    else:
        os.mknod(log_file)

def log_to_file(str,params):
    with open(params["log_file"], "a") as log:
        log.write(str)

def log_write(str,params,screen_print=False):
    if screen_print==True:
        print(str)
    ds= get_time()
    str=ds+" | "+str+"\n"
    log_to_file(str,params)

def get_init_fn(slim,params):
    if(params['run_mode']==0):
        log_write("Training from the scratch run mode: %s"%(params['run_mode']),params)
        return 0
    elif(params['run_mode']==1):
        """Returns a function run by the chief worker to warm-start the training."""
        checkpoint_exclude_scopes=params['checkpoint_exclude_scopes']

        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        # mfile=params["model_file"]
        # model_path = tf.train.latest_checkpoint(params["model_file"])
        model_path = params["model_file"]
        log_write("Imagenet model loaded: %s" % (model_path), params)
        return slim.assign_from_checkpoint_fn(model_path, variables_to_restore)

    elif (params['run_mode'] == 2):
        # mfile=get_last_modelname(params)
        # init_fn = slim.assign_from_checkpoint_fn(mfile,slim.get_model_variables(params['scope']))
        model_path = tf.train.latest_checkpoint(params["cp_file"])
        print ("Last fine-tuned model loaded: %s" % (model_path))
        init_fn = slim.assign_from_checkpoint_fn(model_path,slim.get_model_variables(params['scope']))
        log_write("Last fine-tuned model loaded: %s"%(model_path),params)
        return init_fn
    elif (params['run_mode'] == 3):
        model_path = params["model_file"]
        # model_path = tf.train.latest_checkpoint(params["cp_file"])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, slim.get_model_variables())
        log_write("Given model loaded: %s" % (model_path), params)
        return init_fn

def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
    if (accumulate >= x):
        return i
    print 'error with sampling ensemble'
    return -1

def draw_sample_from_mixture_guassian(out_pi, out_mu, out_sigma):
    shp=out_mu.shape
    d=shp[2]
    n=shp[0]
    result = np.zeros((n, d))
    m = out_pi.shape[1]
    for i in range(n):
        c = int(np.random.choice(range(m), size=1, replace=True, p=out_pi[i, :]))
        # c = int(np.random.choice(range(m), size=1, replace=True, p=out_pi[i, :]))
        # c=np.argmax(out_pi[i, :])
        # mu = out_mu[i, c * d:(c + 1) * d].ravel()
        mu = out_mu[i, c,:].ravel()
        sig = np.diag(out_sigma[i, c,:])
        sample_c = np.random.multivariate_normal(mu, sig ** 2, 1).ravel()
        result[i, :] = sample_c
    return result

def sample_gaussian_2d(mu_lst,sigma_lst):
    # mean = [mu1, mu2]
    # cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mu_lst, sigma_lst, 1)
    # x = np.random.multivariate_normal(mu_lst, sigma_lst, 1)
    return x

def get_last_modelname(params):
    cp=params["cp_file"]
    mfile=cp+'/checkpoint'
    # print mfile
    with open(mfile) as f:
        lines=f.readlines()
        mfile=lines[-1].replace('\n','')
    assert mfile != ''
    print(cp+mfile.split(':')[-1].replace('"','').strip().split('/')[-1])
    return cp+"/"+mfile.split(':')[-1].replace('"','').strip().split('/')[-1]

def save_db(params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list):
    fl=params["data_bin"]+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('F_list_train', data=F_list_train)
    h5f.create_dataset('G_list_train', data=G_list_train)
    h5f.create_dataset('S_Train_list', data=S_Train_list)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.create_dataset('F_list_test', data=F_list_test)
    h5f.create_dataset('G_list_test', data=G_list_test)
    h5f.create_dataset('S_Test_list', data=S_Test_list)
    h5f.close()
    # list=[]
    # list.append(X_train)
    # list.append(Y_train)
    # list.append(F_list_train)
    # list.append(G_list_train)
    # list.append(S_Train_list)
    # list.append(X_test)
    # list.append(Y_test)
    # list.append(F_list_test)
    # list.append(G_list_test)
    # list.append(S_Test_list)
    # nd_lst=np.asarray(list)
    # np.save(fl,nd_lst)

def load_db(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        X_train=f['X_train'][()]
        Y_train=f['Y_train'][()]
        F_list_train=f['F_list_train'][()]
        G_list_train=f['G_list_train'][()]
        S_Train_list=f['S_Train_list'][()]
        X_test=f['X_test'][()]
        Y_test=f['Y_test'][()]
        F_list_test=f['F_list_test'][()]
        G_list_test=f['G_list_test'][()]
        S_Test_list=f['S_Test_list'][()]
    return (X_train,Y_train,F_list_train,G_list_train,S_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    anp=np.asarray(a)
    bnp=np.asarray(b)
    p = np.random.permutation(len(a))
    return anp[p].tolist(), bnp[p].tolist()

#
def samples(sess,mixing,mu,sigma):
        component = tf.multinomial(pvals=mixing)
        component_mean =  tf.reduce_sum(mu *component.dimshuffle(0,'x',1),axis=2)
        component_std = tf.reduce_sum(sigma * component, axis=1, keepdims=True)
        samples = tf.random_normal(avg = component_mean, std=component_std)

        return samples

def select_max_idx(KMIX,NDIM,pi):
  idx=np.argmax(pi)
  return idx


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def select_weighted_mu(pi,mu):
    idx=np.argmax(pi)
    pi_idx=pi[idx]
    mu_idx=mu[idx]
    pimu=mu_idx*pi_idx
    return pimu

def get_weighted_pi(NDIM,pi,sigm):
    #pi=KMIX
    #sigma=KMIX
    #weight_pi=KMIX
    weight_pi=pi/np.power(sigm,float(NDIM)/float(2.0))
    return softmax(weight_pi)

def select_weightedpi_maxidx(NDIM,pi,sigm):
    weight_pi=get_weighted_pi(NDIM,pi,sigm)
    idx=np.argmax(weight_pi)
    # tmp_val=0.0
    # idx=0
    # for i in range(KMIX):
    #   val=np.sum(pi[i]/np.power(sigm[i],float(NDIM)/float(2.0)))
    #   if i==0:
    #     val=tmp_val
    #     idx=i
    #   if val>tmp_val:
    #     tmp_val=val
    #     idx=i
    return idx



def select_weightedpi_maxidx2(KMIX,NDIM,pi,sigm):
    # weight_pi=get_weighted_pi(NDIM,pi,sigm)
    # idx=np.argmax(weight_pi)
    tmp_val=0.0
    idx=0
    for i in range(KMIX):
      val=np.sum(pi[i]/np.power(sigm[i],float(NDIM)/float(2.0)))
      if i==0:
        val=tmp_val
        idx=i
      if val>tmp_val:
        tmp_val=val
        idx=i
    return idx



def generate_ensemble(out_pi, out_mu, out_sigma,sel_mode=0):
    NTEST=out_mu.shape[0]
    KMIX=out_mu.shape[1]
    NDIM=out_mu.shape[2]
    #out_pi=NTEST X KMIX
    #out_mu=NTEST X KMIX X NDIM
    #out_sigma=NTEST X KMIX
    result = np.random.rand(NTEST,NDIM) # initially random [0, 1]
    # rn = np.random.randn(NTEST,NDIM) # normal random matrix (0.0, 1.0)
    # mu = 0
    # std = 0
    # idx = 0
    # transforms result into random ensembles
    # for j in range(0, M):
    for i in range(0, NTEST):
    # idx = get_pi_idx(result[i, j], out_pi[i])
        if sel_mode==0: #select depend on gaussian ff pi
            idx = np.random.choice(range(KMIX),p=out_pi[i])
        if sel_mode==1: #select depend on gaussian ff pi
            weight_pi=get_weighted_pi(NDIM,out_pi[i],out_sigma[i])
            idx = np.random.choice(range(KMIX),p=weight_pi)
        elif sel_mode==2: #check variance also and get max
            idx= select_max_idx(KMIX,out_pi[i])
        elif sel_mode==3: #get pi,mu
            idx=select_weightedpi_maxidx(NDIM,out_pi[i],out_sigma[i])
        elif sel_mode==4: #get pi,mu
            idx=select_weightedpi_maxidx2(KMIX,NDIM,out_pi[i],out_sigma[i])

        if sel_mode==5:
          mu=  select_weighted_mu(out_pi[i],out_sigma[i])
        else:
          mu = out_mu[i, idx]

        xyz_np=mu
        # std = out_sigma[i, idx]
        # xyz_np= multivariate_normal(mu, std).pdf(x_test[i])
        result[i,:] = xyz_np
    return result

def complete_normalise_data(trainx,trainy,testx,testy):
    full=np.vstack((trainx,trainy))
    ln=full.shape[1]
    tmp_full=full[:,1:ln]

    tmp_testx=testx[:,1:ln]
    tmp_trainx=trainx[:,1:ln]
    tmp_testy=testy[:,1:ln]
    tmp_trainy=trainy[:,1:ln]

    men=np.mean(tmp_full, axis = 0,dtype=np.float32) # zero-center the data (important)
    std=np.std(tmp_full,axis=0,dtype=np.float32)

    tmp_testx=(tmp_testx-men)/std
    tmp_trainx=(tmp_trainx-men)/std
    tmp_testy=(tmp_testy-men)/std
    tmp_trainy=(tmp_trainy-men)/std

    trainx[:,1:ln]=tmp_trainx
    testx[:,1:ln]=tmp_testx
    trainy[:,1:ln]=tmp_trainy
    testy[:,1:ln]=tmp_testy

    return trainx,trainy,testx,testy,men,std

def normalise_data(train, test):
    ln=train.shape[1]
    tmp_test=test[:,1:ln]
    tmp_train=train[:,1:ln]

    men=np.mean(tmp_train, axis = 0,dtype=np.float32) # zero-center the data (important)
    std=np.std(tmp_train,axis=0,dtype=np.float32)
    tmp_x_test=(tmp_test-men)/std
    tmp_x_train=(tmp_train-men)/std
    train[:,1:ln]=tmp_x_train
    test[:,1:ln]=tmp_x_test

    return train,test,men,std

def unNormalizeData(origData,data_mean,data_std):
	T = origData.shape[0]
	D = data_mean.shape[0]

	stdMat = data_std.reshape((1,D))
	stdMat = np.repeat(stdMat,T,axis=0)
	meanMat = data_mean.reshape((1,D))
	meanMat = np.repeat(meanMat,T,axis=0)
	origData = np.multiply(origData,stdMat) + meanMat
	return origData

def shufle_data(index_list):
    index_shuf = range(len(index_list))
    index_list=np.asarray(index_list)[index_shuf]
    return (index_list)

