import os
import h5py
import time
from itertools import cycle
import collections
import  numpy as np
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool
from preprocessing import human36m_preprocessing
from preprocessing import vgg_preprocessing
from random import randint

def read_record(input_queue,params):

  class Record(object):
    pass
  result = Record()

  file_contents = tf.read_file(input_queue[1])
  # y_d=tf.decode_raw(file,out_type=tf.string)
  # tf.decode_csv(file_contents, record_defaults=record_defaults, field_delim=' ')

  # textreader = tf.TextLineReader()
  # label_key, label_value = textreader.read(input_queue[1])
  record_defaults = [[1.0 for col in range(1)] for row in range(params['n_output']+1)]
  res = tf.decode_csv(file_contents, record_defaults=record_defaults, field_delim=' ')
  res= [tf.select(tf.is_nan(res[row]), 0.0, res[row]/1000) for row in range(params['n_output'])]
  y_d = tf.pack(res)
  result.label = y_d
  result.label_name=input_queue[1]

  img_contents = tf.read_file(input_queue[0])
  my_img =  tf.image.decode_png(img_contents, channels=3)
  result.image = my_img
  result.image_name=input_queue[0]

  return result

def distorted_inputs(params,is_training):
    batch_size=params['batch_size']
    height =params['height']
    width = params['width']
    write_est= params['write_est']
    if (is_training==True):
        (joint_files,img_files)=params['training_files']
        isshuffle=True
        num_epochs=1
    else:
        (joint_files,img_files)=params['test_files']
        isshuffle=False
        num_epochs=1

    img_files, joint_files=img_files, joint_files

    input_queue = tf.train.slice_input_producer([img_files, joint_files], shuffle=isshuffle,num_epochs=num_epochs)
    read_input =read_record(input_queue,params)

    # reshaped_image = tf.cast(read_input.image, tf.float32)
    if 'vgg' in params["model"]:
        print('vgg preprocessing running...')
        image=vgg_preprocessing.preprocess_image(read_input.image, output_height=height, output_width=width,
                                                 is_training=is_training,resize_side_min=256,resize_side_max=300)
        # image= human36m_preprocessing.preprocess_image(read_input.image, height, width, is_training=is_training)
    else:
        image= human36m_preprocessing.preprocess_image(read_input.image, height, width, is_training=is_training)
    # image_raw = tf.expand_dims(image, 0)
    # image_raw = tf.image.resize_images(image_raw, height, width)
    # image_raw = tf.squeeze(image_raw)

    # Read examples from files in the filename queue.
    # read_input = read_record(img_filename_queue,joints_filename_queue,params)
    # read_input = read_record(input_queue,params)

    min_fraction_of_examples_in_queue=0.4
    min_queue_examples = 1000 * min_fraction_of_examples_in_queue
    capacity=min_queue_examples + 3 * batch_size
    if(write_est==True):
        tensor_list=[image, read_input.label,read_input.image_name,read_input.label_name]
    else:
        tensor_list=[image, read_input.label]

    if (is_training==True):
        batch = tf.train.shuffle_batch(
            tensors=tensor_list,
            min_after_dequeue=batch_size,batch_size=batch_size,
            num_threads=100, capacity=capacity, allow_smaller_final_batch=True)
    else:
        batch = tf.train.batch(
            tensors=tensor_list,
            batch_size=batch_size,
            num_threads=100,
            capacity=capacity, allow_smaller_final_batch=True)


    return batch

def multi_thr_read_full_midlayer_sequence(params,is_training):
    p_count=params['seq_length']
    # base_file=params['est_file']
    base_file=params['data_dir']+"/joints"
    est_file=params['data_dir']+"/est"
    max_count=params['max_count']
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]

    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    start = time.time()
    acto_cnt=0
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(Y_D),passed_time)
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            joint_tmp_folder=base_file+'/'+actor+"/"+sq+"/"
            mid_tmp_folder=est_file+'/'+actor+"/"+sq+"/"

            if os.path.exists(joint_tmp_folder)==False:
                continue
            if os.path.exists(mid_tmp_folder)==False:
                continue

            joint_id_list=os.listdir(joint_tmp_folder)
            mid_id_list=os.listdir(mid_tmp_folder)


            common_lst=[id for id in joint_id_list if id in mid_id_list]

            joint_list=[base_file+'/'+actor+'/'+sq+'/'+p1  for p1 in common_lst]
            midlayer_list=[est_file+'/'+actor+'/'+sq+'/'+p1 for p1 in common_lst]

            f_list=zip(joint_list,midlayer_list)

            pool = ThreadPool(1)
            results = pool.map(load_file, f_list)
            pool.close()

            for r in range(len(results)):
                rs=results[r][0]
                mid_rs=results[r][1]
                f=midlayer_list[r]
                Y_d.append(rs)
                X_d.append(mid_rs)
                F_l.append(f)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        X_D.append(X_d)
                        F_L.append(F_l)
                        S_L.append(seq_id)
                        Y_d=[]
                        X_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)
        if(len(Y_d)>0):
            residual=len(Y_d)%p_count
            residual=p_count-residual
            y=residual*[Y_d[-1]]
            x=residual*[X_d[-1]]
            f=residual*[F_l[-1]]
            Y_d.extend(y)
            X_d.extend(x)
            F_l.extend(f)
            if len(Y_d)==p_count and p_count>0:
                S_L.append(seq_id)
                Y_D.append(Y_d)
                X_D.append(X_d)
                F_L.append(F_l)
                Y_d=[]
                X_d=[]
                F_l=[]
                if len(Y_D)>=max_count:
                    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)


    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)

def prepare_sequences(params,db_values_x,db_values_y,db_names):
    p_count=params['seq_length']
    max_count=params['max_count']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    Y_d=[]
    X_d=[]
    F_l=[]
    prev_sq_id=0
    for item_id in range(len(db_names)):
        f=db_names[item_id]
        sq_id=int(db_values_x[item_id][0])
        x=db_values_x[item_id][1:]
        y=db_values_y[sq_id]

        if prev_sq_id!=sq_id:
            prev_sq_id=sq_id
            if(len(Y_d)>0):
                residual=len(Y_d)%p_count
                residual=p_count-residual
                y=residual*[Y_d[-1]]
                x=residual*[X_d[-1]]
                f=residual*[F_l[-1]]
                Y_d.extend(y)
                X_d.extend(x)
                F_l.extend(f)
                if len(Y_d)==p_count and p_count>0:
                    S_L.append(sq_id)
                    Y_D.append(Y_d)
                    X_D.append(X_d)
                    F_L.append(F_l)
                    Y_d=[]
                    X_d=[]
                    F_l=[]
                    if len(Y_D)>=max_count:
                        return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)
        else:
            Y_d.append(y)
            X_d.append(x)
            F_l.append(f)
        if len(Y_d)==p_count and p_count>0:
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            S_L.append(sq_id)
            Y_d=[]
            X_d=[]
            F_l=[]
        if len(Y_D)>=max_count:
                    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)

    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)

def prepare_db(params,is_training):
    # base_file=params['est_file']

    max_count=params["max_count"]
    if is_training==True:
        base_file="/mnt/Data3/Ansh/LSTM/LSTM-FEATURES"
        class_file="/mnt/Data3/Ansh/LSTM/namesWithLabels.txt"
    else:
        base_file="/mnt/Data3/Ansh/LSTM/LSTM-TEST"
        class_file="/home/kapila/Desktop/classInd.txt"

    nc=101 #number of classes
    db_names=[]
    start = time.time()
    seq_id=0
    seq_y=[]
    seq_x=[]
    class_lst={}
    with open(class_file, "rb") as f:
        data=f.readlines()
        for d in data:
            f=d.strip().split(' ')
            f1=f[0]
            f2=f[1]
            if is_training==True:
                class_lst[f1]=int(f2)
            else:
                f2=f2.replace('.txt','')
                class_lst[f2]=int(f1)-1

    lst_sq=os.listdir(base_file)[0:max_count]
    cnt=0
    for sq in lst_sq:
        fsq=sq
        if is_training==True:
            sq=sq.replace('.txt','')
        else:
            sq=sq.split('_')[1]
        end = time.time()
        passed_time=end-start
        cnt+=1
        if sq not in class_lst:
            print "%s sequence not loaded"%(sq)
            continue
        print "(%i/%i)-%s loading... %i loaded, time: %s "%(cnt,len(lst_sq),sq,len(db_names),passed_time)
        mid_tmp_folder=base_file+"/"+fsq

        with open(mid_tmp_folder, "rb") as f:
            data=f.readlines()
            for d in data:
                item=d.strip().split(' ')
                y_d= [np.float32(val) for val in item]
                y_d=np.asarray(y_d,dtype=np.float32)
                seq_x.append(np.hstack((seq_id,y_d)))
                f.close()
        c=class_lst[sq]
        onehot = np.zeros((1, nc))
        onehot[0, c] = 1
        seq_y.append(onehot)
        db_names.extend(mid_tmp_folder)
        seq_id+=1

    return (np.squeeze(np.asarray(seq_x)),np.squeeze(np.asarray(seq_y)),db_names)

def load_and_save_db(params):
    (db_values_x_training,db_values_y_training,db_names_training)=prepare_db(params,is_training=True)
    (db_values_x_test,db_values_y_test,db_names_test)=prepare_db(params,is_training=False)
    fl=params["data_bin"]+"-"+str(len(db_values_x_training))+"-"+str(len(db_values_x_test))+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('db_values_x_training', data=db_values_x_training)
    h5f.create_dataset('db_values_y_training', data=db_values_y_training)
    h5f.create_dataset('db_names_training', data=db_names_training)
    h5f.create_dataset('db_values_x_test', data=db_values_x_test)
    h5f.create_dataset('db_values_y_test', data=db_values_y_test)
    h5f.create_dataset('db_names_test', data=db_names_test)
    h5f.close()
    return (db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test)

def load_from_bin_db(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_values_x_training=f['db_values_x_training'][()]
        db_values_y_training=f['db_values_y_training'][()]
        db_names_training=f['db_names_training'][()]
        db_values_x_test=f['db_values_x_test'][()]
        db_values_y_test=f['db_values_y_test'][()]
        db_names_test=f['db_names_test'][()]

    return (db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test)

def prepare_training_set(params):
    if(params["reload_data"]==1):
        start = time.time()
        db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test= load_and_save_db(params)
        end = time.time()
        tm=end-start
        print "Fresh dataset saved and loaded: %s"%(tm)
    else:
        start = time.time()
        db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test= load_from_bin_db(params)
        end = time.time()
        tm=end-start
        print "Dataset loaded from the loacl: %s"%(tm)

    X_train,Y_train,F_list_train,G_list_train,S_Train_list=prepare_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
    del db_values_x_training
    del db_values_y_training
    del db_names_training
    # params["seq_length"]=-1
    X_test,Y_test,F_list_test,G_list_test,S_Test_list=prepare_sequences(params,db_values_x_test,db_values_y_test,db_names_test)
    del db_values_x_test
    del db_values_y_test
    del db_names_test
    return (X_train,Y_train,F_list_train,G_list_train,S_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list)

def load_file(fl):
    yfl=fl[0]
    xfl=fl[1]
    with open(yfl, "rb") as f:
        data=f.read().strip().split(' ')
        y_d= [np.float32(val) for val in data]
        y_d=np.asarray(y_d,dtype=np.float32)/1000
        f.close()

    with open(xfl, "rb") as f:
        data=f.read().strip().split(' ')
        x_d= [np.float32(val) for val in data]
        x_d=np.asarray(x_d,dtype=np.float32)
        f.close()
    return (y_d,x_d)

def load_file_nodiv(fl):
    with open(fl, "rb") as f:
        data=f.read().strip().split(' ')
        y_d= [np.float32(val) for val in data]
        y_d=np.asarray(y_d,dtype=np.float32)
        f.close()
        return y_d

def load_files(params,is_training):
    max_count=params['max_count']
    residual=max_count
    avarage=1500

    base_file=params['data_dir']
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]
    if("_v3" in params["model"] or "vgg" in params["model"] or "resnet" in params["model"]):
        img_folder=base_file+'/img350'
    else:
        img_folder=base_file+'/img'
    # img_folder=base_file+'/img'
    joints_file=base_file+'/joints'
    joint_files=[]
    img_files=[]
    for actor in lst_act:
        tmp_folder=img_folder+"/"+actor
        if not os.path.exists(tmp_folder):
            continue
        lst_sq=os.listdir(tmp_folder)
        if(len(params["action"])>0):
            sel_lst_sq=[]
            for l in lst_sq:
                pl=l.split('.')[0].split(' ')[0].replace('WalkingDog','WalkDog').replace('TakingPhoto','Photo')
                if params["action"] == pl:
                    sel_lst_sq.append(l)
            lst_sq=sel_lst_sq
        # print("Subject: %s, %i, %s || %s"%(actor,len(lst_sq),params["action"],','.join(lst_sq)))

        cnt=(residual/avarage)+1
        if(cnt<=len(lst_sq)):
            lst_sq=lst_sq[0:cnt]
        if(len(lst_sq)<1):
            continue
        pool = ThreadPool(len(lst_sq))
        args=zip(cycle([img_folder]),cycle([joints_file]),cycle([actor]),lst_sq)
        results=pool.map(load_seq_files, args)
        joint_flat_lst=[res[0] for res in results]
        img_flat_lst=[res[1] for res in results]
        joint_flat_lst=[item for sublist in joint_flat_lst for item in sublist]
        img_flat_lst=[item for sublist in img_flat_lst for item in sublist]
        residual=max_count-len(img_files)
        if(residual>0):
            if(residual>len(joint_flat_lst)):
                joint_files.extend(joint_flat_lst)
                img_files.extend(img_flat_lst)
            else:
                joint_files.extend(joint_flat_lst[0:residual])
                img_files.extend(img_flat_lst[0:residual])
        else:
            return (joint_files,img_files)

    return (joint_files,img_files)

def load_seq_files(args):
    img_folder,joints_file,actor,sq=args[0],args[1],args[2],args[3]
    joint_list=[]
    img_list=[]
    tmp_folder_img=img_folder+"/"+actor+"/"+sq+"/"
    tmp_folder_joints=joints_file+"/"+actor+"/"+sq+".cdf/"

    if os.path.exists(tmp_folder_joints)==False:
        return (joint_list,img_list)
    if os.path.exists(tmp_folder_img)==False:
        return (joint_list,img_list)

    joint_id_list=os.listdir(tmp_folder_joints)
    img_id_list=os.listdir(tmp_folder_img)

    for j in joint_id_list:
        img='frame_'+(j.replace('.txt','')).zfill(5)+'.png'
        if(img in img_id_list):
            joint_list.append(tmp_folder_joints+j)
            img_list.append(tmp_folder_img+img)
    return (joint_list,img_list)

def bi_prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,fw_LStateList,fw_LStateList_pre,bw_LStateList,bw_LStateList_pre, params, Y, X,state_reset_counter_lst):
    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    fw_new_S=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(len(fw_LStateList))]
    bw_new_S=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(len(bw_LStateList))]
    for idx in range(batch_size):
        state_reset_counter=state_reset_counter_lst[idx]
        if(minibatch_index==0):
            for s in range(len(fw_new_S)):
                fw_new_S[s][idx,:]=fw_LStateList[s][idx,:]
                bw_new_S[s][idx,:]=bw_LStateList[s][idx,:]
        elif(pre_sid[idx]!=curr_sid[idx]) or ((state_reset_counter%params['reset_state']==0) and state_reset_counter*params['reset_state']>0):
            for s in range(len(fw_new_S)):
                fw_new_S[s][idx,:]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                bw_new_S[s][idx,:]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
            state_reset_counter_lst[idx]=0
        elif (curr_id_lst[idx]==pre_id_lst[idx]): #If value repeated, we should repeat state also
            state_reset_counter_lst[idx]=state_reset_counter-1
            for s in range(len(fw_new_S)):
                fw_new_S[s][idx,:]=fw_LStateList_pre[s][idx,:]
                bw_new_S[s][idx,:]=bw_LStateList_pre[s][idx,:]
        else:
            for s in range(len(fw_new_S)):
                fw_new_S[s][idx,:]=fw_LStateList[s][idx,:]
                bw_new_S[s][idx,:]=bw_LStateList[s][idx,:]
    x=X[curr_id_lst]
    y=Y[curr_id_lst]

    return (fw_new_S,bw_new_S,x,y,state_reset_counter_lst)

def prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,LStateList,LStateList_pre, params, Y, X,state_reset_counter_lst):
    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    new_S=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(len(LStateList))]
    for idx in range(batch_size):
        state_reset_counter=state_reset_counter_lst[idx]
        if(minibatch_index==0):
            for s in range(len(new_S)):
                new_S[s][idx,:]=LStateList[s][idx,:]
        elif(pre_sid[idx]!=curr_sid[idx]) or ((state_reset_counter%params['reset_state']==0) and state_reset_counter*params['reset_state']>0):
            for s in range(len(new_S)):
                new_S[s][idx,:]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
            state_reset_counter_lst[idx]=0
        elif (curr_id_lst[idx]==pre_id_lst[idx]): #If value repeated, we should repeat state also
            state_reset_counter_lst[idx]=state_reset_counter-1
            for s in range(len(new_S)):
                new_S[s][idx,:]=LStateList_pre[s][idx,:]
        else:
            for s in range(len(new_S)):
                new_S[s][idx,:]=LStateList[s][idx,:]
    x=X[curr_id_lst]
    y=Y[curr_id_lst]


    return (new_S,x,y,state_reset_counter_lst)

def get_seq_indexes(params,S_L):
    bs=params['batch_size']
    new_S_L=[]
    counter=collections.Counter(S_L)
    lst=[list(t) for t  in counter.items()]
    a=np.asarray(lst)
    ss=a[a[:,1].argsort()][::-1]
    b_index=0
    new_index_lst=dict()
    b_index_lst=dict()
    count=0
    for item in ss:
        # seq_srt_intex= np.sum(a[0:item[0]-1],axis=0)[1]
        # seq_end_intex= seq_srt_intex+item[1]
        # count+=(seq_end_intex-seq_srt_intex)
        # if(len(S_L)<=seq_end_intex):
        #     print('sth wring')
        # sub_idx_lst=S_L[seq_srt_intex:seq_end_intex]
        # new_S_L.extend(sub_idx_lst)
        new_S_L.extend([item[0]]*item[1])

    for i in range(bs):
        b_index_lst[i]=0
    batch_inner_index=0
    for l_idx in range(len(new_S_L)):
        l=new_S_L[l_idx]
        if(l_idx>0):
            if(l!=new_S_L[l_idx-1]):
                for i in range(bs):
                    if(b_index>b_index_lst[i]):
                        b_index=b_index_lst[i]
                        batch_inner_index=i

        index=b_index*bs+batch_inner_index
        if(index in new_index_lst):
            print 'exist'
        new_index_lst[index]=l_idx
        b_index+=1
        b_index_lst[batch_inner_index]=b_index

    mx=max(b_index_lst.values())
    for b in b_index_lst.keys():
        b_index=b_index_lst[b]
        diff=mx-b_index
        if(diff>0):
            index=(b_index-1)*bs+b
            rpt=new_index_lst[index]
            for inc in range(diff):
                new_index=(b_index+inc)*bs+b
                new_index_lst[new_index]=rpt

    new_lst = collections.OrderedDict(sorted(new_index_lst.items())).values()
    return (new_lst,np.asarray(new_S_L))
