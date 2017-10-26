import os
import h5py
import time
from random import randint
from itertools import cycle
import collections
import  numpy as np
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool
from preprocessing import human36m_preprocessing
from preprocessing import vgg_preprocessing
import utils as ut
import locale
locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

def read_record(input_queue,params):

  class Record(object):
    pass
  result = Record()

  file_contents = tf.read_file(input_queue[1])
  # y_d=tf.decode_raw(file,out_type=tf.string)
  # tf.decode_csv(file_contents, record_defaults=record_defaults, field_delim=' ')

  # textreader = tf.TextLineReader()
  # label_key, label_value = textreader.read(input_queue[1])
  record_defaults = [[1.0 for col in range(1)] for row in range(params['n_output'])]
  res = tf.decode_csv(file_contents, record_defaults=record_defaults, field_delim=' ')
  res= [tf.where(tf.is_nan(res[row]), 0.0, res[row]) for row in range(params['n_output'])]
  y_d = tf.stack(res)
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
        print('human36m_preprocessing preprocessing running...')
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
            num_threads=500, capacity=capacity, allow_smaller_final_batch=True)
    else:

        # batch = tf.train.batch(
        #     tensors=tensor_list,
        #     batch_size=batch_size,
        #     num_threads=500,
        #     capacity=capacity, allow_smaller_final_batch=True)

        batch = tf.train.shuffle_batch(
            tensors=tensor_list,
            batch_size=batch_size,
            min_after_dequeue=batch_size,
            num_threads=500,
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

def subsample_frames(params,db_values_x,db_values_y,db_names):
    seq_rel={}

    ln=db_values_x.shape[1]
    max_seq=np.max(db_values_x[:,0])
    seq_rel["seq_idx_lst"]=db_values_x[:,0]
    tmp_db_values_x=np.zeros_like(db_values_x)
    tmp_db_values_y=np.zeros_like(db_values_y)
    tmp_db_names=np.zeros_like(db_names)
    sub_sample=params['subsample']+1
    new_id=0
    for idx in range(sub_sample):
        for item_id in range(len(db_names)):
            mod=item_id%sub_sample
            if mod==idx:
                sq_id=int(db_values_x[item_id][0])
                new_seq=sq_id+max_seq*mod #adding one ensures that we are not assigning same id with the maximum
                if sq_id not in seq_rel:
                    seq_rel[sq_id]=[new_seq]
                else:
                    lst=seq_rel[sq_id]
                    if new_seq not in lst:
                        lst.append(new_seq)
                        seq_rel[sq_id]=lst
                tmp_db_values_x[new_id][0]=new_seq
                tmp_db_values_y[new_id][0]=new_seq
                tmp_db_values_x[new_id,1:ln]=db_values_x[item_id,1:ln]
                tmp_db_values_y[new_id,1:ln]=db_values_y[item_id,1:ln]
                tmp_db_names[new_id]=db_names[item_id]
                new_id+=1
    return tmp_db_values_x,tmp_db_values_y,tmp_db_names,seq_rel

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
    R_L=[] #Repeating frame list....
    r_l=[]
    prev_sq_id=0
    curr_id=0
    for item_id in range(len(db_names)):
        f=db_names[item_id]
        sq_id=int(db_values_x[item_id][0])
        x=db_values_x[item_id][1:]
        y=db_values_y[item_id][1:]

        if prev_sq_id!=sq_id:
            prev_sq_id=sq_id
            if(len(Y_d)>0): #If there is left over from previus sequence add them...
                residual=len(Y_d)%p_count
                residual=p_count-residual
                res_y=residual*[Y_d[-1]]
                res_x=residual*[X_d[-1]]
                res_f=residual*[F_l[-1]]
                Y_d.extend(res_y)
                X_d.extend(res_x)
                F_l.extend(res_f)
                r_l.extend(residual*[0])
                if len(Y_d)==p_count and p_count>0:
                    S_L.append(curr_id)
                    Y_D.append(Y_d)
                    X_D.append(X_d)
                    F_L.append(F_l)
                    R_L.append(r_l)
                    Y_d=[]
                    X_d=[]
                    F_l=[]
                    r_l=[]
                    if len(Y_D)>=max_count:
                        return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L,np.asarray(R_L,dtype=np.int32))
            #we should add current frame to sequence.
            curr_id=sq_id
            Y_d.append(y)
            X_d.append(x)
            F_l.append(f)
            r_l.append(1)

        else:
            curr_id=sq_id
            Y_d.append(y)
            X_d.append(x)
            F_l.append(f)
            r_l.append(1)
        if len(Y_d)==p_count and p_count>0:
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            S_L.append(sq_id)
            R_L.append(r_l)
            Y_d=[]
            X_d=[]
            F_l=[]
            r_l=[]
        if len(Y_D)>=max_count:
                    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L,np.asarray(R_L,dtype=np.int32))

    if(len(Y_d)>0): #If there is left over from previus sequence add them...
        residual=len(Y_d)%p_count
        residual=p_count-residual
        res_y=residual*[Y_d[-1]]
        res_x=residual*[X_d[-1]]
        res_f=residual*[F_l[-1]]
        Y_d.extend(res_y)
        X_d.extend(res_x)
        F_l.extend(res_f)
        r_l.extend(residual*[0])
        if len(Y_d)==p_count and p_count>0:
            S_L.append(curr_id)
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            R_L.append(r_l)
    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L,np.asarray(R_L,dtype=np.int32))

def prepare_forcast_sequences(params,db_values_y,db_names,seq_rel):
    #Residuals are not important,
    fsc=params['forcast_sequence_count']
    seed_length=params['seed_length']
    forcast_length=params['forcast_length']
    seq_id_lst=seq_rel["seq_idx_lst"]
    forcast_id_lst=[]
    #random selection
    np.random.shuffle(seq_id_lst)
    selected_lst=seq_id_lst[:fsc]
    X_Seed=[]
    Y_Forcast=[]
    DB_names_Forcast=[]
    for s in selected_lst:
        id_lst=seq_rel[s]
        seed_start=-1
        for seq_ed in id_lst:
            s_idx=np.where(db_values_y[:,0]==seq_ed)
            sequence=db_values_y[s_idx]
            names_sequence=db_names[s_idx]
            if seed_start<0:
                full_length=sequence.shape[0]
                tmp_len=full_length-(seed_length+forcast_length)
                seed_start=randint(0,tmp_len)
                seed_end=seed_start+seed_length
                sequence_end=seed_start+seed_length+forcast_length
            x=sequence[seed_start:seed_end,1:]
            y=sequence[seed_start:sequence_end,1:]
            f=names_sequence[seed_start:sequence_end,1:]
            fd=[0]*(seed_end-seed_start)
            fd.extend([1]*(sequence_end-seed_end))
            X_Seed.append(x)
            Y_Forcast.append(y)
            forcast_id_lst.append(fd)
            DB_names_Forcast.append(f)
    return np.asarray(X_Seed), np.asarray(Y_Forcast),np.asarray(DB_names_Forcast),np.asarray(forcast_id_lst)

def prepare_prediction_sequences(params,db_values_x,db_values_y,db_names):
    #Residuals are not important,
    p_count=params['seq_length']
    max_count=params['max_count']
    forcast_distance=params['forcast_distance']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    Y_d=[]
    X_d=[]
    F_l=[]
    prev_sq_id=0
    for item_id in range(len(db_names)-forcast_distance):
        f=db_names[item_id]
        x_seq_id=int(db_values_y[item_id][0])
        y_seq_id=int(db_values_y[item_id+forcast_distance][0])
        if x_seq_id != y_seq_id:
            Y_d=[]
            X_d=[]
            F_l=[]
            continue

        x=db_values_y[item_id][1:]
        y=db_values_y[item_id+forcast_distance][1:]

        if prev_sq_id!=x_seq_id:
            Y_d=[]
            X_d=[]
            F_l=[]
            prev_sq_id=x_seq_id
            continue

        Y_d.append(y)
        X_d.append(x)
        F_l.append(f)

        if len(Y_d)==p_count and p_count>0:
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            S_L.append(x_seq_id)
            Y_d=[]
            X_d=[]
            F_l=[]
        if len(Y_D)>=max_count:
                    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)


    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L)


def prepare_save_db(params,is_training):
    # base_file=params['est_file']
    base_file=params['data_dir']
    max_count=params['max_count']
    lst_act=params["lst_act"]
    # if is_training==True:#load training data.
    #     lst_act=params["train_lst_act"]
    # else:
    #     lst_act=params["test_lst_act"]
    db_names=[]
    seq_id_names=[]
    start = time.time()
    acto_cnt=0
    seq_id=0
    seq_y=[]
    seq_x=[]
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
            joint_tmp_folder=base_file+'/'+actor+"/"+sq+"/"

            if os.path.exists(joint_tmp_folder)==False:
                continue

            joint_id_list=os.listdir(joint_tmp_folder)

            common_id_lst=sorted([int(f[0:-4]) for f in joint_id_list])

            joint_list=[base_file+'/'+actor+'/'+sq+'/'+str(p1)+".txt"  for p1 in common_id_lst]

            f_list=joint_list

            pool = ThreadPool(1000)
            results = pool.map(load_file, f_list)
            pool.close()
            for r in range(len(results)):
                seq_y.append(np.hstack((seq_id,results[r])))
            db_names.extend(f_list)

            seq_id_names.append(str(seq_id)+"|"+actor+"|"+sq)
            seq_id+=1
            if len(db_names) >max_count:
                return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

    return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

def prepare_prediction_db(params,is_training):
    # base_file=params['est_file']
    base_file=params['data_dir_y']+"/joints"
    max_count=params['max_count']
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]
    db_names=[]
    seq_id_names=[]
    start = time.time()
    acto_cnt=0
    seq_id=0
    seq_y=[]
    seq_x=[]
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
            joint_tmp_folder=base_file+'/'+actor+"/"+sq+"/"

            if os.path.exists(joint_tmp_folder)==False:
                continue

            joint_id_list=os.listdir(joint_tmp_folder)

            common_id_lst=sorted([int(f[0:-4]) for f in joint_id_list])

            joint_list=[base_file+'/'+actor+'/'+sq+'/'+str(p1)+".txt"  for p1 in common_id_lst]

            f_list=joint_list

            pool = ThreadPool(1000)
            results = pool.map(load_file, f_list)
            pool.close()
            for r in range(len(results)):
                seq_y.append(np.hstack((seq_id,results[r])))
            db_names.extend(f_list)

            seq_id_names.append(str(seq_id)+"|"+actor+"|"+sq)
            seq_id+=1
            if len(db_names) >max_count:
                return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

    return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

def prepare_db(params,is_training):
    # base_file=params['est_file']
    base_file=params['data_dir_y']+"/joints"
    est_file=params['data_dir_x']+"/fl_"+str(params['n_input'])
    max_count=params['max_count']
    print "Dataset loading from:  %s, %s " % (params['data_dir_x'],params['data_dir_y'])
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]
    db_names=[]
    seq_id_names=[]
    start = time.time()
    acto_cnt=0
    seq_id=0
    seq_y=[]
    seq_x=[]
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
            joint_tmp_folder=base_file+'/'+actor+"/"+sq+"/"
            mid_tmp_folder=est_file+'/'+actor+"/"+sq+"/"

            if os.path.exists(joint_tmp_folder)==False:
                continue
            if os.path.exists(mid_tmp_folder)==False:
                continue

            joint_id_list=os.listdir(joint_tmp_folder)
            mid_id_list=os.listdir(mid_tmp_folder)


            common_lst=[id for id in joint_id_list if id in mid_id_list]

            common_id_lst=sorted([int(f[0:-4]) for f in common_lst])

            joint_list=[base_file+'/'+actor+'/'+sq+'/'+str(p1)+".txt"  for p1 in common_id_lst]
            midlayer_list=[est_file+'/'+actor+'/'+sq+'/'+str(p1)+".txt" for p1 in common_id_lst]

            f_list=zip(joint_list,midlayer_list)

            pool = ThreadPool(1000)
            results = pool.map(load_file, f_list)
            pool.close()
            for r in range(len(results)):
                seq_y.append(np.hstack((seq_id,results[r][0])))
                seq_x.append(np.hstack((seq_id,results[r][1])))
            db_names.extend(f_list)

            seq_id_names.append(str(seq_id)+"|"+actor+"|"+sq)
            seq_id+=1
            if len(db_names) >max_count:
                return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

    return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)


def create_hdf5db(params):
    # base_file=params['est_file']
    base_file=params['fold_data_dir']
    lst_act = params["lst_act"]
    max_count=params['max_count']
    print "Dataset loading from:  %s" % (params['fold_data_dir'])
    db_names=[]
    seq_id_names=[]
    start = time.time()
    acto_cnt=0
    seq_id=0

    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            seq_x = []
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
            sq_tmp_folder=base_file+'/'+actor+"/"+sq+"/"

            sq_id_list=os.listdir(sq_tmp_folder)
            common_id_lst = sorted([int(f[0:-4]) for f in sq_id_list])
            sq_pathlist=[base_file+'/'+actor+'/'+sq+'/'+str(p1)+'.txt' for p1 in common_id_lst]


            pool = ThreadPool(1000)
            results = pool.map(load_file, sq_pathlist)
            pool.close()
            for r in range(len(results)):
                seq_x.append(results[r])
            dirr=params["data_bin"] +base_file.split('/')[-2]+'/'+actor
            if not os.path.exists(dirr):
                os.makedirs(dirr)
            fl = dirr+'/'+sq+ ".h5"
            h5f = h5py.File(fl, 'w')
            h5f.create_dataset('seq_x', data=seq_x)
            h5f.close()
            # print "Dataset save to:  %s" % (fl)

    return (np.asarray(seq_x),db_names,seq_id_names)

def load_and_save_db(params):
    if params['is_forcasting']==1:
        (db_values_x_training,db_values_y_training,db_names_training,seq_id_names_training)=prepare_prediction_db(params,is_training=True)
        (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)=prepare_prediction_db(params,is_training=False)
    else:
        (db_values_x_training,db_values_y_training,db_names_training,seq_id_names_training)=prepare_db(params,is_training=True)
        (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)=prepare_db(params,is_training=False)
    fl=params["data_bin"]+"-"+str(len(db_values_x_training))+"-"+str(len(db_values_x_test))+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('db_values_x_training', data=db_values_x_training)
    h5f.create_dataset('db_values_y_training', data=db_values_y_training)
    h5f.create_dataset('db_names_training', data=db_names_training)
    h5f.create_dataset('seq_id_names_training', data=seq_id_names_training)
    h5f.create_dataset('db_values_x_test', data=db_values_x_test)
    h5f.create_dataset('db_values_y_test', data=db_values_y_test)
    h5f.create_dataset('db_names_test', data=db_names_test)
    h5f.create_dataset('seq_id_names_test', data=seq_id_names_test)
    h5f.close()
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
        print "Dataset loaded from the local: %s"%(tm)

    if params["subsample"]>0:
        print("Subsampling frames %s"%params["subsample"])
        db_values_x_training,db_values_y_training,db_names_training,seq_rel_train=subsample_frames(params,db_values_x_training,db_values_y_training,db_names_training)
        db_values_x_test,db_values_y_test,db_names_test,seq_rel_test=subsample_frames(params,db_values_x_test,db_values_y_test,db_names_test)
        params["seq_rel_train"]=seq_rel_train
        params["seq_rel_test"]=seq_rel_test
    else:
        print("No subsampling frames %s"%params["frame_shift"])

    if params["normalise_data"]==1: #only normalise input data
        print("Normalasing the input values")
        db_values_x_training,db_values_x_test,men,std=ut.normalise_data(db_values_x_training, db_values_x_test)
        params["x_men"]=men
        params["x_std"]=std

    if params["normalise_data"]==2: #We normalise the target values.
        print("Normalasing the target values")
        db_values_y_training,db_values_y_test,men,std=ut.normalise_data(db_values_y_training, db_values_y_test)
        params["y_men"]=men
        params["y_std"]=std

    if params["normalise_data"]==3: #We normalise the target and input values.
        print("Normalasing the input and target values")
        db_values_x_training,db_values_x_test,men,std=ut.normalise_data(db_values_x_training, db_values_x_test)
        params["x_men"]=men
        params["x_std"]=std
        db_values_y_training,db_values_y_test,men,std=ut.normalise_data(db_values_y_training, db_values_y_test)
        params["y_men"]=men
        params["y_std"]=std
    if params["normalise_data"] == 4:  # We normalise the target and input values.
        print("Normalasing the input and target values with same std of entire training set")
        db_values_x_training, db_values_y_training, db_values_x_test, db_values_y_test, men, std = \
            ut.complete_normalise_data(db_values_x_training,db_values_y_training,db_values_x_test,db_values_y_test)
        params["x_men"] = men
        params["x_std"] = std
        # db_values_y_training, db_values_y_test, men, std = ut.normalise_data(db_values_y_training, db_values_y_test)
        # params["y_men"] = men
        # params["y_std"] = std

    if params['is_forcasting']==1:
        print("Preparing the data for next nth time forcasting")
        X_train,Y_train,F_list_train,G_list_train,S_Train_list=prepare_prediction_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
        X_test,Y_test,F_list_test,G_list_test,S_Test_list=prepare_prediction_sequences(params,db_values_x_test,db_values_y_test,db_names_test)
        return (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,db_values_y_training,db_names_training,X_test,Y_test,F_list_test,G_list_test,S_Test_list,db_values_y_test,db_names_test)
    else:
        print("Preparing the data for current time prediction")
        X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list=prepare_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
        del db_values_x_training
        del db_values_y_training
        del db_names_training
        X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list=prepare_sequences(params,db_values_x_test,db_values_y_test,db_names_test)
        del db_values_x_test
        del db_values_y_test
        del db_names_test
        return (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)

def prepare_char_dataset(params):
    base_file=params['data_dir']+"/joints"

def load_from_bin_db(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_values_x_training=f['db_values_x_training'][()].astype(dtype=np.float32)
        db_values_y_training=f['db_values_y_training'][()].astype(dtype=np.float32)
        db_names_training=f['db_names_training'][()]
        db_values_x_test=f['db_values_x_test'][()].astype(dtype=np.float32)
        db_values_y_test=f['db_values_y_test'][()].astype(dtype=np.float32)
        db_names_test=f['db_names_test'][()]

    return (db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test)

def load_test_from_bin_db(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_values_x_test=f['db_values_x_test'][()].astype(dtype=np.float32)
        db_values_y_test=f['db_values_y_test'][()].astype(dtype=np.float32)
        db_names_test=f['db_names_test'][()]
        seq_id_names_test=f['seq_id_names_test'][()]

    return (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)

def prepare_sequences_fnames(params,db_names,db_seq_id):
    p_count=params['seq_length']
    max_count=params['max_count']
    S_L=[]
    F_L=[]
    F_l=[]
    R_L=[]
    prev_sq_id=0
    for item_id in range(len(db_names)):
        f=db_names[item_id]
        sq_id=db_seq_id[item_id]

        if prev_sq_id!=sq_id:
            prev_sq_id=sq_id
            if(len(F_l)>0):
                residual=len(F_l)%p_count
                residual=p_count-residual
                f=residual*[F_l[-1]]
                F_l.extend(f)
                if len(F_l)==p_count and p_count>0:
                    S_L.append(sq_id)
                    F_L.append(F_l)
                    F_l=[]
                    if len(F_l)>=max_count:
                        return np.asarray(F_L),S_L
        else:
            F_l.append(f)
        if len(F_l)==p_count and p_count>0:
            F_L.append(F_l)
            S_L.append(sq_id)
            F_l=[]
        if len(F_L)>=max_count:
                    return np.asarray(F_L),S_L

    return np.asarray(F_L),S_L

def prepare_db_flnames(params,is_training):
    # base_file=params['est_file']
    base_file=params['data_dir']+"/joints"
    est_file=params['data_dir']+"/mid2048"
    max_count=params['max_count']
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]

    db_names=[]
    db_seq_id=[]
    start = time.time()
    acto_cnt=0
    seq_id=0
    seq_y=[]
    seq_x=[]
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
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
            db_names.extend(f_list)
            db_seq_id.extend([seq_id]*len(f_list))

            if(len(db_names)>max_count):
                return (db_names,db_seq_id)
            seq_id+=1

    return (db_names,db_seq_id)

def load_and_save_db_fnames(params):
    db_names_training,db_seq_id_training=prepare_db_flnames(params,is_training=True)
    db_names_test,db_seq_id_test=prepare_db_flnames(params,is_training=False)
    fl=params["data_bin"]+"-"+str(len(db_names_training))+"-"+str(len(db_names_test))+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('db_names_training', data=db_names_training)
    h5f.create_dataset('db_seq_id_training', data=db_seq_id_training)
    h5f.create_dataset('db_names_test', data=db_names_test)
    h5f.create_dataset('db_seq_id_test', data=db_seq_id_test)
    h5f.close()
    return (db_names_training,db_seq_id_training,db_names_test,db_seq_id_test)

def load_from_bin_db_fnames(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_names_training=f['db_names_training'][()]
        db_seq_id_training=f['db_seq_id_training'][()]
        db_names_test=f['db_names_test'][()]
        db_seq_id_test=f['db_seq_id_test'][()]

    return (db_names_training,db_seq_id_training,db_names_test,db_seq_id_test)

def prepare_training_set_fnames(params):
    if(params["reload_data"]==1):
        start = time.time()
        db_names_training,db_seq_id_training,db_names_test,db_seq_id_test= load_and_save_db_fnames(params)
        end = time.time()
        tm=end-start
        print "Fresh dataset saved and loaded: %s"%(tm)
    else:
        start = time.time()
        db_names_training,db_seq_id_training,db_names_test,db_seq_id_test= load_from_bin_db_fnames(params)
        end = time.time()
        tm=end-start
        print "Dataset loaded from the loacl: %s"%(tm)

    F_names_training,S_Train_list=prepare_sequences_fnames(params,db_names_training,db_seq_id_training)
    del db_names_training
    F_names_test,S_Test_list=prepare_sequences_fnames(params,db_names_test,db_seq_id_test)
    del db_names_test
    return (F_names_training,S_Train_list,F_names_test,S_Test_list)

def load_file(fl):
    if len(fl)>10:
        yfl=fl
        with open(yfl, "rb") as f:
            data=f.read().strip().split(' ')
            y_d= [np.float32(val) for val in data]
            y_d=np.asarray(y_d,dtype=np.float32)/1000
            f.close()
        return (y_d)

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
        if is_training == True:
            img_folder=base_file+'/ntu-36m'
        else:
            # img_folder = base_file + '/ntu-36m'
            img_folder = base_file + '/img350'
        # img_folder=base_file+'/img350'
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
        print 'read listt...................'
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
    tmp_folder_joints=joints_file+"/"+actor+"/"+sq+"/"

    if os.path.exists(tmp_folder_joints)==False:
        return (joint_list,img_list)
    if os.path.exists(tmp_folder_img)==False:
        return (joint_list,img_list)

    joint_id_list=os.listdir(tmp_folder_joints)
    img_id_list=os.listdir(tmp_folder_img)

    for j in joint_id_list:
        img=(j.replace('.txt',''))+'.png'
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

def prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,LStateList,LStateList_pre, params, Y, X,R_L_list,F_list,state_reset_counter_lst):
    #index_list= list of ids for sequences..
    #LStateList current states
    #LStateList_pre previus batch states
    #repeat list with in a single sequnce.
    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    new_S=ut.get_zero_state(params)

    if(minibatch_index>0):
        for idx in range(batch_size):
            state_reset_counter=state_reset_counter_lst[idx]
            if state_reset_counter%params['reset_state']==0 and params['reset_state']>0:
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx]=0
            elif(pre_sid[idx]!=curr_sid[idx]):# if sequence changed reset state also
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx]=0
            elif (curr_id_lst[idx]==pre_id_lst[idx]): #if we repeated the value we should repeat state also.
                state_reset_counter_lst[idx]=state_reset_counter-1
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=LStateList_pre[s][0][idx]
                    new_S[s][1][idx]=LStateList_pre[s][1][idx]
                    # new_S[s][idx,:]=LStateList_pre[s][idx,:]
            else:
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=LStateList[s][0][idx]
                    new_S[s][1][idx]=LStateList[s][1][idx]
                    # new_S[s][idx,:]=LStateList[s][idx,:]
    x=X[curr_id_lst]
    y=Y[curr_id_lst]
    if R_L_list != None:
        r=R_L_list[curr_id_lst]
    else:
        r=None
    f=F_list[curr_id_lst]


    return (new_S,x,y,r,f,state_reset_counter_lst)
#
#
# def prepare_lstm_batch_joints(index_list, minibatch_index, batch_size, S_list,LStateList,LStateList_pre, params, F_names,state_reset_counter_lst):
#     curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
#     pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
#     curr_sid=S_list[curr_id_lst]
#     pre_sid=S_list[pre_id_lst]
#     new_S=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(len(LStateList))]
#     for idx in range(batch_size):
#         state_reset_counter=state_reset_counter_lst[idx]
#         if(minibatch_index==0):
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=LStateList[s][idx,:]
#         elif(pre_sid[idx]!=curr_sid[idx]) or ((state_reset_counter%params['reset_state']==0) and state_reset_counter*params['reset_state']>0):
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
#             state_reset_counter_lst[idx]=0
#         elif (curr_id_lst[idx]==pre_id_lst[idx]): #If value repeated, we should repeat state also
#             state_reset_counter_lst[idx]=state_reset_counter-1
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=LStateList_pre[s][idx,:]
#         else:
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=LStateList[s][idx,:]
#     flist=F_names[curr_id_lst]
#     x=X[curr_id_lst]
#     y=Y[curr_id_lst]
#
#
#     return (new_S,x,y,state_reset_counter_lst)

def get_action_dataset(params,X,Y,F_list,G_list,S_list,R_L_list):
    action= params["action"]
    seq_id_lst=[]
    for index in range(F_list.shape[0]):
        seq=F_list[index]
        if action in seq[0][0]:
            seq_id_lst.append(index)

    X=X[seq_id_lst]
    Y=Y[seq_id_lst]
    F_list=F_list[seq_id_lst]
    # G_list=np.asarray(G_list)[seq_id_lst].tolist()
    S_list=np.asarray(S_list)[seq_id_lst].tolist()
    R_L_list=R_L_list[seq_id_lst]

    return X,Y,F_list,G_list,S_list,R_L_list

def get_action_seq_indexes(params,S_L,F_list):
    bs=params['batch_size']
    new_S_L=[]
    counter=collections.Counter(S_L)
    lst=[list(t) for t  in counter.items()]
    a=np.asarray(lst)
    ss=a[a[:,1].argsort()][::-1]
    b_index=0
    new_index_lst=dict()
    b_index_lst=dict()

    for item in ss:
        # seq_srt_intex= np.sum(a[0:item[0]-1],axis=0)[1]
        # seq_end_intex= seq_srt_intex+item[1]
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


def prepare_db(params,is_training):
    # base_file=params['est_file']
    base_file=params['data_dir_y']+"/joints"
    est_file=params['data_dir_x']+"/fl_"+str(params['n_input'])
    max_count=params['max_count']
    print "Dataset loading from:  %s, %s " % (params['data_dir_x'],params['data_dir_y'])
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]
    db_names=[]
    seq_id_names=[]
    start = time.time()
    acto_cnt=0
    seq_id=0
    seq_y=[]
    seq_x=[]
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
            joint_tmp_folder=base_file+'/'+actor+"/"+sq+"/"
            mid_tmp_folder=est_file+'/'+actor+"/"+sq+"/"

            if os.path.exists(joint_tmp_folder)==False:
                continue
            if os.path.exists(mid_tmp_folder)==False:
                continue

            joint_id_list=os.listdir(joint_tmp_folder)
            mid_id_list=os.listdir(mid_tmp_folder)


            common_lst=[id for id in joint_id_list if id in mid_id_list]

            common_id_lst=sorted([int(f[0:-4]) for f in common_lst])

            joint_list=[base_file+'/'+actor+'/'+sq+'/'+str(p1)+".txt"  for p1 in common_id_lst]
            midlayer_list=[est_file+'/'+actor+'/'+sq+'/'+str(p1)+".txt" for p1 in common_id_lst]

            f_list=zip(joint_list,midlayer_list)

            pool = ThreadPool(1000)
            results = pool.map(load_file, f_list)
            pool.close()
            for r in range(len(results)):
                seq_y.append(np.hstack((seq_id,results[r][0])))
                seq_x.append(np.hstack((seq_id,results[r][1])))
            db_names.extend(f_list)

            seq_id_names.append(str(seq_id)+"|"+actor+"|"+sq)
            seq_id+=1
            if len(db_names) >max_count:
                return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

    return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

def load_and_save_db(params):
    if params['is_forcasting']==1:
        (db_values_x_training,db_values_y_training,db_names_training,seq_id_names_training)=prepare_prediction_db(params,is_training=True)
        (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)=prepare_prediction_db(params,is_training=False)
    else:
        (db_values_x_training,db_values_y_training,db_names_training,seq_id_names_training)=prepare_db(params,is_training=True)
        (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)=prepare_db(params,is_training=False)
    fl=params["data_bin"]+"-"+str(len(db_values_x_training))+"-"+str(len(db_values_x_test))+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('db_values_x_training', data=db_values_x_training)
    h5f.create_dataset('db_values_y_training', data=db_values_y_training)
    h5f.create_dataset('db_names_training', data=db_names_training)
    h5f.create_dataset('seq_id_names_training', data=seq_id_names_training)
    h5f.create_dataset('db_values_x_test', data=db_values_x_test)
    h5f.create_dataset('db_values_y_test', data=db_values_y_test)
    h5f.create_dataset('db_names_test', data=db_names_test)
    h5f.create_dataset('seq_id_names_test', data=seq_id_names_test)
    h5f.close()
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
        print "Dataset loaded from the local: %s"%(tm)

    if params["subsample"]>0:
        print("Subsampling frames %s"%params["subsample"])
        db_values_x_training,db_values_y_training,db_names_training,seq_rel_train=subsample_frames(params,db_values_x_training,db_values_y_training,db_names_training)
        db_values_x_test,db_values_y_test,db_names_test,seq_rel_test=subsample_frames(params,db_values_x_test,db_values_y_test,db_names_test)
        params["seq_rel_train"]=seq_rel_train
        params["seq_rel_test"]=seq_rel_test
    else:
        print("No subsampling frames %s"%params["frame_shift"])

    if params["normalise_data"]==1: #only normalise input data
        print("Normalasing the input values")
        db_values_x_training,db_values_x_test,men,std=ut.normalise_data(db_values_x_training, db_values_x_test)
        params["x_men"]=men
        params["x_std"]=std

    if params["normalise_data"]==2: #We normalise the target values.
        print("Normalasing the target values")
        db_values_y_training,db_values_y_test,men,std=ut.normalise_data(db_values_y_training, db_values_y_test)
        params["y_men"]=men
        params["y_std"]=std

    if params["normalise_data"]==3: #We normalise the target and input values.
        print("Normalasing the input and target values")
        db_values_x_training,db_values_x_test,men,std=ut.normalise_data(db_values_x_training, db_values_x_test)
        params["x_men"]=men
        params["x_std"]=std
        db_values_y_training,db_values_y_test,men,std=ut.normalise_data(db_values_y_training, db_values_y_test)
        params["y_men"]=men
        params["y_std"]=std
    if params["normalise_data"] == 4:  # We normalise the target and input values.
        print("Normalasing the input and target values with same std of entire training set")
        db_values_x_training, db_values_y_training, db_values_x_test, db_values_y_test, men, std = \
            ut.complete_normalise_data(db_values_x_training,db_values_y_training,db_values_x_test,db_values_y_test)
        params["x_men"] = men
        params["x_std"] = std
        # db_values_y_training, db_values_y_test, men, std = ut.normalise_data(db_values_y_training, db_values_y_test)
        # params["y_men"] = men
        # params["y_std"] = std

    if params['is_forcasting']==1:
        print("Preparing the data for next nth time forcasting")
        X_train,Y_train,F_list_train,G_list_train,S_Train_list=prepare_prediction_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
        X_test,Y_test,F_list_test,G_list_test,S_Test_list=prepare_prediction_sequences(params,db_values_x_test,db_values_y_test,db_names_test)
        return (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,db_values_y_training,db_names_training,X_test,Y_test,F_list_test,G_list_test,S_Test_list,db_values_y_test,db_names_test)
    else:
        print("Preparing the data for current time prediction")
        X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list=prepare_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
        del db_values_x_training
        del db_values_y_training
        del db_names_training
        X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list=prepare_sequences(params,db_values_x_test,db_values_y_test,db_names_test)
        del db_values_x_test
        del db_values_y_test
        del db_names_test
        return (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)

def prepare_char_dataset(params):
    base_file=params['data_dir']+"/joints"

def load_from_bin_db(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_values_x_training=f['db_values_x_training'][()].astype(dtype=np.float32)
        db_values_y_training=f['db_values_y_training'][()].astype(dtype=np.float32)
        db_names_training=f['db_names_training'][()]
        db_values_x_test=f['db_values_x_test'][()].astype(dtype=np.float32)
        db_values_y_test=f['db_values_y_test'][()].astype(dtype=np.float32)
        db_names_test=f['db_names_test'][()]

    return (db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test)

def load_test_from_bin_db(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_values_x_test=f['db_values_x_test'][()].astype(dtype=np.float32)
        db_values_y_test=f['db_values_y_test'][()].astype(dtype=np.float32)
        db_names_test=f['db_names_test'][()]
        seq_id_names_test=f['seq_id_names_test'][()]

    return (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)

def prepare_sequences_fnames(params,db_names,db_seq_id):
    p_count=params['seq_length']
    max_count=params['max_count']
    S_L=[]
    F_L=[]
    F_l=[]
    R_L=[]
    prev_sq_id=0
    for item_id in range(len(db_names)):
        f=db_names[item_id]
        sq_id=db_seq_id[item_id]

        if prev_sq_id!=sq_id:
            prev_sq_id=sq_id
            if(len(F_l)>0):
                residual=len(F_l)%p_count
                residual=p_count-residual
                f=residual*[F_l[-1]]
                F_l.extend(f)
                if len(F_l)==p_count and p_count>0:
                    S_L.append(sq_id)
                    F_L.append(F_l)
                    F_l=[]
                    if len(F_l)>=max_count:
                        return np.asarray(F_L),S_L
        else:
            F_l.append(f)
        if len(F_l)==p_count and p_count>0:
            F_L.append(F_l)
            S_L.append(sq_id)
            F_l=[]
        if len(F_L)>=max_count:
                    return np.asarray(F_L),S_L

    return np.asarray(F_L),S_L

def prepare_db_flnames(params,is_training):
    # base_file=params['est_file']
    base_file=params['data_dir']+"/joints"
    est_file=params['data_dir']+"/mid2048"
    max_count=params['max_count']
    if is_training==True:#load training data.
        lst_act=params["train_lst_act"]
    else:
        lst_act=params["test_lst_act"]

    db_names=[]
    db_seq_id=[]
    start = time.time()
    acto_cnt=0
    seq_id=0
    seq_y=[]
    seq_x=[]
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        acto_cnt+=1
        cnt=0
        for sq in lst_sq:
            end = time.time()
            passed_time=end-start
            cnt+=1
            print "%s, (%i/%i)-%s loading... %i loaded, time: %s "%(actor,cnt,len(lst_sq),sq,len(db_names),passed_time)
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
            db_names.extend(f_list)
            db_seq_id.extend([seq_id]*len(f_list))

            if(len(db_names)>max_count):
                return (db_names,db_seq_id)
            seq_id+=1

    return (db_names,db_seq_id)

def load_and_save_db_fnames(params):
    db_names_training,db_seq_id_training=prepare_db_flnames(params,is_training=True)
    db_names_test,db_seq_id_test=prepare_db_flnames(params,is_training=False)
    fl=params["data_bin"]+"-"+str(len(db_names_training))+"-"+str(len(db_names_test))+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('db_names_training', data=db_names_training)
    h5f.create_dataset('db_seq_id_training', data=db_seq_id_training)
    h5f.create_dataset('db_names_test', data=db_names_test)
    h5f.create_dataset('db_seq_id_test', data=db_seq_id_test)
    h5f.close()
    return (db_names_training,db_seq_id_training,db_names_test,db_seq_id_test)

def load_from_bin_db_fnames(params):
    fl=params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_names_training=f['db_names_training'][()]
        db_seq_id_training=f['db_seq_id_training'][()]
        db_names_test=f['db_names_test'][()]
        db_seq_id_test=f['db_seq_id_test'][()]

    return (db_names_training,db_seq_id_training,db_names_test,db_seq_id_test)

def prepare_training_set_fnames(params):
    if(params["reload_data"]==1):
        start = time.time()
        db_names_training,db_seq_id_training,db_names_test,db_seq_id_test= load_and_save_db_fnames(params)
        end = time.time()
        tm=end-start
        print "Fresh dataset saved and loaded: %s"%(tm)
    else:
        start = time.time()
        db_names_training,db_seq_id_training,db_names_test,db_seq_id_test= load_from_bin_db_fnames(params)
        end = time.time()
        tm=end-start
        print "Dataset loaded from the loacl: %s"%(tm)

    F_names_training,S_Train_list=prepare_sequences_fnames(params,db_names_training,db_seq_id_training)
    del db_names_training
    F_names_test,S_Test_list=prepare_sequences_fnames(params,db_names_test,db_seq_id_test)
    del db_names_test
    return (F_names_training,S_Train_list,F_names_test,S_Test_list)

def load_file(fl):
    if len(fl)>10:
        yfl=fl
        with open(yfl, "rb") as f:
            data=f.read().strip().split(' ')
            y_d= [np.float32(val) for val in data]
            y_d=np.asarray(y_d,dtype=np.float32)/1000
            f.close()
        return (y_d)

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
        if is_training == True:
            img_folder=base_file+'/img/'+params["ds_training"]
        else:
            # img_folder = base_file + '/ntu-36m'
            img_folder = base_file + '/img/'+params["ds_test"]
        # img_folder=base_file+'/img350'
    else:
        img_folder=base_file+'/img'
    # img_folder=base_file+'/img'
    joints_file=base_file+'/joints/48_mono'
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
    if params['shufle_data']==1:
        joint_files, img_files=ut.unison_shuffled_copies(joint_files,img_files)
    return (joint_files,img_files)

def load_seq_files(args):
    img_folder,joints_file,actor,sq=args[0],args[1],args[2],args[3]
    joint_list=[]
    img_list=[]
    tmp_folder_img=img_folder+"/"+actor+"/"+sq+"/"
    tmp_folder_joints=joints_file+"/"+actor+"/"+sq+"/"

    if os.path.exists(tmp_folder_joints)==False:
        return (joint_list,img_list)
    if os.path.exists(tmp_folder_img)==False:
        return (joint_list,img_list)

    joint_id_list=os.listdir(tmp_folder_joints)
    img_id_list=os.listdir(tmp_folder_img)

    for j in joint_id_list:
        img=(j.replace('.txt',''))+'.png'
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

def prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,LStateList,LStateList_pre, params, Y, X,R_L_list,F_list,state_reset_counter_lst):
    #index_list= list of ids for sequences..
    #LStateList current states
    #LStateList_pre previus batch states
    #repeat list with in a single sequnce.
    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    new_S=ut.get_zero_state(params)

    if(minibatch_index>0):
        for idx in range(batch_size):
            state_reset_counter=state_reset_counter_lst[idx]
            if state_reset_counter%params['reset_state']==0 and params['reset_state']>0:
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx]=0
            elif(pre_sid[idx]!=curr_sid[idx]):# if sequence changed reset state also
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx]=0
            elif (curr_id_lst[idx]==pre_id_lst[idx]): #if we repeated the value we should repeat state also.
                state_reset_counter_lst[idx]=state_reset_counter-1
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=LStateList_pre[s][0][idx]
                    new_S[s][1][idx]=LStateList_pre[s][1][idx]
                    # new_S[s][idx,:]=LStateList_pre[s][idx,:]
            else:
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=LStateList[s][0][idx]
                    new_S[s][1][idx]=LStateList[s][1][idx]
                    # new_S[s][idx,:]=LStateList[s][idx,:]
    x=X[curr_id_lst]
    y=Y[curr_id_lst]
    if R_L_list != None:
        r=R_L_list[curr_id_lst]
    else:
        r=None
    f=F_list[curr_id_lst]


    return (new_S,x,y,r,f,state_reset_counter_lst)
#
#
# def prepare_lstm_batch_joints(index_list, minibatch_index, batch_size, S_list,LStateList,LStateList_pre, params, F_names,state_reset_counter_lst):
#     curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
#     pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
#     curr_sid=S_list[curr_id_lst]
#     pre_sid=S_list[pre_id_lst]
#     new_S=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=np.float32) for i in range(len(LStateList))]
#     for idx in range(batch_size):
#         state_reset_counter=state_reset_counter_lst[idx]
#         if(minibatch_index==0):
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=LStateList[s][idx,:]
#         elif(pre_sid[idx]!=curr_sid[idx]) or ((state_reset_counter%params['reset_state']==0) and state_reset_counter*params['reset_state']>0):
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
#             state_reset_counter_lst[idx]=0
#         elif (curr_id_lst[idx]==pre_id_lst[idx]): #If value repeated, we should repeat state also
#             state_reset_counter_lst[idx]=state_reset_counter-1
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=LStateList_pre[s][idx,:]
#         else:
#             for s in range(len(new_S)):
#                 new_S[s][idx,:]=LStateList[s][idx,:]
#     flist=F_names[curr_id_lst]
#     x=X[curr_id_lst]
#     y=Y[curr_id_lst]
#
#
#     return (new_S,x,y,state_reset_counter_lst)

def get_action_dataset(params,X,Y,F_list,G_list,S_list,R_L_list):
    action= params["action"]
    seq_id_lst=[]
    for index in range(F_list.shape[0]):
        seq=F_list[index]
        if action in seq[0][0]:
            seq_id_lst.append(index)

    X=X[seq_id_lst]
    Y=Y[seq_id_lst]
    F_list=F_list[seq_id_lst]
    # G_list=np.asarray(G_list)[seq_id_lst].tolist()
    S_list=np.asarray(S_list)[seq_id_lst].tolist()
    R_L_list=R_L_list[seq_id_lst]

    return X,Y,F_list,G_list,S_list,R_L_list

def get_action_seq_indexes(params,S_L,F_list):
    bs=params['batch_size']
    new_S_L=[]
    counter=collections.Counter(S_L)
    lst=[list(t) for t  in counter.items()]
    a=np.asarray(lst)
    ss=a[a[:,1].argsort()][::-1]
    b_index=0
    new_index_lst=dict()
    b_index_lst=dict()

    for item in ss:
        # seq_srt_intex= np.sum(a[0:item[0]-1],axis=0)[1]
        # seq_end_intex= seq_srt_intex+item[1]
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

    for item in ss:
        # seq_srt_intex= np.sum(a[0:item[0]-1],axis=0)[1]
        # seq_end_intex= seq_srt_intex+item[1]
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
    #All bacthes should have same number of sequence.
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

def get_seq_indexes(params,S_L):
    bs=params['batch_size']
    new_S_L=[]
    counter=collections.Counter(S_L)
    lst=[list(t) for t  in counter.items()]
    a=np.asarray(lst)
    ss=a #a[a[:,1].argsort()][::-1] no shorting
    b_index=0
    new_index_lst=dict()
    b_index_lst=dict()

    for item in ss:
        # seq_srt_intex= np.sum(a[0:item[0]-1],axis=0)[1]
        # seq_end_intex= seq_srt_intex+item[1]
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
    #All bacthes should have same number of sequence.
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
