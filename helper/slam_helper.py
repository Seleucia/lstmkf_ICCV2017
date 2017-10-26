import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool
import utils as ut
import collections
import math
import dt_utils as dut

def load_file(fl):
    if len(fl)>10:
        yfl=fl
        with open(yfl, "rb") as f:
            data=f.read().strip().split(' ')
            y_d= [np.float32(val) for val in data]
            y_d=np.asarray(y_d,dtype=np.float32)
            f.close()
        return (y_d)

    yfl=fl[0]
    xfl=fl[1]
    with open(yfl, "rb") as f:
        data=f.read().strip().split(' ')
        y_d= [np.float32(val) for val in data]
        y_d=np.asarray(y_d,dtype=np.float32)
        f.close()

    with open(xfl, "rb") as f:
        data=f.read().strip().split(' ')
        x_d= [np.float32(val) for val in data]
        x_d=np.asarray(x_d,dtype=np.float32)
        f.close()
    return (y_d,x_d)


def load_kalman_data(params):
    if params["data_mode"]=="human":
        _, _, _, db_values_x_test, db_values_y_test, _ = \
            dut.load_from_bin_db(params)
    else:
        (db_values_x_test,db_values_y_test,_,seq_id_names_test)=load_dataset(params,is_training=True)

    return db_values_x_test, db_values_y_test,seq_id_names_test


def load_estimation(params):
    est_folder=params["est_file"]

    lst_sq=os.listdir(est_folder)
    lst_sq.sort()
    seq_y=[]
    seq_x=[]
    db_names=[]
    seq_id_names={}
    seq_id=0
    for sq in lst_sq:
        est_id_lst=os.listdir(est_folder+sq)
        # print common_lst
        if '.txt' in est_id_lst[0]:
                common_id_lst=sorted([int(f.replace('.txt','')) for f in est_id_lst])
                est_list=[est_folder+sq+'/'+str(p1)+'.txt'  for p1 in common_id_lst]
        else:
            common_id_lst=sorted([int(f) for f in est_id_lst])
            est_list=[est_folder+sq+'/'+str(p1)  for p1 in common_id_lst]

        f_list=est_list

        pool = ThreadPool(1000)
        results = pool.map(load_file, f_list)
        pool.close()
        for r in range(len(results)):
            seq_y.append(np.hstack((seq_id,results[r])))
        db_names.extend(f_list)
        seq_id_names[seq_id]=sq
        seq_id+=1

    return (np.asarray(seq_y),db_names,seq_id_names)

        # args=zip(cycle([img_folder]),cycle([joints_file]),cycle([actor]),lst_sq)
        # results=pool.map(load_seq_files, args)
        # joint_flat_lst=[res[0] for res in results]
        # img_flat_lst=[res[1] for res in results]
        # joint_flat_lst=[item for sublist in joint_flat_lst for item in sublist]
        # img_flat_lst=[item for sublist in img_flat_lst for item in sublist]

def load_dataset(params,is_training=True):
    if is_training==True:
        if params["train_mode"]=="partial":
            data_dir=params["data_dir"]+params["train_mode"]+"/"+params["sequence"]+"/"
            gt_folder=data_dir+"training/"+"gt/"
            est_folder=data_dir+"training/"+"est/"
        else:
            if params["sequence"] in params["cambridge"]:
                data_dir=params["data_dir"]+params["train_mode"]+"/cambridge/"
            else:
                data_dir=params["data_dir"]+params["train_mode"]+"/7scene/"

            gt_folder=data_dir+"training/"+"gt/"
            est_folder=data_dir+"training/"+"est/"

    else:
        data_dir=params["data_dir"]+"partial/"+params["sequence"]+"/"
        gt_folder=data_dir+"test/"+"gt/"
        est_folder=data_dir+"test/"+"est/"
    print est_folder

    lst_sq=os.listdir(gt_folder)
    lst_sq.sort()
    seq_y=[]
    seq_x=[]
    db_names=[]
    seq_id_names={}
    seq_id=0
    for sq in lst_sq:
        gt_id_lst=os.listdir(gt_folder+sq)
        est_id_lst=os.listdir(est_folder+sq)
        common_lst=[id for id in est_id_lst if id in gt_id_lst]
        # print common_lst
        if params["train_mode"]=="full":
            if '.txt' in common_lst[0]:
                common_id_lst=sorted([int(f.replace('.txt','')) for f in common_lst])
                gt_list=[gt_folder+sq+'/'+str(p1)+'.txt'  for p1 in common_id_lst]
                est_list=[est_folder+sq+'/'+str(p1)+'.txt'  for p1 in common_id_lst]
            else:
                common_id_lst=sorted([int(f) for f in common_lst])
                gt_list=[gt_folder+sq+'/'+str(p1)  for p1 in common_id_lst]
                est_list=[est_folder+sq+'/'+str(p1)  for p1 in common_id_lst]
        else:
            if params["sequence"] in ["ShopFacade",'OldHospital','StMarysChurch','chess','fire','heads','office','David','redkitchen','stairs','pumpkin']:
                common_id_lst=sorted([int(f) for f in common_lst])
                gt_list=[gt_folder+sq+'/'+str(p1)  for p1 in common_id_lst]
                est_list=[est_folder+sq+'/'+str(p1)  for p1 in common_id_lst]
            elif params["sequence"]=="Street":

                common_id_lst=sorted([int(f.replace('.txt','')) for f in common_lst])
                gt_list=[gt_folder+sq+'/'+str(p1)+'.txt'  for p1 in common_id_lst]
                est_list=[est_folder+sq+'/'+str(p1)+'.txt'  for p1 in common_id_lst]

            else:
                common_id_lst=sorted([int(f[0:-4]) for f in common_lst])
                gt_list=[gt_folder+sq+'/'+str(p1)+".txt"  for p1 in common_id_lst]
                est_list=[est_folder+sq+'/'+str(p1)+".txt"  for p1 in common_id_lst]

        f_list=zip(gt_list,est_list)

        pool = ThreadPool(1000)
        results = pool.map(load_file, f_list)
        pool.close()
        for r in range(len(results)):
            if params["data_mode"]=="xyx":
                seq_y.append(np.hstack((seq_id,results[r][0][0:3])))
                seq_x.append(np.hstack((seq_id,results[r][1][0:3])))
            elif params["data_mode"]=="q":
                seq_y.append(np.hstack((seq_id,results[r][0][3:7])))
                seq_x.append(np.hstack((seq_id,results[r][1][3:7])))
            else:
                seq_y.append(np.hstack((seq_id,results[r][0])))
                seq_x.append(np.hstack((seq_id,results[r][1])))
        db_names.extend(f_list)
        seq_id_names[seq_id]=sq
        seq_id+=1

    return (np.asarray(seq_x),np.asarray(seq_y),db_names,seq_id_names)

        # args=zip(cycle([img_folder]),cycle([joints_file]),cycle([actor]),lst_sq)
        # results=pool.map(load_seq_files, args)
        # joint_flat_lst=[res[0] for res in results]
        # img_flat_lst=[res[1] for res in results]
        # joint_flat_lst=[item for sublist in joint_flat_lst for item in sublist]
        # img_flat_lst=[item for sublist in img_flat_lst for item in sublist]

def load_db(params):
    (db_values_x_training,db_values_y_training,db_names_training,seq_id_names_training)=load_dataset(params,is_training=True)
    (db_values_x_test,db_values_y_test,db_names_test,seq_id_names_test)=load_dataset(params,is_training=False)
    return (db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test)

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

def load_flat_data(params):
    db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test=load_db(params)
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

    if params["subsample"]>0:
        print("Subsampling frames %s"%params["subsample"])
        db_values_x_training,db_values_y_training,db_names_training,seq_rel_train=subsample_frames(params,db_values_x_training,db_values_y_training,db_names_training)
        db_values_x_test,db_values_y_test,db_names_test,seq_rel_test=subsample_frames(params,db_values_x_test,db_values_y_test,db_names_test)
        params["seq_rel_train"]=seq_rel_train
        params["seq_rel_test"]=seq_rel_test
    return params,db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test

def prepare_training_set(params):
    db_values_x_training,db_values_y_training,db_names_training,db_values_x_test,db_values_y_test,db_names_test=load_db(params)
    if params["normalise_data"]==3: #We normalise the target and input values.
        print("Normalasing the input and target values")
        db_values_x_training,db_values_x_test,men,std=ut.normalise_data(db_values_x_training, db_values_x_test)
        params["x_men"]=men
        params["x_std"]=std
        db_values_y_training,db_values_y_test,men,std=ut.normalise_data(db_values_y_training, db_values_y_test)
        params["y_men"]=men
        params["y_std"]=std

    if params["subsample"]>0:
        print("Subsampling frames %s"%params["subsample"])
        db_values_x_training,db_values_y_training,db_names_training,seq_rel_train=subsample_frames(params,db_values_x_training,db_values_y_training,db_names_training)
        db_values_x_test,db_values_y_test,db_names_test,seq_rel_test=subsample_frames(params,db_values_x_test,db_values_y_test,db_names_test)
        params["seq_rel_train"]=seq_rel_train
        params["seq_rel_test"]=seq_rel_test

    X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list=prepare_sequences(params,db_values_x_training,db_values_y_training,db_names_training)
    X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list=prepare_sequences(params,db_values_x_test,db_values_y_test,db_names_test)

    return (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list,X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)


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

def get_single_seq_loss(gt,est):
    loss=[]
    for i in range(len(gt)):
        pose_q=gt[i][3:7]
        predicted_q=est[i][3:7]
        pose_x=gt[i][0:3]
        predicted_x=est[i][0:3]
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        if d<=-1:
            d=-1+0.00001
        if d>=1:
            d=1-0.00001
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(pose_x-predicted_x)
        loss.append([error_x,theta])
    return loss

def get_loss(flist,gt,est):
    dict_err={}
    flist=[a.split('/')[-2] for a in flist[:,1].tolist()]
    for f in flist:
        dict_err[f]=[]
    i=0
    theta_lst=[]
    error_x_lst=[]
    loss=[]
    for i in range(len(gt)):
            f=flist[i]
            pose_q=gt[i][3:7]
            predicted_q=est[i][3:7]
            pose_x=gt[i][0:3]
            predicted_x=est[i][0:3]
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1,q2)))
            if d<=-1:
                d=-1+0.00001
            if d>=1:
                d=1-0.00001
            theta = 2 * np.arccos(d) * 180/math.pi
            error_x = np.linalg.norm(pose_x-predicted_x)
            error_x_lst.append(error_x)
            theta_lst.append(theta)
            loss.append([error_x,theta])
            perr=dict_err[f]
            perr.append([error_x,theta])
            dict_err[f]=perr


    # else:
    #     for rr in r:
    #         if rr==1:
    #             cid=curr_sid[cnt/params['seq_length']]
    #             if params["data_mode"]=="xyx":
    #                 pose_x=gt[i]
    #                 predicted_x=est[i]
    #                 error_x = np.linalg.norm(pose_x-predicted_x)
    #                 error_x_lst.append(error_x)
    #                 err=dict_err[cid]
    #                 err.append([error_x])
    #             elif params["data_mode"]=="q":
    #                 pose_q=gt[i]
    #                 predicted_q=est[i]
    #                 q1 = pose_q / np.linalg.norm(pose_q)
    #                 q2 = predicted_q / np.linalg.norm(predicted_q)
    #                 d = abs(np.sum(np.multiply(q1,q2)))
    #                 theta = 2 * np.arccos(d) * 180/math.pi
    #                 theta_lst.append(theta)
    #                 err=dict_err[cid]
    #                 err.append([theta])
    #
    #             else:
    #                 pose_q=gt[i][3:7]
    #                 predicted_q=est[i][3:7]
    #                 pose_x=gt[i][0:3]
    #                 predicted_x=est[i][0:3]
    #                 q1 = pose_q / np.linalg.norm(pose_q)
    #                 q2 = predicted_q / np.linalg.norm(predicted_q)
    #                 d = abs(np.sum(np.multiply(q1,q2)))
    #                 if d<=-1:
    #                     d=-1+0.00001
    #                 if d>=1:
    #                     d=1-0.00001
    #                 theta = 2 * np.arccos(d) * 180/math.pi
    #                 error_x = np.linalg.norm(pose_x-predicted_x)
    #                 error_x_lst.append(error_x)
    #                 theta_lst.append(theta)
    #                 err=dict_err[cid]
    #                 err.append([error_x,theta])
    #                 loss.append([error_x,theta])
    #             dict_err[cid]=err
    #
    #             i+=1
    #         cnt+=1
    return loss,dict_err,
    # print 'Iteration:  ', 0, '  Error XYZ (m):  ', np.mean(error_x_lst), '  Error Q (degrees):  ', np.mean(theta_lst)

def get_nondublicate_lst(lst):
    index_lst=[True for i in range(len(lst))]
    seen=[]
    cnt=0
    for item in lst:
        if item not in seen:
            seen.append(item)
            index_lst[cnt]=True
        else:
            index_lst[cnt]=False
        cnt+=1
    return index_lst



