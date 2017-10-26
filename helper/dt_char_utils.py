from helper import config
import numpy as np
import time
import urllib
import h5py
import pickle

def save_char_data(params):
    max_count=params["max_count"]
    url="http://www.textfiles.com/etext/FICTION/war_peace_text"
    txt = urllib.urlopen(url).read()
    txt = txt.encode('ascii',errors='ignore')[0:max_count]

    char_len=len(txt)
    char_lst=list(set(txt))
    char_dict={}
    for idx in range(len(char_lst)):
        c=char_lst[idx]
        char_dict[c]=idx


    char_mat = np.zeros((char_len, len(char_lst)), dtype=int)
    for idx in range(len(txt)):
        c=txt[idx]
        chr_idx=char_dict[c]
        char_mat[idx, chr_idx] = 1

    fl=params["data_bin"]+"char-dic"+".p"

    with open(fl, 'wb') as handle:
        pickle.dump(char_dict, handle)

    fl=params["data_bin"]+"char-mat"+".h5"
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('char_mat', data=char_mat)
    # h5f.create_dataset('char_dict', data=char_dict)
    h5f.close()
    return char_mat,char_dict

def load_char_data(params):
    fl=params["data_bin"]

    with h5py.File(fl+"char-mat"+".h5", 'r') as f:
        char_mat=f['char_mat'][()]

    with open(fl+"char-dic"+".p", 'rb') as handle:
        char_dict = pickle.load(handle)

    return char_mat,char_dict

def prepare_char_seq(params,char_mat):
    p_count=params['seq_length']
    X_D=[]
    Y_D=[]
    Y_d=[]
    X_d=[]
    for i in range(char_mat.shape[0]-1):
        X_d.append(char_mat[i])
        Y_d.append(char_mat[i+1])
        if len(Y_d)==p_count and p_count>0:
            X_D.append(X_d)
            Y_D.append(Y_d)
            X_d=[]
            Y_d=[]
    return X_D,Y_D


def prepare_training_set(params):
    # params["max_count"]=500
    if(params["reload_data"]==1):
        start = time.time()
        char_mat,char_dict=save_char_data(params)
        end = time.time()
        tm=end-start
        print "Fresh dataset saved and loaded: %s"%(tm)
    else:
        start = time.time()
        char_mat,char_dict=load_char_data(params)
        end = time.time()
        tm=end-start
        print "Dataset loaded from the loacl: %s"%(tm)

    X_train,Y_train=prepare_char_seq(params,char_mat)
    return X_train,Y_train,char_dict
