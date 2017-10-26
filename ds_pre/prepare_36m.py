import numpy as np
import os
import time
import shutil


ind = [1, 2 ,3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]
idx_lst=[i-1 for i in  ind]

<<<<<<< HEAD
uni_path='/mnt/2tb/datasets/36m/joints/96_uni'
path_51_uni='/mnt/2tb/datasets/36m/joints/51_uni'
path_48_root='/mnt/2tb/datasets/36m/joints/48_root'
=======
uni_path='/mnt/2tb/datasets/36m/joints/96_mono'
path_51_uni='/mnt/2tb/datasets/36m/joints/51_mono'
path_48_root='/mnt/2tb/datasets/36m/joints/48_mono'
>>>>>>> 85e3e7d1ce1bbe7c5b68b49dc0247cee5d1f141b


def load_file_nodiv(fl):
    with open(fl, "rb") as f:
        data=f.read().strip().split('\\')
        y_d= [np.float32(val) for val in data[0].split('\n')]
        y_d=np.asarray(y_d,dtype=np.float32)
        f.close()
        return y_d



def prepare_db_flnames():
    # base_file=params['est_file']
    base_file=uni_path
    lst_act=os.listdir(base_file)
    for actor in lst_act:
        tmp_folder=base_file+'/'+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        cnt=0
        for sq in lst_sq:
            joint_tmp_folder=uni_path+'/'+actor+"/"+sq+"/"
            wr51_joint_tmp_folder=path_51_uni+'/'+actor+"/"+sq+"/"
            wr48_joint_tmp_folder=path_48_root+'/'+actor+"/"+sq+"/"

            if os.path.exists(joint_tmp_folder)==False:
                continue
            if os.path.exists(wr51_joint_tmp_folder)==False:
                os.makedirs(wr51_joint_tmp_folder)
            else:
                shutil.rmtree(wr51_joint_tmp_folder)
                os.makedirs(wr51_joint_tmp_folder)

            if os.path.exists(wr48_joint_tmp_folder)==False:
                os.makedirs(wr48_joint_tmp_folder)
            else:
                shutil.rmtree(wr48_joint_tmp_folder)
                os.makedirs(wr48_joint_tmp_folder)

            joint_id_list=os.listdir(joint_tmp_folder)
            for j in joint_id_list:
                jpath=uni_path+'/'+actor+"/"+sq+"/"+j
                wr_jpath_51=path_51_uni+'/'+actor+"/"+sq+"/"+j
                wr_jpath_48=path_48_root+'/'+actor+"/"+sq+"/"+j
                jmat=load_file_nodiv(jpath)
                np_j=np.asarray(jmat).reshape((32,3))
                j51_sel_mat=np_j[idx_lst]
                j48_sel_mat=np.asarray([(j-j51_sel_mat[0])/1000.0 for  j in j51_sel_mat[1:]]).flatten().tolist()

                j51_sel_mat=j51_sel_mat.flatten().tolist()
                j51_sel_mat = [str(f) for f in j51_sel_mat]
                j48_sel_mat = [str(f) for f in j48_sel_mat]

                with open(wr_jpath_51, 'w') as fout:
                    fout.write(" ".join(j51_sel_mat))

                with open(wr_jpath_48, 'w') as fout:
                    fout.write(" ".join(j48_sel_mat))


                # np.savetxt(wr_jpath_48,j48_sel_mat.T,fmt='%1.8f',delimiter=' ')


prepare_db_flnames()