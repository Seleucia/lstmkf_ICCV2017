import os
import math
import numpy as np
import shutil


def mat2quat(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q

def compute_lost(gt,est):
    loss=[]
    for i in range(len(gt)):
        pose_q=gt[i][3:7]
        predicted_q=est[i][3:7]
        pose_x=gt[i][0:3]
        predicted_x=est[i][0:3]
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(pose_x-predicted_x)
        loss.append([error_x,theta])
    return loss

seq_lst=['kinect_box','milk','orange_juice','tide']
base_path='/home/coskun/PycharmProjects/PoseNet/David/'
test=['kinect_box','tide']

if os.path.exists(base_path+'test/'):
    shutil.rmtree(base_path+'test/')
if os.path.exists(base_path+'training/'):
    shutil.rmtree(base_path+'training/')

for s in seq_lst:
    est_ffile=base_path+'raw_data/'+s+'_est.txt'
    gt_ffile=base_path+'/raw_data/'+s+'_gt.txt'

    if s in test:
        gt_base_folder=base_path+'test/gt/'+s+'/'
        est_base_folder=base_path+'test/est/'+s+'/'
    else:
        gt_base_folder=base_path+'training/gt/'+s+'/'
        est_base_folder=base_path+'training/est/'+s+'/'

    os.makedirs(est_base_folder)
    os.makedirs(gt_base_folder)
    est_vect=[]
    gt_vect=[]

    with open(gt_ffile) as f:
        lines=f.readlines()
        i=1
        cnt=1
        fline=""
        for line in lines:
            gt_file=gt_base_folder+str(i)
            cnt+=1
            fline=fline+line
            if (cnt==4):
                i+=1
                # fline=fline.replace('\t',' ').replace('\r',' ').replace('\n','').strip()
                homo_mat=np.asarray([[float(ll.replace('\r','')) for ll in l.split('\t')] for l in fline.split('\n') if len(l)>0],dtype=np.float32)
                xyz=homo_mat[:,3]
                rot_mat=homo_mat[:,:3]
                quat=mat2quat(rot_mat)
                xyzwpqr=np.hstack((xyz,quat)).tolist()
                gt_vect.append(xyzwpqr)
                str_xyzwpqr=' '.join([str(val) for val in xyzwpqr])
                # gt_vect.append([float(f) for f in fline.split(' ')])
                with open(gt_file, "a") as p:
                    p.write(str_xyzwpqr)
                    p.close()
                    fline=""
                    cnt=1

    with open(est_ffile) as f:
        lines=f.readlines()
        i=1
        cnt=1
        fline=""
        for line in lines:
            est_file=est_base_folder+str(i)
            cnt+=1
            fline=fline+line
            if (cnt==4):
                i+=1
                homo_mat=np.asarray([[float(ll.replace('\r','')) for ll in l.split('\t')] for l in fline.split('\n') if len(l)>0],dtype=np.float32)
                xyz=homo_mat[:,3]
                rot_mat=homo_mat[:,:3]
                quat=mat2quat(rot_mat)
                xyzwpqr=np.hstack((xyz,quat)).tolist()
                est_vect.append(xyzwpqr)
                str_xyzwpqr=' '.join([str(val) for val in xyzwpqr])
                with open(est_file, "a") as p:
                    p.write(str_xyzwpqr)
                    p.close()
                    fline=""
                    cnt=1
    # loss=[]
    # est_vect=np.asarray(est_vect)
    # gt_vect=np.asarray(gt_vect)
    # for est, gt in zip(est_vect,gt_vect):
    #     x_est=est[0:3]*1000
    #     x_gt=gt[0:3]*1000
    #     y_est=est[4:7]*1000
    #     y_gt=gt[4:7]*1000
    #     z_est=est[8:11]*1000
    #     z_gt=gt[8:11]*1000
    #     r_est=est[3]
    #     r_gt=gt[3]
    #     p_est=est[7]
    #     p_gt=gt[7]
    #     yaw_est=est[11]
    #     yaw_gt=gt[11]
    #     x_loss=np.linalg.norm(x_est-x_gt)/3
    #     y_loss=np.linalg.norm(y_est-y_gt)/3
    #     z_loss=np.linalg.norm(z_est-z_gt)/3
    #     r_loss=np.linalg.norm(r_est-r_gt)
    #     p_loss=np.linalg.norm(p_est-p_gt)
    #     yaw_loss=np.linalg.norm(yaw_est-yaw_gt)
    #     ls=[x_loss,y_loss,z_loss,r_loss,p_loss,yaw_loss]
    #     # print ls
    #
    #     loss.append(ls)
    full_err=compute_lost(np.asarray(gt_vect),np.asarray(est_vect))
    median_result= np.median(full_err,axis=0)
    mean_result= np.mean(full_err,axis=0)
    print 'Full sequence median/mean error ',s, median_result[0],'/', mean_result[0], 'm  and ', median_result[1],'/', mean_result[1], 'degrees.'

    # mean_result=np.mean(loss,axis=0)
    # print 'Sequence: %s, Error: x: %f, y: %f, z: %f, Roll: %f, Pitch: %f, Yaw: %f, '%(s,mean_result[0],mean_result[1],
    #                                                                                   mean_result[2],mean_result[3],
    #                                                                                   mean_result[4],mean_result[5])

