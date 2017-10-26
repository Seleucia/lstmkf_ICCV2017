import  numpy as np
import math
from helper import utils as ut
import collections

class Ema(object):
    def __init__(self):
        print "EMA Init..."

    def raw_moving_average(self,params,x, n, type='simple'):
        """
        compute an n period moving average.
        type is 'simple' | 'exponential'
        """

        x = np.asarray(x)
        x_ret=np.zeros(x.shape)
        if type == 'simple':
            weights = np.ones(n)
        else:
            weights = np.exp(np.linspace(-1., 0., n))

        weights /= weights.sum()
        for i in range(x.shape[1]):
            x_ret[:, i] = np.convolve(x[:,i], weights, mode='same')

        return x_ret

    def compute_quat_loss(self, gt, est):
        loss=[]
        for i in range(len(gt)):
            pose_x = gt[i][0:3]
            pose_q = gt[i][3:7]
            predicted_x = est[i][0:3]
            predicted_q = est[i][3:7]
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180 / math.pi
            error_x = np.linalg.norm(pose_x - predicted_x)
            loss.append([error_x, theta])
        return loss


    def run(self,params,db_values_x_test,db_values_y_test,seq_id_names_test=None):
        print "EMA Model is running..."

        i=0
        seq_loss=[]
        seq_meas_loss=[]
        total_loss=[]
        total_meas_loss=[]
        results = []
        last_dim=params['n_output']+1
        seq_id_lst=db_values_x_test[:,0]

        counter = collections.Counter(seq_id_lst)
        n=5
        # db_names_test = params['db_names_test']
        avg_mode=params['avg_mode']
        n=params['n']
        params["est_file"] = "/mnt/Data1/hc/est/"+avg_mode+"_"+str(n)

        for seq_id in counter:
            seq_x=db_values_x_test[db_values_x_test[:, 0] == seq_id][:,1:]
            seq_y=db_values_y_test[db_values_y_test[:, 0] == seq_id][:,1:]
            # fnames=db_names_test[db_values_y_test[:, 0] == seq_id]

            # new_zarr=np.zeros(seq_x.shape)
            # new_arr=seq_x[1:-1,:]
            # dif_arr=new_arr[:,:]-seq_x[0:-2,:]
            # av_diff_seq_x=self.raw_moving_average(params, dif_arr, n, type=avg_mode)
            # new_zarr[1:-1,:]=av_diff_seq_x
            # av_seq_x=new_zarr+seq_x

            av_seq_x=self.raw_moving_average(params, seq_x, n, type=avg_mode)



            val_cnt=len(seq_x)
            if params["data_mode"] == "human":
                avg_est = av_seq_x.reshape((val_cnt,params['n_output'] / 3, 3))
                meas = seq_x.reshape(val_cnt,params['n_output'] / 3, 3)
                gt = seq_y.reshape(val_cnt,params['n_output'] / 3, 3)
                avg_diff_vec = avg_est - gt  # 16*3
                meas_diff_vec = meas - gt  # 16*3
                avg_loss = np.mean(np.sqrt(np.sum(avg_diff_vec ** 2, axis=2)))
                meas_loss = np.mean(np.sqrt(np.sum(meas_diff_vec ** 2, axis=2)))
                seq_loss.append(avg_loss)
                seq_meas_loss.append(meas_loss)
                print "Total Frame Count: %i, Avg Loss: %f, Meas Loss: %f" % (
                    len(db_values_y_test), np.mean(seq_loss), np.mean(seq_meas_loss))
            else:
                ema_est = av_seq_x
                meas = seq_x
                # x_pred = x_pred[0:params['n_output']]
                gt = seq_y
                print ema_est.shape
                loss = self.compute_quat_loss(gt, ema_est)
                # print error_x
                results.extend(loss)
                seq_loss.extend(loss)
                loss = self.compute_quat_loss(gt, meas)
                seq_meas_loss.extend(loss)
                print "Total Frame Count: %i, Median (T,R) Loss: %f, %f, Median Meas Loss (T,R): %f, %f" % (
                    len(av_seq_x), np.median(seq_loss,axis=0)[0], np.median(seq_loss,axis=0)[1]
                    , np.median(seq_meas_loss,axis=0)[0], np.median(seq_meas_loss,axis=0)[1])
                seq_loss=[]
                seq_meas_loss=[]


                # ut.write_est(est_file=params["est_file"], est=av_seq_x, file_names=fnames)

        if params["data_mode"] == "human":
            print "Total Frame Count: %i, Avg Loss: %f, Meas Loss: %f" % (
                len(db_values_y_test), np.mean(seq_loss), np.mean(seq_meas_loss))
        else:
            print "Total Frame Count: %i, Median Loss (T,R): %f,%f, Median Meas Loss (T,R): %f, %f" % (
                len(db_values_y_test), np.median(results,axis=0)[0],np.median(results,axis=0)[1],
                np.median(seq_meas_loss,axis=0)[0],np.median(seq_meas_loss,axis=0)[1])



