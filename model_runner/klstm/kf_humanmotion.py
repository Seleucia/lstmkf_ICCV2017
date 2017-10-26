import numpy as np
import scipy.linalg as linalg
from scipy.linalg import expm
import math

class kftf(object):
    def __init__(self):
        print "Init..."

    def predict(self,Q):
        # x = Fx + Bu
        self._x = np.dot(self.F,self._x)

        # P = FPF' + Q
        self._P = np.dot(self.F,np.dot(self._P,self.F.T)) + Q

    def update(self,z,R):

        # rename for readability and a tiny extra bit of speed
        # H = self.H
        P = self._P
        x = self._x


        # y = z - Hx
        # error (residual) between measurement and prediction
        # Hx = np.dot(H, x)
        Hx = x

        self._y = z - Hx

        # S = HPH' + R
        # project system uncertainty into measurement space
        # S = np.dot(H, np.dot(P, H.T)) + R
        S = P + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        # K = np.dot(P, np.dot(H.T, linalg.inv(S)))
        K = np.dot(P, linalg.inv(S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self._x = x + np.dot(K, self._y)

        # P = (I-KH)P(I-KH)' + KRK'
        # I_KH = self._I - np.dot(K, H)
        I_KH = self._I - K
        self._P = np.dot(I_KH, np.dot(P, I_KH.T)) + np.dot(K, np.dot(R, K.T))

        self._S = S
        self._K = K

    def update_prediction_matrices(self, ztm1,zt):
        #ztm1 previus measurement
        #zt current measurement
        self.process_model=zt-ztm1

    def van_loan_discretization(self,F, G, dt):

        """ Discretizes a linear differential equation which includes white noise
        according to the method of C. F. van Loan [1]. Given the continuous
        model

            x' =  Fx + Gu

        where u is the unity white noise, we compute and return the sigma and Q_k
        that discretizes that equation.


        Examples
        --------

            Given y'' + y = 2u(t), we create the continuous state model of

            x' = [ 0 1] * x + [0]*u(t)
                 [-1 0]       [2]

            and a time step of 0.1:


            >>> F = np.array([[0,1],[-1,0]], dtype=float)
            >>> G = np.array([[0.],[2.]])
            >>> phi, Q = van_loan_discretization(F, G, 0.1)

            >>> phi
            array([[ 0.99500417,  0.09983342],
                   [-0.09983342,  0.99500417]])

            >>> Q
            array([[ 0.00133067,  0.01993342],
                   [ 0.01993342,  0.39866933]])

            (example taken from Brown[2])


        References
        ----------

        [1] C. F. van Loan. "Computing Integrals Involving the Matrix Exponential."
            IEEE Trans. Automomatic Control, AC-23 (3): 395-404 (June 1978)

        [2] Robert Grover Brown. "Introduction to Random Signals and Applied
            Kalman Filtering." Forth edition. John Wiley & Sons. p. 126-7. (2012)
        """


        n = F.shape[0]

        A = np.zeros((2*n, 2*n),dtype=np.float32)

        # we assume u(t) is unity, and require that G incorporate the scaling term
        # for the noise. Hence W = 1, and GWG' reduces to GG"

        A[0:n,     0:n] = -F.dot(dt)
        A[0:n,   n:2*n] = G.dot(G.T).dot(dt)
        A[n:2*n, n:2*n] = F.T.dot(dt)

        B=expm(A)

        sigma = B[n:2*n, n:2*n].T

        Q = sigma.dot(B[0:n, n:2*n])

        return (sigma, Q)




    def run(self,params,db_values_x_test,db_values_y_test,seq_id_names_test):
        print "Kalman Velocity Model is running..."

        i=0
        p_mult=100.0
        R_mult=0.001
        Q_mult=0.002
        pre_seq=0
        seq_counter=0
        seq_pred_loss=[]
        seq_loss=[]
        seq_meas_loss=[]
        total_loss=[]
        total_meas_loss=[]
        results = []
        last_dim=params['n_output']+1

        for i in range(len(db_values_x_test)):
            z_seq=db_values_x_test[i]
            y_seq=db_values_y_test[i][1:last_dim]
            # print db_values_y_test.shape
            # print y_seq.shape




            seq_id=z_seq[0]

            z=z_seq[1:last_dim]
            if i==0:
                pre_seq=seq_id
                dim_x = z.shape[0]*2
                self._x =  np.hstack((z, (z-z)))
                self.F=np.identity(dim_x)
                for i in range(dim_x/2-1):
                    self.F[i, dim_x/2 + i] = 1
                G=np.identity(dim_x)
                self._I=np.identity(dim_x)
                self._P = np.identity(dim_x)*p_mult  # uncertainty covariance
                (sigma, Q)=self.van_loan_discretization(self.F, G, 1)
                (sigma, R)=self.van_loan_discretization(self.F, G, 1)
                Q=Q*Q_mult
                R=R*R_mult



            if pre_seq!=seq_id: #reset values for new sequence
                self.F = np.identity(dim_x)
                # for i in range(dim_x / 2 - 1):
                #     self.F[i, dim_x / 2 + i] = 1
                # (sigma, Q) = kl.van_loan_discretization(self.F, G, 1)
                # (sigma, R) = kl.van_loan_discretization(self.F, G, 1)
                # self.process_model = np.zeros((dim_x, 1))  # state
                self._x = np.hstack((z, (z - z)))
                self._P = np.identity(dim_x)*p_mult  # uncertainty covariance
                if params["data_mode"]=="human":
                    print "Sequence: id: %i , Frame Count: %i, Kalman Seq Loss: %f,Kalman Seq Pred Loss: %f, Meas Seq Loss: %f, Kalman Loss: %f, Meas Loss: %f" %\
                          (pre_seq,len(seq_loss),np.mean(seq_loss),np.mean(seq_pred_loss),np.mean(seq_meas_loss),np.mean(total_loss),np.mean(total_meas_loss))
                else:
                    # self.print_slam_lost(pre_seq, seq_id_names_test, seq_loss)
                    self.print_slam_lost('Kalman Losss:',pre_seq, seq_id_names_test, seq_loss)
                    self.print_slam_lost('Meass Losss:',pre_seq, seq_id_names_test, seq_meas_loss)

                seq_loss=[]
                seq_pred_loss=[]
                seq_meas_loss=[]
                pre_seq = seq_id
                seq_counter=0
                # Q = Q * 0.65852
                # R = R * 0.065852

            if seq_counter>1:
                ztm1=db_values_x_test[i-1][1:last_dim]
            else:
                ztm1=z

            z_state=np.hstack((z, (z-ztm1)))


            self.predict(Q=Q)
            # means_p[i,:]         = self._x
            # covariances_p[i,:,:] = self.P
            x_pred=self._x

            self.update(z_state, R=R)
            # means[i,:]         = self._x
            # covariances[i,:,:] = self.P

            val_cnt=dim_x/2

            if params["data_mode"]=="human":
                kalman_est = self._x[0:params['n_output']].reshape(val_cnt / 3, 3)
                meas = z.reshape(val_cnt / 3, 3)
                x_pred = x_pred[0:params['n_output']].reshape(val_cnt / 3, 3)
                gt = y_seq.reshape(val_cnt / 3, 3)

                kalman_diff_vec = kalman_est - gt  # 16*3
                meas_diff_vec = meas - gt  # 16*3
                pred_diff_vec = x_pred - gt  # 16*3
                # diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
                kalman_loss = np.mean(np.sqrt(np.sum(kalman_diff_vec ** 2, axis=1)))
                meas_loss = np.mean(np.sqrt(np.sum(meas_diff_vec ** 2, axis=1)))
                pred_loss = np.mean(np.sqrt(np.sum(pred_diff_vec ** 2, axis=1)))
                seq_pred_loss.append(pred_loss)
                seq_loss.append(kalman_loss)
                seq_meas_loss.append(meas_loss)
            else:
                kalman_est = self._x[0:params['n_output']]
                meas = z
                x_pred = x_pred[0:params['n_output']]
                gt = y_seq
                error_x, theta = self.compute_quat_loss(gt, kalman_est)
                results.append([error_x,theta])
                seq_loss.append([error_x,theta])
                error_x, theta = self.compute_quat_loss(gt, meas)
                seq_meas_loss.append([error_x, theta])



            if params["data_mode"]=="human":
                total_loss.append(kalman_loss)
                total_meas_loss.append(meas_loss)
            i+=1
            seq_counter+=1
        if params["data_mode"]=="human":
            print "Total Frame Count: %i, Kalman Loss: %f, Meas Loss: %f" % (len(db_values_x_test), np.mean(total_loss), np.mean(total_meas_loss))
        else:
            self.print_slam_lost('Kalman Losss:',seq_id, seq_id_names_test, seq_loss)
            self.print_slam_lost('Meass Losss:',seq_id, seq_id_names_test, seq_meas_loss)
            median_result = np.median(results,axis=0)
            print 'Final: Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'
            mean_result = np.mean(results,axis=0)
            print 'Final: Mean error ', mean_result[0], 'm  and ', mean_result[1], 'degrees.'

    def compute_quat_loss(self, gt, est):
        pose_x = gt[0:3]
        pose_q = gt[3:7]
        predicted_x = est[0:3]
        predicted_q = est[3:7]
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2 * np.arccos(d) * 180 / math.pi
        error_x = np.linalg.norm(pose_x - predicted_x)
        return error_x, theta

    def print_slam_lost(self,pre_text, seq_id, seq_id_names_test, seq_loss):
        median_result = np.median(seq_loss, axis=0)
        # print results
        # print median_result
        print pre_text,' Sequence:', seq_id_names_test[int(seq_id)], ' Median error ', median_result[0], 'm  and ', median_result[
            1], 'degrees.'
        mean_result = np.mean(seq_loss, axis=0)
        print pre_text,' Sequence:', seq_id_names_test[int(seq_id)], ' Mean error ', mean_result[0], 'm  and ', mean_result[
            1], 'degrees.'


