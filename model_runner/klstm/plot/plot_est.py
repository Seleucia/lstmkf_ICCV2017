import matplotlib.pyplot as plt
import numpy as np
import code2.book_plots as bp

base_file="/home/coskun/PycharmProjects/poseft/files/cp_mp/"
# base_file="//home/coskun/PycharmProjects/poseft/files/cp_mp/"
est_file_kfl_QRf=base_file+"0.06221_9_5kfl_QRf_vals.npy"
X_file=base_file+"X_test.npy"
# Y_file=base_file+"Y_test.npy"
Y_file_val=base_file+"0.06221_9_5kfl_QRf_actual_vals.npy"
Y_file=base_file+"Y_test.npy"
index_test_list_file=base_file+"index_test_list.npy"
S_Test_list_file=base_file+"S_Test_list.npy"
R_L_Test_list_file=base_file+"R_L_Test_list.npy"


est_kfl_QRf=np.load(est_file_kfl_QRf)
X_test=np.load(X_file)
# Y_test=np.load(Y_file)
Y_test_val=np.load(Y_file_val)
Y_test=np.load(Y_file)
index_test_list=np.load(index_test_list_file)
S_Test_list=np.load(S_Test_list_file)
R_L_Test_list=np.load(R_L_Test_list_file)
# est_kfl_QRf=est_kfl_QRf.reshape(X_test.shape)

# MeassErr=np.mean(np.sqrt(np.sum(np.square(X_test.reshape((X_test.shape[0]*X_test.shape[1],2))- Y_test.reshape((Y_test.shape[0]*Y_test.shape[1],2))),axis=1)))
# TestErr=np.mean(np.sqrt(np.sum(np.square(est_kfl_QRf.reshape((est_kfl_QRf.shape[0]*est_kfl_QRf.shape[1],2))- Y_test.reshape((Y_test.shape[0]*Y_test.shape[1],2))),axis=1)))
# print "Measurment Error %f"%MeassErr
# print "Test Error %f"%TestErr

muu_kfl_QRf=[]
muu_kfl_QRf.append(est_kfl_QRf[0][0:50])
muu_kfl_QRf.append(est_kfl_QRf[2][100:150])
muu_kfl_QRf.append(est_kfl_QRf[0][250:300])
muu_kfl_QRf.append(est_kfl_QRf[2][350:400])
muu_kfl_QRf=np.asarray(muu_kfl_QRf).reshape(200,2)


groundtruth_idx=[]
groundtruth_idx.append(X_test[0])
groundtruth_idx.append(X_test[1])
groundtruth_idx.append(X_test[2])
groundtruth_idx.append(X_test[3])
groundtruth_idx=np.asarray(groundtruth_idx).reshape(200,2)

measurements_idx=[]
measurements_idx.append(Y_test[0])
measurements_idx.append(Y_test[1])
measurements_idx.append(Y_test[2])
measurements_idx.append(Y_test[3])
# measurements_idx.append(Y_test[4])
# measurements_idx.append(Y_test[5])
# measurements_idx.append(Y_test[6])
# measurements_idx.append(Y_test[7])
# measurements_idx.append(Y_test[8])
# measurements_idx.append(Y_test[9])
# measurements_idx.append(Y_test[10])
# measurements_idx.append(Y_test[11])
# measurements_idx.append(Y_test[12])
measurements_idx=np.asarray(measurements_idx).reshape(len(measurements_idx)*50,2)

print "done...."

# sel_lst=[0,10,20,30]
# sel_lst=[0+10*i for i in range(4)]
# sel_lst=[200]
# seq_id=10
# seq_length=200
# batch_size=10
# fid=(seq_length*seq_id)/(batch_size*seq_length)
# fid2=(seq_length*seq_id)%(batch_size*seq_length)
# muu_kfl_QRf=est_kfl_QRf[fid][fid2:fid2+seq_length]
# groundtruth_idx2=Y_test[fid][fid2:fid2+seq_length]
#
#
# seq_id_lst=index_test_list[sel_lst]
# new_shape=(200,2)
# # muu_kfl_QRf=est_kfl_QRf[seq_id_lst].reshape(new_shape)
# groundtruth_idx=Y_test2[seq_id]
# measurements_idx=X_test[seq_id]


plt.figure()

#plot results
# zs *= .3048 # convert to meters
# ys *= .3048 # convert to meters
# muu /= .3048 # convert to meters
# kalman_mu *= .3048 # convert to meters
# bp.plot_filter(np.asarray(muu)[:, 0], np.asarray(muu)[:, 2])
# bp.plot_measurements(np.asarray(zs)[:, 0], np.asarray(zs)[:, 1])
# bp.plot_measurements(np.asarray(muu_kfl_QRFf)[:, 0], np.asarray(muu_kfl_QRFf)[:, 1], color='#008B8B', label='kfl_QRFf', lw=1)
# bp.plot_measurements(np.asarray(muu_kf_QR)[:, 0], np.asarray(muu_kf_QR)[:, 1], color='#9400D3', label='kf_QR', lw=1)
# bp.plot_measurements(np.asarray(muu_kfl_QRf)[:, 0], np.asarray(muu_kfl_QRf)[:, 1], color='#654321', label='kfl_QRf', lw=1)
# bp.plot_measurements(np.asarray(muu_kfl_f)[:, 0], np.asarray(muu_kfl_f)[:, 1], color='#056608', label='kfl_f', lw=1)
# bp.plot_measurements(np.asarray(muu_lstm)[:, 0], np.asarray(muu_lstm)[:, 1], color='#000000', label='lstm',lw=1)
# if test_kalman==0:
#     bp.plot_measurements(np.asarray(kalman_mu)[:, 0], np.asarray(kalman_mu)[:, 1],color='#FFF600', label='Kalman Filter',lw=1)
#
bp.plot_measurements(np.asarray(groundtruth_idx)[:, 0], np.asarray(groundtruth_idx)[:, 1],color='#CC0000', label='Ground Truth',lw=1)
bp.plot_measurements(np.asarray(measurements_idx)[:, 0], np.asarray(measurements_idx)[:, 1],color='blue', label='Measurements',lw=1)
# plt.legend(loc=2)
xmin=np.min(np.asarray(groundtruth_idx)[:, 0])
xmax=np.max(np.asarray(groundtruth_idx)[:, 0])
# plt.xlim((xmin, xmax));
plt.show()
# ==> [[ 12.]]

# Close the Session when we're done.
