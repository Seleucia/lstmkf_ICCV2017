import matplotlib.pyplot as plt
import numpy as np
import code2.book_plots as bp

kfl_QRf=[0.240565,0.426858,0.634414, 0.929990, 1.300349]
kfl_QRFf=[0.250626,0.457030,0.645655,0.963544, 1.623101]
kfl_K=[0.250669,0.437358,0.661484, 1.064001, 1.241311]
kf_QR=[0.140400,0.246269, 0.365266, 0.590431, 0.812341]
kf=[0.100120,0.180169, 0.201215, 0.300431, 0.6012300]
# lst=[8.092599,9.890555, 10.256124, 11.107804, 13.107804]
meass=[0.25, 0.501300, 0.751859, 1.253584, 1.880721]
t=range(len(meass))

ticks=[0.2,0.4,0.6,1.0,1.5]
fig, ax = plt.subplots()

plt.plot(ticks,kfl_QRf, color='#654321', label='KalmanLSTM', lw=1)
plt.plot(ticks,kf_QR, color='#056608', label='Kalman QR', lw=1)
plt.plot(ticks,kf, color='#FFF600', label='Kalman Filter', lw=1)
plt.plot(ticks,meass, color='red', label='Measurement Error', lw=1)
# bp.plot_measurements(np.asarray(muu_kfl_f)[:, 0], np.asarray(muu_kfl_f)[:, 1], color='#056608', label='kfl_f', lw=1)
# bp.plot_measurements(np.asarray(muu_lstm)[:, 0], np.asarray(muu_lstm)[:, 1], color='#000000', label='lstm',lw=1)
# if test_kalman==0:
#     bp.plot_measurements(np.asarray(kalman_mu)[:, 0], np.asarray(kalman_mu)[:, 1],color='#FFF600', label='Kalman Filter',lw=1)
#
# bp.plot_measurements(np.asarray(groundtruth_idx)[:, 0], np.asarray(groundtruth_idx)[:, 1],color='#CC0000', label='Ground Truth',lw=1)
# bp.plot_measurements(np.asarray(measurements_idx)[:, 0], np.asarray(measurements_idx)[:, 1],color='#0247FE', label='Measurements',lw=1)
# plt.legend(loc=2)
# xmin=np.min(np.asarray(lst))
# xmax=np.max(np.asarray(lst))
# plt.xlim((xmin, xmax));
plt.grid()
ax.set_xticks(ticks)
plt.legend(loc=2, borderaxespad=0.)
plt.show()