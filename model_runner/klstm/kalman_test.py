import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

import code2.book_plots as bp
from helper.PosSensor1 import PosSensor2
from kalman_filter import KalmanFilter

tracker = KalmanFilter(dim_x=4, dim_z=2)

R_std = 0.35
Q_std = 0.04

def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    tracker.H = np.array([[1/0.3048, 0, 0, 0],
                          [0, 0, 1/0.3048, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker

plt.figure()
# simulate robot movement
N = 30
sensor = PosSensor2((0, 0), (2, .2), noise_std=R_std)

results=[sensor.read() for _ in range(N)]
# zs = np.array([np.array([sensor.read()]).T for _ in range(N)])
zs=np.expand_dims(np.asarray([list(item) for item in results])[:,0,:],2)
ys=np.asarray([list(item) for item in results])[:,1,:]
# run filter
robot_tracker = tracker1()
mu, cov, _, _ = robot_tracker.batch_filter(zs)

for x, P in zip(mu, cov):
    # covariance of x and y
    cov = np.array([[P[0, 0], P[2, 0]],
                    [P[0, 2], P[2, 2]]])
    mean = (x[0, 0], x[2, 0])
    # plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)

#plot results
zs *= .3048 # convert to meters
bp.plot_filter(mu[:, 0], mu[:, 2])
bp.plot_measurements(zs[:, 0], zs[:, 1])
# bp.plot_measurements(ys[:, 0], ys[:, 1],c='green',label='Ground Truth')
plt.legend(loc=2)
plt.xlim((0, N));
plt.show()