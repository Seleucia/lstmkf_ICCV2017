#!/usr/bin/python
import sys
import cv

import random
import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
import model_runner.klstm.code2.book_plots as bp
from numpy import mat,eye,zeros,array,diag,average

import model_runner.klstm.bc.drawing
import model_runner.klstm.bc.kalcommon as kc
from model_runner.klstm.bc.brownian import brownian
import matplotlib.pyplot as plt

total_seconds = 18
total_frames = total_seconds*kc.fps
center_coords = (kc.img_size[0]/2, kc.img_size[1]/2)
# center_coords=[640,360]
# ball_motion = kc.get_simple_ball_motion(center_coords, total_seconds)
# print ball_motion
center_coords=[6,3]
ball_motion = kc.get_simple_ball_motion(center_coords, total_seconds)
print ball_motion
ball_motion = kc.get_brownian_ball_motion(center_coords, total_seconds)
print ball_motion
cam1_estimates = []
cam2_estimates = []
closest_camera_estimates = []
kalman_estimates = []
average_estimates = []
d = model_runner.klstm.bc.drawing.Drawing()
# do the kalman filtering:
A_x = A_y = 0           # velocity of ball (calculated at each time step)
xhat_x = xhat_y = 0
P_x = P_y = 0
xhatminus_x = xhatminus_y = 0
Pminus_x = Pminus_y = 0
K_x = K_y = 0

xhat0_x = xhat0_y = 300
P0_x = P0_y = 1

xhatprev_x = xhat0_x
Pprev_x = P0_x
xhatprev_y = xhat0_y
Pprev_y = P0_y
F=np.eye(4)
F[1,0]=1
F[3,2]=1
for cnt,c in enumerate(ball_motion):
    (cam1_estimate,cam1_sigma) = kc.get_cam_estimate(c, d.cam1center)
    (cam2_estimate,cam2_sigma) = kc.get_cam_estimate(c, d.cam2center)
    print cam1_sigma

    # xhatminus=[xhatprev_x,v_x,xhatprev_y,v_y]
    # xhatminus=np.dot(xhatminus,F)


    # "PREDICT" (time update)
    xhatminus_x = xhatprev_x + (A_x)
    xhatminus_y = xhatprev_y + (A_y)

    Pminus_x = Pprev_x + kc.process_uncertainty
    Pminus_y = Pprev_y + kc.process_uncertainty

    # z = mat([cam1_estimate, cam2_estimate, cam3_estimate])
    (z_x1,z_y1) = cam1_estimate
    (z_x2,z_y2) = cam2_estimate

    # Measurement noise covariance matrix:
    R = pow(cam1_sigma,2) + pow(cam2_sigma,2)
    z_x = (pow(cam2_sigma,2)/(R))*z_x1 + (pow(cam1_sigma,2)/(R))*z_x2
    z_y = (pow(cam2_sigma,2)/(R))*z_y1 + (pow(cam1_sigma,2)/(R))*z_y2
    # R = .01

    # "CORRECT" (measurement update)
    K_x = Pminus_x / (Pminus_x + R)
    xhat_x = xhatminus_x + K_x * (z_x - xhatminus_x)
    P_x = (1 - K_x) * Pminus_x
    K_y = Pminus_y / (Pminus_y + R)
    xhat_y = xhatminus_y + K_y * (z_y - xhatminus_y)
    P_y = (1 - K_y) * Pminus_y

    if cnt > 0:
            v_x = ball_motion[cnt][0] - ball_motion[cnt-1][0]
            v_y = ball_motion[cnt][1] - ball_motion[cnt-1][1]

    # save this result for the next iteration:
    xhatprev_x = xhat_x
    Pprev_x = P_x
    xhatprev_y = xhat_y
    Pprev_y = P_y


    # save measurements for later plotting:
    cam1_estimates.append(cam1_estimate)
    cam2_estimates.append(cam2_estimate)
    closest_camera_estimates.append(cam1_estimate if cam1_sigma < cam2_sigma else cam2_estimate)
    kalman_estimates.append((xhat_x, xhat_y))

    average_estimates.append((
            int(average([cam1_estimate[0], cam2_estimate[0]])),
            int(average([cam1_estimate[1], cam2_estimate[1]]))
            ))



plt.plot([c[0] for c in ball_motion], [kc.img_size[1] - c[1] for c in ball_motion], 'blue', label="Actual Trajectory")
# bp.plot_measurements(np.asarray([c[0] for c in ball_motion]), np.asarray([kc.img_size[1] - c[1] for c in ball_motion]), color='#008B8B', label='kfl_QRFf', lw=1)
plt.plot([c[0] for c in cam1_estimates], [kc.img_size[1] - c[1] for c in cam1_estimates], "red", label="Camera 1 Estimate")
plt.plot([c[0] for c in cam2_estimates], [kc.img_size[1] - c[1] for c in cam2_estimates], "green", label="Camera 2 Estimate")
# plt.plot([c[1] for c in average_estimates], kc.plotcolors['average'], label="Average Estimate")
# plt.plot([c[1] for c in closest_camera_estimates], kc.plotcolors['closest'], label="Closest Camera Estimate")
plt.plot([c[0] for c in kalman_estimates], [kc.img_size[1] - c[1] for c in kalman_estimates], "yellow", label="Kalman Estimate")
# plt.plot([c[1] for c in kalman_estimates], kc.plotcolors['kalman'], label="Kalman Estimate")
# plt.plot(ball_motion)
plt.ylabel('some numbers')
plt.show()
