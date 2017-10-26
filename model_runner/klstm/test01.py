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
from helper import config
import matplotlib.pyplot as plt

total_seconds = 18
total_frames = total_seconds*kc.fps
center_coords = (kc.img_size[0]/2, kc.img_size[1]/2)
# center_coords=[640,360]
# ball_motion = kc.get_simple_ball_motion(center_coords, total_seconds)
# print ball_motion

center_coords=[6.2,3.0]
params=config.get_params()
number_of_batches=400
batch_size=params['batch_size']
params['fseq_length']=100

(X_train1,X_train1,R1,Y_train)=kc.get_ball_motion_with_measurement(params,batch_size*number_of_batches)
ball_motion =Y_train[Y_train[:,0]==0][:,[1,2]]
ball_motion1 =Y_train[Y_train[:,0]==100][:,[1,2]]
ball_motion2 =Y_train[Y_train[:,0]==200][:,[1,2]]
print ball_motion1
plt.plot([c[0] for c in ball_motion], [c[1] for c in ball_motion], 'blue', label="Actual Trajectory")
plt.plot([c[0] for c in ball_motion1], [c[1] for c in ball_motion1], 'red', label="Actual Trajectory")
plt.plot([c[0] for c in ball_motion2], [c[1] for c in ball_motion2], 'green', label="Actual Trajectory")
plt.ylabel('some numbers')
plt.show()