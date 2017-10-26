from robotcar import robotcar
from random import seed, gauss
from random import seed, uniform
import numpy as np
import matplotlib.pyplot as plt


def get_robot_data(params,number_of_seq):
    # Create Random Physical Landmarks
    num_landmarks = 20
    # Initialize Car and Kalman Filter

    process_uncertainty=params['process_uncertainty']
    fseq_length=params['fseq_length']
    actual_pos=[]
    meas_pos=[]
    for  b in range(number_of_seq):
        robot = robotcar(2, 0.5, num_landmarks, ts=0.1)
        for i in range(fseq_length):
            l_wheel = uniform(5, 15)
            r_wheel = uniform(5, 15)
            robot.move_wheels(l_wheel, r_wheel)
            x1= robot.positionVector[0]
            x2= robot.positionVector[1]
            pos=np.asarray([x1,x2])
            gt = np.hstack((b, pos))
            actual_pos.append(gt)
            x1=x1+gauss(0, process_uncertainty)
            x2=x2+gauss(0, process_uncertainty)
            meas=np.asarray([x1,x2])
            z = np.hstack((b, meas))
            meas_pos.append(z)

    meas_pos=np.asarray(meas_pos,dtype=np.float32)
    actual_pos=np.asarray(actual_pos,dtype=np.float32)
    return (meas_pos,actual_pos)


    # plt.figure(1)
    # actual, = plt.plot(actual_pos[:,0],actual_pos[:,1])
    # actual, = plt.plot(meas_pos[:,0],meas_pos[:,1])
    # # updated, = plt.plot(x_update, y_update)
    # # plt.figlegend( (actual, predicted, updated, lms), ('Actual Position', 'Predicted Position','Updated Position', 'Landmarks'), 'lower right')
    # plt.title('Map of Actual, Predicted and Updated Position')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    #
    # plt.show()