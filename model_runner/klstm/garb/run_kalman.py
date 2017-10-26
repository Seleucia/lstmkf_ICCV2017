import matplotlib.pyplot as plt
import numpy as np
from model_runner.klstm.kalman_filter import KalmanFilter
import helper.kalman_libs as kl

def test_filter(params,zs):

    # R_std = 0.35
    Q_std = 0.04
    def tracker1():
        dim_x=8
        dim_z=4
        tracker = KalmanFilter(dim_x,dim_z)
        dt = 1.0   # time step


        f_mat=np.identity(dim_x)
        for i in range(dim_x):
            if i %2==0:
                f_mat[i,i+1]=1

        tracker.F =f_mat

        # tracker.F = np.array([[1, dt, 0,  0],
        #                       [0,  1, 0,  0],
        #                       [0,  0, 1, dt],
        #                       [0,  0, 0,  1]])
        tracker.u = 0.
        h_mat=np.zeros((4,8))
        for i in range(dim_x):
            if i %2==0:
                h_mat[i/2,i]=1
        tracker.H=h_mat
        # tracker.H = np.array([[1/0.3048, 0, 0, 0],
        #                       [0, 0, 1/0.3048, 0]])

        tracker.R = (np.eye(dim_z) * R_std**2)*200
        # q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=Q_std**2)
        # tracker.Q = block_diag(q, q)
        G=np.identity(dim_x)
        (sigma, Q)=kl.van_loan_discretization(f_mat, G, dt)
        tracker.Q =Q
        tracker.x = np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).T
        tracker.P = np.eye(dim_x) * 500.
        return tracker

    robot_tracker = tracker1()
    R_std=params['R_std']
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    return mu[:,(0,2,4,6),:], cov


def test_filter2d(params,zs):

    def tracker1():
        dim_x=4
        dim_z=2
        tracker = KalmanFilter(dim_x,dim_z)
        dt = 1.0   # time step
        tracker.F =params['F']
        tracker.H=params['H']
        tracker.u = 0.

        tracker.R = (np.eye(dim_z) * R_std**2)*200
        G=np.identity(dim_x)
        (sigma, Q)=kl.van_loan_discretization(tracker.F, G, dt)
        tracker.Q =Q
        tracker.x = np.array([[0, 0, 0, 0]]).T
        tracker.P = np.eye(dim_x) * 500.
        return tracker

    R_std=params['R_std']
    robot_tracker = tracker1()
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    return mu[:,(0,2),:], cov

def test_filter3(params,zs):
    # R_std = 0.35
    # Q_std = 0.04

    def tracker1():
        dim_x=params['F'].shape[0]
        dim_z=params['H'].shape[0]
        tracker = KalmanFilter(dim_x,dim_z)
        dt = 1.0   # time step


        # f_mat=np.identity(dim_x)
        # for i in range(dim_x):
        #     if i %2==0:
        #         f_mat[i,i+1]=1

        # f_mat=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype=np.float32)
        tracker.F =params['F']

        # tracker.F = np.array([[1, dt, 0,  0],
        #                       [0,  1, 0,  0],
        #                       [0,  0, 1, dt],
        #                       [0,  0, 0,  1]])
        # tracker.u = 0.
        # h_mat=np.zeros((2,2))
        # for i in range(dim_x):
        #     if i %2==0:
        #         h_mat[i/2,i]=1
        tracker.H=params['H']
        # tracker.H = np.array([[1/0.3048, 0, 0, 0],
        #                       [0, 0, 1/0.3048, 0]])

        tracker.R = (np.eye(dim_z) * R_std**2)
        # q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=Q_std**2)
        # tracker.Q = block_diag(q, q)
        G=np.identity(dim_x)
        (sigma, Q)=kl.van_loan_discretization(params['F'], G, dt)
        tracker.Q =Q/10000000.0
        tracker.x = np.array([[0, 0,]]).T
        tracker.P = np.eye(dim_x) * 100.
        return tracker
    R_std=params['R_std']
    theta= params["theta"]
    robot_tracker = tracker1()
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    return mu, cov


def test_filterxx(params,zs,R_std):

    # R_std = 0.35
    Q_std = 0.04
    def tracker1():
        dim_x=2
        dim_z=2
        tracker = KalmanFilter(dim_x,dim_z)
        dt = 1.0   # time step


        # f_mat=np.identity(dim_x)
        # for i in range(dim_x):
        #     if i %2==0:
        #         f_mat[i,i+1]=1

        theta=0.1
        f_mat=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype=np.float32)
        tracker.F =f_mat

        # tracker.F = np.array([[1, dt, 0,  0],
        #                       [0,  1, 0,  0],
        #                       [0,  0, 1, dt],
        #                       [0,  0, 0,  1]])
        tracker.u = 0.
        h_mat=np.zeros((2,2))
        for i in range(dim_x):
            if i %2==0:
                h_mat[i/2,i]=1
        tracker.H=h_mat
        # tracker.H = np.array([[1/0.3048, 0, 0, 0],
        #                       [0, 0, 1/0.3048, 0]])

        tracker.R = (np.eye(dim_z) * R_std**2)
        # q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=Q_std**2)
        # tracker.Q = block_diag(q, q)
        G=np.identity(dim_x)
        (sigma, Q)=kl.van_loan_discretization(f_mat, G, dt)
        tracker.Q =Q
        tracker.x = np.array([[0, 0,]]).T
        tracker.P = np.eye(dim_x) * 100.
        return tracker

    R_std=params['R_std']
    robot_tracker = tracker1()
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    return mu, cov