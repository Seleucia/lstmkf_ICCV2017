import numpy as np
import model_runner.klstm.bc.kalcommon as kc
from scipy.linalg import expm
from helper.PosSensor1 import PosSensor1,PosSensor3,PosSensor12d
import ball.rb_gen as rb
import math

def Q_discrete_white_noise(dim, dt=1., var=1.):
    """ Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2 or 3, dt is the time step, and sigma is the
    variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2 or 3)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise
    """

    # assert dim == 2 or dim == 3
    if dim == 2:
        Q = np.array([[.25*dt**4, .5*dt**3],
                   [ .5*dt**3,    dt**2]], dtype=float)
    elif dim==3:
        Q = np.array([[.25*dt**4, .5*dt**3, .5*dt**2],
                   [ .5*dt**3,    dt**2,       dt],
                   [ .5*dt**2,       dt,        1]], dtype=float)
    elif dim==4:
        Q = np.array([[.25*dt**4, .5*dt**3, .5*dt**2],
                   [ .5*dt**3,    dt**2,       dt],
                   [ .5*dt**2,       dt,        1]], dtype=float)

    return Q * var

def van_loan_discretization(F, G, dt):

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

def get_step2d_data(params,number_of_seq):
    (X_train,Y_train)=get_data2d(params,number_of_seq)
    X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list=prepare_sequences(params, X_train, Y_train)
    (X_test,Y_test)=get_data2d(params,number_of_seq/4)
    X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list = prepare_sequences(params, X_test, Y_test)
    # F_list_train=np.ones(shape=(number_of_seq,params['seq_length']))
    # F_list_test=np.ones(shape=(number_of_seq/4,params['seq_length']))
    # G_list_train=np.ones(shape=(number_of_seq,params['seq_length']))
    # G_list_test=np.ones(shape=(number_of_seq/4,params['seq_length']))
    # S_Train_list=range(len(X_train))
    # S_Test_list=range(len(X_test))
    # R_L_Train_list=np.ones(shape=(number_of_seq,params['seq_length']),dtype=np.int32)
    # R_L_Test_list=np.ones(shape=(number_of_seq/4,params['seq_length']),dtype=np.int32)
    return (params,X_train,Y_train,F_list_train,G_list_train,S_Train_list,R_L_Train_list, X_test,Y_test,F_list_test,G_list_test,S_Test_list,R_L_Test_list)


def get_ball_motion(params,number_of_seq):
    (X_train1,X_train2,R1,Y_train)=kc.get_ball_motion_with_measurement(params,number_of_seq)
    (X_test1,X_test2,R1,Y_test)=kc.get_ball_motion_with_measurement(params,number_of_seq/4)
    X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list=prepare_sequences(params, X_train1, Y_train)
    X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list = prepare_sequences(params, X_test1, Y_test)
    # F_list_train=np.ones(shape=(number_of_seq,params['seq_length']))
    # F_list_test=np.ones(shape=(number_of_seq/4,params['seq_length']))
    # G_list_train=np.ones(shape=(number_of_seq,params['seq_length']))
    # G_list_test=np.ones(shape=(number_of_seq/4,params['seq_length']))
    # S_Train_list=range(len(X_train))
    # S_Test_list=range(len(X_test))
    # R_L_Train_list=np.ones(shape=(number_of_seq,params['seq_length']),dtype=np.int32)
    # R_L_Test_list=np.ones(shape=(number_of_seq/4,params['seq_length']),dtype=np.int32)
    return (params,X_train/1000.,Y_train/1000.,F_list_train,G_list_train,S_Train_list,R_L_Train_list, X_test/1000.,Y_test/1000.,F_list_test,G_list_test,S_Test_list,R_L_Test_list)


def get_robot_motion(params,number_of_seq):
    (X_train,Y_train)=rb.get_robot_data(params,number_of_seq)
    (X_test,Y_test)=rb.get_robot_data(params,number_of_seq/4)
    X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list=prepare_sequences(params, X_train, Y_train)
    X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list = prepare_sequences(params, X_test, Y_test)
    # F_list_train=np.ones(shape=(number_of_seq,params['seq_length']))
    # F_list_test=np.ones(shape=(number_of_seq/4,params['seq_length']))
    # G_list_train=np.ones(shape=(number_of_seq,params['seq_length']))
    # G_list_test=np.ones(shape=(number_of_seq/4,params['seq_length']))
    # S_Train_list=range(len(X_train))
    # S_Test_list=range(len(X_test))
    # R_L_Train_list=np.ones(shape=(number_of_seq,params['seq_length']),dtype=np.int32)
    # R_L_Test_list=np.ones(shape=(number_of_seq/4,params['seq_length']),dtype=np.int32)
    return (params,X_train/100.,Y_train/100.,F_list_train,G_list_train,S_Train_list,R_L_Train_list, X_test/100.,Y_test/100.,F_list_test,G_list_test,S_Test_list,R_L_Test_list)

def get_data(params,number_of_seq):
    R_std=params['R_std']
    seq_length=params['seq_length']
    batch_size=params["batch_size"]
    measurements_lst=[]
    groundtruth_lst=[]
    cnt=4
    for b in range(number_of_seq):
        vel=[]
        for i in range(cnt):
            vel_1=tuple([abs(np.random.random()) for k in range(cnt)])
            vel.append(vel_1)

        sensor = PosSensor1(tuple([np.random.random()*10 for i in range(cnt)]), vel, noise_std=R_std)
        res=[]
        per_count=int(math.ceil(seq_length/float(cnt)))
        lst=np.asarray([[i]*per_count for i in range(cnt)]).flatten()

        for i in range(seq_length):
            res.append(sensor.read(lst[i]))
        zy=np.array(res)
        z=zy[:,0,:] #measurement
        y=zy[:,1,:] #ground_truth

        measurements_lst.append(z)
        groundtruth_lst.append(y)


    measurements=np.asarray(measurements_lst,dtype=np.float32)
    groundtruth=np.asarray(groundtruth_lst,dtype=np.float32)
    return (measurements,groundtruth)

def get_data2d(params,number_of_seq):
    R_std=params['R_std']
    fseq_length=params['fseq_length']
    batch_size=params["batch_size"]
    measurements_lst=[]
    groundtruth_lst=[]
    cnt=4
    for b in range(number_of_seq):
        vel=[]
        for i in range(cnt):
            vel_1=tuple([abs(np.random.random()) for k in range(cnt)])
            vel.append(vel_1)

        sensor = PosSensor12d(tuple([np.random.random()*10 for i in range(cnt)]), vel, noise_std=R_std)
        per_count=int(math.ceil(fseq_length/float(cnt)))
        lst=np.asarray([[i]*per_count for i in range(cnt)]).flatten()

        for i in range(fseq_length):
            res=sensor.read(lst[i])
            zy = np.array(res)
            z = zy[0]  # measurement
            y = zy[1]  # ground_truth
            z = np.hstack((b, z))
            y = np.hstack((b, y))

            measurements_lst.append(z)
            groundtruth_lst.append(y)

    measurements=np.asarray(measurements_lst,dtype=np.float32)
    groundtruth=np.asarray(groundtruth_lst,dtype=np.float32)
    return (measurements,groundtruth)

def get_dataV2(params,number_of_seq):
    R_std=params['R_std']
    seq_length=params['seq_length']
    batch_size=params["batch_size"]
    theta=params["theta"]
    measurements_lst=[]
    groundtruth_lst=[]
    cnt=2
    for b in range(number_of_seq):
        sensor = PosSensor3(tuple([np.random.random()*5 for i in range(cnt)]), theta, noise_std=R_std)
        res=[]

        for i in range(seq_length):
            res.append(sensor.read())
        zy=np.array(res)
        z=zy[:,0,:] #measurement
        y=zy[:,1,:] #ground_truth
        # men_z=np.mean(z, axis = 0,dtype=np.float32) # zero-center the data (important)
        # max_z=np.max(z, axis = 0) # zero-center the data (important)
        # min_z=np.min(z, axis = 0) # zero-center the data (important)
        # z=(z-min_z)/(max_z-min_z)
        # z=(z-men_z)

        # men_y=np.mean(y, axis = 0,dtype=np.float32) # zero-center the data (important)
        # max_y=np.max(y, axis = 0) # zero-center the data (important)
        # min_y=np.min(y, axis = 0) # zero-center the data (important)
        # y=(y-min_y)/(max_y-min_y)
        # y=(y-men_y)

        # tmp_x_train=(tmp_train-men)/std

        measurements_lst.append(z)
        groundtruth_lst.append(y)


    measurements=np.asarray(measurements_lst)
    groundtruth=np.asarray(groundtruth_lst)
    return (measurements,groundtruth)

def get_dataV3(params,number_of_seq):
    R_std=params['R_std']
    seq_length=params['seq_length']
    theta=params["theta"]
    measurements_lst=[]
    groundtruth_lst=[]
    cnt=2
    for b in range(number_of_seq):
        sensor = PosSensor3(tuple([np.random.random()*5 for i in range(cnt)]), theta, noise_std=R_std)
        res=[]

        for i in range(seq_length):
            res.append(sensor.read())
        zy=np.array(res)
        z=zy[:,0,:] #measurement
        y=zy[:,1,:] #ground_truth
        # men_z=np.mean(z, axis = 0,dtype=np.float32) # zero-center the data (important)
        # max_z=np.max(z, axis = 0) # zero-center the data (important)
        # min_z=np.min(z, axis = 0) # zero-center the data (important)
        # z=(z-min_z)/(max_z-min_z)
        # z=(z-men_z)

        # men_y=np.mean(y, axis = 0,dtype=np.float32) # zero-center the data (important)
        # max_y=np.max(y, axis = 0) # zero-center the data (important)
        # min_y=np.min(y, axis = 0) # zero-center the data (important)
        # y=(y-min_y)/(max_y-min_y)
        # y=(y-men_y)

        # tmp_x_train=(tmp_train-men)/std

        measurements_lst.append(z)
        groundtruth_lst.append(y)


    measurements=np.asarray(measurements_lst,dtype=np.float32)
    groundtruth=np.asarray(groundtruth_lst,dtype=np.float32)
    return (measurements,groundtruth)

def get_data_combine(params,number_of_seq):
    R_std=params['R_std']
    seq_length=params['seq_length']
    theta=params["theta"]
    measurements_lst=[]
    groundtruth_lst=[]
    cnt=4
    rot_count=25
    step_count=seq_length-rot_count
    for b in range(number_of_seq):
        sensor = PosSensor3(tuple([np.random.random()*2 for i in range(cnt)]), theta, noise_std=R_std)
        res=[]


        for i in range(rot_count):
            res.append(sensor.read())

        vel=[]
        for i in range(cnt):
            vel_1=tuple([abs(np.random.random()/2) for k in range(cnt)])
            vel.append(vel_1)


        sensor = PosSensor12d(tuple(res[-1][1]), vel, noise_std=R_std)
        per_count=int(math.ceil(step_count/float(cnt)))
        lst=np.asarray([[i]*per_count for i in range(cnt)]).flatten()

        for i in range(step_count):
            res.append(sensor.read(lst[i]))

        zy=np.array(res)
        z=zy[:,0,:] #measurement
        y=zy[:,1,:] #ground_truth

        # tmp_x_train=(tmp_train-men)/std

        measurements_lst.append(z)
        groundtruth_lst.append(y)


    measurements=np.asarray(measurements_lst,dtype=np.float32)
    groundtruth=np.asarray(groundtruth_lst,dtype=np.float32)
    return (measurements,groundtruth)

def build_matrices(params):
    test_mode=params['test_mode']
    if test_mode=='rotation':
        F=np.asarray([[np.cos(params["theta"]),-np.sin(params["theta"])],[np.sin(params["theta"]),np.cos(params["theta"])]],dtype=np.float32)
        H=np.zeros((2,2))
        for i in range(params['n_output']):
            if i %2==0:
                H[i/2,i]=1
    elif test_mode=='combine':
        F=np.asarray([[np.cos(params["theta"]),-np.sin(params["theta"])],[np.sin(params["theta"]),np.cos(params["theta"])]],dtype=np.float32)
        H=np.zeros((2,2))
        for i in range(params['n_output']):
            if i %2==0:
                H[i/2,i]=1
    elif test_mode=='step2d':
        dim_x=4
        H=np.zeros((2,4))
        for i in range(dim_x):
            if i %2==0:
                H[i/2,i]=1

        F=np.identity(dim_x)
        for i in range(dim_x):
            if i %2==0:
                F[i,i+1]=1
    elif test_mode=='ball':
        dim_x=4
        H=np.zeros((2,4))
        for i in range(dim_x):
            if i %2==0:
                H[i/2,i]=1

        F=np.identity(dim_x)
        for i in range(dim_x):
            if i %2==0:
                F[i,i+1]=1
    elif test_mode=='robot':
        dim_x=4
        H=np.zeros((2,4))
        for i in range(dim_x):
            if i %2==0:
                H[i/2,i]=1

        F=np.identity(dim_x)
        for i in range(dim_x):
            if i %2==0:
                F[i,i+1]=1
    H=np.asarray([H]*params["batch_size"],dtype=np.float32)
    F=np.asarray([F]*params["batch_size"],dtype=np.float32)

    return F,H
#
# def get_dataV4(params,number_of_seq,R_std):
#     #Change theta
#     seq_length=params['seq_length']
#     theta=params["theta"]
#     measurements_lst=[]
#     groundtruth_lst=[]
#     cnt=2
#     for b in range(number_of_seq):
#         sensor1 = PosSensor3(tuple([np.random.random()*5 for i in range(cnt)]), theta, noise_std=R_std)
#         res=[]
#
#         for i in range(seq_length):
#             theta=theta+0.0000
#             res.append(sensor1.read(theta))
#         zy=np.array(res)
#         z=zy[:,0,:] #measurement
#         y=zy[:,1,:] #ground_truth
#         # men_z=np.mean(z, axis = 0,dtype=np.float32) # zero-center the data (important)
#         # max_z=np.max(z, axis = 0) # zero-center the data (important)
#         # min_z=np.min(z, axis = 0) # zero-center the data (important)
#         # z=(z-min_z)/(max_z-min_z)
#         # z=(z-men_z)
#
#         # men_y=np.mean(y, axis = 0,dtype=np.float32) # zero-center the data (important)
#         # max_y=np.max(y, axis = 0) # zero-center the data (important)
#         # min_y=np.min(y, axis = 0) # zero-center the data (important)
#         # y=(y-min_y)/(max_y-min_y)
#         # y=(y-men_y)
#
#         # tmp_x_train=(tmp_train-men)/std
#
#         measurements_lst.append(z)
#         groundtruth_lst.append(y)
#
#
#     measurements=np.asarray(measurements_lst)
#     groundtruth=np.asarray(groundtruth_lst)
#     return (measurements,groundtruth)


def prepare_sequences(params,db_values_x,db_values_y):
    p_count=params['seq_length']
    max_count=params['max_count']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    Y_d=[]
    X_d=[]
    F_l=[] #Repeating frame list....
    R_L=[] #Repeating frame list....
    r_l=[]
    prev_sq_id=0
    curr_id=0
    for item_id in range(len(db_values_x)):
        f=""
        sq_id=int(db_values_x[item_id][0])
        x=db_values_x[item_id][1:]
        y=db_values_y[item_id][1:]

        if prev_sq_id!=sq_id:
            prev_sq_id=sq_id
            if(len(Y_d)>0): #If there is left over from previus sequence add them...
                residual=len(Y_d)%p_count
                residual=p_count-residual
                res_y=residual*[Y_d[-1]]
                res_x=residual*[X_d[-1]]
                res_f=residual*[F_l[-1]]
                Y_d.extend(res_y)
                X_d.extend(res_x)
                F_l.extend(res_f)
                r_l.extend(residual*[0])
                if len(Y_d)==p_count and p_count>0:
                    S_L.append(curr_id)
                    Y_D.append(Y_d)
                    X_D.append(X_d)
                    F_L.append(F_l)
                    R_L.append(r_l)
                    Y_d=[]
                    X_d=[]
                    F_l=[]
                    r_l=[]
                    if len(Y_D)>=max_count:
                        return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L,np.asarray(R_L,dtype=np.int32))
            #we should add current frame to sequence.
            curr_id=sq_id
            Y_d.append(y)
            X_d.append(x)
            F_l.append(f)
            r_l.append(1)

        else:
            curr_id=sq_id
            Y_d.append(y)
            X_d.append(x)
            F_l.append(f)
            r_l.append(1)
        if len(Y_d)==p_count and p_count>0:
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            S_L.append(sq_id)
            R_L.append(r_l)
            Y_d=[]
            X_d=[]
            F_l=[]
            r_l=[]
        if len(Y_D)>=max_count:
                    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L,np.asarray(R_L,dtype=np.int32))

    if(len(Y_d)>0): #If there is left over from previus sequence add them...
        residual=len(Y_d)%p_count
        residual=p_count-residual
        res_y=residual*[Y_d[-1]]
        res_x=residual*[X_d[-1]]
        res_f=residual*[F_l[-1]]
        Y_d.extend(res_y)
        X_d.extend(res_x)
        F_l.extend(res_f)
        r_l.extend(residual*[0])
        if len(Y_d)==p_count and p_count>0:
            S_L.append(curr_id)
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            R_L.append(r_l)
    return (np.asarray(X_D,dtype=np.float32),np.asarray(Y_D,dtype=np.float32),np.asarray(F_L),G_L,S_L,np.asarray(R_L,dtype=np.int32))

def compute_loss(gt,est):
    shp=gt.shape
    if len(shp)==2: #kalman
        diff_vec=np.abs(gt - est) #13*3
        sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
        return np.mean(sq_m)
    elif len(shp)==3:
        lst=[]
        for i in range(shp[0]):
            g=gt[i]
            e=est[i]
            diff_vec=np.abs(g - e) #13*3
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            lst.append(np.mean(sq_m))
        return np.mean(lst)

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump