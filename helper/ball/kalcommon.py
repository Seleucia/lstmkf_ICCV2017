import cv
import numpy as np
import random
import numpy
from math import sqrt
from model_runner.klstm.bc.brownian import brownian

# params for gaussian noise for ball movement:
mu = 0
sigma = 3
process_uncertainty = 15

# frame size and other video params
img_size = (1280, 720)
img_depth = 8
img_channels = 3
fps = 30
codec = cv.CV_FOURCC('M', 'J', 'P', 'G')
# codec = cv.CV_FOURCC('P', 'I', 'M', '1')

# painting colors
ballcolor = cv.RGB(0,54,94)
cam1color = cv.RGB(139,69,19)
cam2color = cv.RGB(255,255,0)
cam3color = cv.RGB(224,255,255)
kalmancolor = cv.RGB(255,255,255)
averagecolor = cv.RGB(255,0,255)


font_color = cv.RGB(0,255,0)

# plotting colors
plotcolors = {
    'actual':'b',
    'cam1':'g',
    'cam2':'r',
    'average':'c',
    'closest':'y',
    'kalman':'m'
    }

# some filenames:
x_estimates_filename = 'kal-x_estimates.png'
y_estimates_filename = 'kal-y_estimates.png'
cam_errors_filtered_filename = 'kal-cam_errors_filtered.png'
cam_errors_filename = 'kal-cam_errors.png'
trajectories_filename = 'kal-trajectories.png'


# helper functions


def get_dist_between_2_points(point1, point2):
    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))

def get_cam_estimate(ball_coords, cam_coords):
    """
    Gets a phoney estimate for where the ball is depending on how far
    away the camera is. Returns sigma value and "measured" x,y values
    in a tuple like so:
    ((x,y), sigma)
    
    Arguments:
    - `ball_coords`: The ball's real location
    - `cam_coords`: The camera's coordinates
    """
    alpha = .009
    sigma = alpha * get_dist_between_2_points(ball_coords, cam_coords)
    x = ball_coords[0] + random.gauss(0, pow(sigma,2))
    y = ball_coords[1] + random.gauss(0, pow(sigma,2))
    return ((int(x), int(y)) , sigma)


def get_brownian_ball_motion(start, nframes):
    """
    returns a list of coords representing the ball motion using
    Brownian motion
    """
    # The Wiener process parameter.
    delta = 60
    # Total time.
    T = 10.0
    # Number of steps.
    N = nframes
    # Time step size
    dt = T/N
    # Initial values of x.
    x = numpy.empty((2,N+1))
    x[:, 0] = start

    brownian(x[:,0], N, dt, delta, out=x[:,1:])

    out = []
    for i in x.transpose():
        out.append(( int(i[0]), int(i[1]) ))
    return out

def get_simple_ball_motion(start, nseconds):
    "Gets simple ball motion"
    total_frames = nseconds * fps
    out = [start]
    third_of_frames = total_frames/3

    # spend the first third of the time creeping towards a camera:
    dx = float((img_size[0]-100) - start[0]) / third_of_frames
    dy = float(100 - start[1]) / third_of_frames
    # print dx, dy, start, third_of_frames, total_frames, nseconds

    for i in xrange(1, third_of_frames):
        prev_coords = out[i-1]
        pU = random.gauss(0,process_uncertainty)
        new_coords = (prev_coords[0] + dx + pU, prev_coords[1] + dy + pU)
        out.append( new_coords )

    # spend the second third of the time going towards the left
    dx = float(100 - out[len(out)-1][0]) / third_of_frames
    dy = float(start[1] - out[len(out)-1][1]) / third_of_frames
    for i in xrange(third_of_frames, (third_of_frames)*2):
        prev_coords = out[i-1]
        pU = random.gauss(0,process_uncertainty)
        new_coords = (prev_coords[0] + dx + pU, prev_coords[1] + dy + pU)
        out.append( new_coords )

    # spend the final third of the time doing something else
    initial_spot = new_coords
    for i in xrange(0, third_of_frames):
        sz0 = img_size[0] - 100
        sz1 = img_size[1] - 100
        xval = i*(sz0/third_of_frames)
        pU = random.gauss(0,process_uncertainty)
        out.append( (xval + initial_spot[0] + pU, start[1] + (sz1/2.5)*numpy.sin(2*numpy.pi*(float(2)/sz0)*xval) + pU) )

    ret = []
    for i in out:
        ret.append( (int(i[0]), int(i[1])) )
    return ret


def get_ball_motion_with_measurement(params,number_of_seq):
    "Gets simple ball motion"
    measurements1_lst=[]
    measurements2_lst=[]
    sigma_lst=[]
    groundtruth_lst=[]

    fseq_length=params['fseq_length']
    for  b in range(number_of_seq):
        start=numpy.random.uniform(1, 2, size=2)*100
        total_frames = fseq_length
        out = [start]
        third_of_frames = total_frames/3

        # spend the first third of the time creeping towards a camera:
        dx = float((img_size[0]-100) - start[0]) / third_of_frames
        dy = float(100 - start[1]) / third_of_frames
        # print dx, dy, start, third_of_frames, total_frames, nseconds

        for i in xrange(1, third_of_frames):
            prev_coords = out[i-1]
            pU = random.gauss(0,process_uncertainty)
            new_coords = (prev_coords[0] + dx + pU, prev_coords[1] + dy + pU)
            out.append( new_coords )

        # spend the second third of the time going towards the left
        dx = float(100 - out[len(out)-1][0]) / third_of_frames
        dy = float(start[1] - out[len(out)-1][1]) / third_of_frames
        for i in xrange(third_of_frames, (third_of_frames)*2):
            prev_coords = out[i-1]
            pU = random.gauss(0,process_uncertainty)
            new_coords = (prev_coords[0] + dx + pU, prev_coords[1] + dy + pU)
            out.append( new_coords )

        # spend the final third of the time doing something else
        initial_spot = new_coords
        for i in xrange(0, third_of_frames):
            sz0 = img_size[0] - 100
            sz1 = img_size[1] - 100
            xval = i*(sz0/third_of_frames)
            pU = random.gauss(0,process_uncertainty)
            out.append( (xval + initial_spot[0] + pU, start[1] + (sz1/2.5)*numpy.sin(2*numpy.pi*(float(2)/sz0)*xval) + pU) )


        for i in out:
            y=np.asarray([int(i[0]), int(i[1])])
            (cam1_estimate,cam1_sigma)=get_cam_estimate(y, (10,10))
            (cam2_estimate,cam2_sigma)=get_cam_estimate(y, (1270, 10))
            z1=np.asarray([cam1_estimate[0],cam1_estimate[1]])
            z2=np.asarray([cam2_estimate[0],cam2_estimate[1]])
            s1=[cam1_sigma,cam1_sigma]
            z1 = np.hstack((b, z1))
            z2 = np.hstack((b, z2))
            y = np.hstack((b, y))
            groundtruth_lst.append(y)
            measurements1_lst.append(z1)
            measurements2_lst.append(z2)
            sigma_lst.append(s1)
    measurements1=np.asarray(measurements1_lst,dtype=np.float32)
    measurements2=np.asarray(measurements2_lst,dtype=np.float32)
    sigma1=np.asarray(sigma_lst,dtype=np.float32)
    groundtruth=np.asarray(groundtruth_lst,dtype=np.float32)
    return (measurements1,measurements2,sigma1,groundtruth)

def get_still_ball_motion(start, nseconds):
    "Gets still ball motion -- no motion"
    return [start for frame in xrange(fps * nseconds)]

# returns the linear interpolation of the value given the two ranges
# (passed in as 2-tuples). For example, if you have a number in the
# range [0..500] that you'd like to map to a number in the range
# [0..255] you would call this function like so:
# newVal = lerp(num, (0,500), (0,255))
def lerp(val, from_range, to_range):
    # equation taken from http://en.wikipedia.org/wiki/Linear_interpolation
    x0 = from_range[0];
    x1 = from_range[1];
    y0 = to_range[0];
    y1 = to_range[1];
    x = val;
    return y0 + (x - x0)*((y1-y0)/(x1-x0))


