from numpy.random import randn
import math
import numpy as np

class PosSensor1(object):
    def __init__(self, pos, vel, noise_std):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1], pos[2], pos[3]]

    def read(self,idx):
        if idx==2:
            self.pos[0] -= self.vel[idx][0]
            self.pos[1] -= self.vel[idx][1]
            self.pos[2] -= self.vel[idx][2]
            self.pos[3] -= self.vel[idx][3]
        if idx==3:
            self.pos[1] -= self.vel[idx][1]
            self.pos[0] -= self.vel[idx][0]
            self.pos[2] -= self.vel[idx][2]
            self.pos[3] -= self.vel[idx][3]
        else:
            self.pos[0] += self.vel[idx][0]
            self.pos[1] += self.vel[idx][1]
            self.pos[2] += self.vel[idx][2]
            self.pos[3] += self.vel[idx][3]
        # self.pos[0] += self.vel[idx][0]
        # self.pos[1] += self.vel[idx][1]
        # self.pos[2] += self.vel[idx][2]
        # self.pos[3] += self.vel[idx][3]

        measurement=[self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std,
                self.pos[2] + randn() * self.noise_std,
                self.pos[3] + randn() * self.noise_std
                ]
        ground_truth=[self.pos[0] ,
                self.pos[1] ,
                self.pos[2],
                self.pos[3]
                ]


        return measurement,ground_truth

class PosSensor12d(object):
    def __init__(self, pos, vel, noise_std):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self,idx):
        if idx==2:
            self.pos[0] -= self.vel[idx][0]
            self.pos[1] -= self.vel[idx][1]
        if idx==3:
            self.pos[1] -= self.vel[idx][1]
            self.pos[0] -= self.vel[idx][0]
        else:
            self.pos[0] += self.vel[idx][0]
            self.pos[1] += self.vel[idx][1]


        measurement=[self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std
                ]
        ground_truth=[self.pos[0] ,
                self.pos[1]
                ]

        return measurement,ground_truth

class PosSensor3(object):
    def __init__(self, pos, theta, noise_std):
        self.theta = theta
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        self.R=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype=np.float32)

    def read(self,theta=None):
        pos=np.asarray( self.pos )
        if theta !=None:
            self.R=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),-np.cos(theta)]],dtype=np.float32)
        res=np.dot(self.R,pos)
        self.pos[0] = res[0]
        self.pos[1] = res[1]
        measurement=[self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std
                ]
        ground_truth=[self.pos[0] ,
                self.pos[1]
                ]


        return measurement,ground_truth

class PosSensor_3drotate(object):
    def rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    def __init__(self, pos, theta, noise_std):
        self.theta = theta
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1], pos[2]]

    def read(self):
        v = [1, 1.2, 1]
        res=np.dot(self.rotation_matrix(self.pos,self.theta), v)
        self.pos[0] = res[0]
        self.pos[1] = res[1]
        self.pos[2] = res[2]
        measurement=[self.pos[0] + randn() * self.noise_std,
                     self.pos[1] + randn() * self.noise_std,
                     self.pos[2] + randn() * self.noise_std]
        ground_truth=[self.pos[0] ,self.pos[1] ,self.pos[2]
                ]

        return measurement,ground_truth

class PosSensor2(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        measurement=[self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std
                ]
        ground_truth=[self.pos[0] ,
                self.pos[1]
                ]


        return measurement,ground_truth