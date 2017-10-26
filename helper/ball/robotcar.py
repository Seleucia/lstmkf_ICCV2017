from math import sin, cos, pow, pi, atan, sqrt
from random import seed, gauss
import numpy as np

class robotcar():
	""" A robot car object that defines the true movement and simulated sensors for differential drive robot """
	def __init__(self, dist_between_wheels, wheel_radius, num_landmarks, ts=0.1):
		""" True Position and Velocity """
		self.x = 0.0
		self.y = 0.0
		self.theta = 0.0
		self.positionVector = np.array([self.x, self.y, self.theta])
		self.omega = 0.0
		self.right_wheel_speed = 0.0
		self.left_wheel_speed = 0.0

		""" Vehicle Properties """
		self.r = wheel_radius
		self.L = dist_between_wheels
		self.processNoiseMean = 0
		self.processNoiseStdDev = 0.0
		self.processNoiseVar = self.processNoiseStdDev*self.processNoiseStdDev
		
		""" Sensors """
		# Encoders
		self.right_encoder = 0
		self.left_encoder = 0

		# Range Finder
		self.range = 0
		self.bearing = 0
		self.num_landmarks = num_landmarks
		
		""" Sensor Properties """
		self.measurementNoiseMean = 0
		self.encoderStdDev = 0.4
		self.encoderVar = self.encoderStdDev*self.encoderStdDev
		self.rangerStdDev = 0.4
		self.rangerVar = self.rangerStdDev*self.rangerStdDev
		self.rangerBearingStdDev = 0.4
		self.rangerBearingVar = self.rangerBearingStdDev*self.rangerBearingStdDev

		""" Time """
		self.time = 0
		self.ts = ts
		
		seed()

		""" Matrices """
		self.states = np.array([[0],[0],[0],[0],[0]])
		self.P = np.array([[self.encoderVar, 1, 0],
					  	   [1, self.encoderVar, 0],
					  	   [0,0,1]

					  	 ])

		self.Q = np.array([[0.1, 0],
						   [0, 0.1]
						  ])
		
		self.R = np.array(
			[
				[0.5, 0],
				[0, 0.5]
			])

		""" Initial Conditions """
		self.position = np.zeros((3+self.num_landmarks*2,1))
		self.theta_prev = self.position[2,0]
		self.x_odom = 0
		self.y_odom = 0
		self.theta_odom = 0.0


	""" These Functions update the actual robot and are used to create simulated data """
	def update_actual_position(self):
		""" Update position based on velocities and add in process noise """
		self.x = self.x + self.V*cos(self.theta)*self.ts
		self.y = self.y + self.V*sin(self.theta)*self.ts
		self.theta = self.omega*self.ts + self.theta
		self.time = self.time + self.ts
		self.positionVector = np.array([self.x, self.y, self.theta])

	def move_wheels(self, left_wheel_speed, right_wheel_speed):
		""" Moves the robot one time step based on wheel speeds (Differential drive) and adds process noise """
		# Create velocities based on wheel speeds
		self.left_wheel_speed = left_wheel_speed
		self.right_wheel_speed = right_wheel_speed
		self.V = (self.left_wheel_speed + self.right_wheel_speed)/2.0
		self.omega = (self.right_wheel_speed - self.left_wheel_speed)/self.L
		self.update_actual_position()

	""" Create Sensor Data """
	def get_odometry(self):
		""" Gets Encoder data and creates sensed velocity and heading """
		self.right_encoder = self.right_wheel_speed + gauss(self.measurementNoiseMean, self.encoderStdDev)
		self.left_encoder = self.left_wheel_speed + gauss(self.measurementNoiseMean, self.encoderStdDev)
		self.V_odom = (self.left_encoder + self.right_encoder)/2
		self.omega_odom = (self.right_encoder - self.left_encoder)/self.L
		self.x_odom = self.x_odom + self.V_odom*sin(self.theta_odom)*self.ts
		self.y_odom = self.y_odom + self.V_odom*cos(self.theta_odom)*self.ts
		self.odometry = np.array([[self.x_odom],[self.y_odom],[self.theta_odom]])
		self.theta_odom = self.omega_odom*self.ts + self.theta_odom


	def get_ranger(self, loc_x, loc_y):
		""" Gets range and bearing to landmark at (loc_x, loc_y) and sets xL, yL and thetaL """
		self.range = sqrt(pow(loc_x - self.x, 2) + pow(loc_y - self.y, 2)) + gauss(self.measurementNoiseMean, self.rangerStdDev)
		if loc_x == self.x:
			if self.x > loc_x:
				self.thetaL = -pi/2 + gauss(self.measurementNoiseMean,self.rangerBearingStdDev)
			else:
				self.thetaL = pi/2 + gauss(self.measurementNoiseMean,self.rangerBearingStdDev)
		else:
			self.thetaL = atan((loc_y-self.y)/(loc_x-self.x)) + gauss(self.measurementNoiseMean, self.rangerBearingStdDev)
		self.xL = self.range*cos(self.thetaL)
		self.yL = self.range*sin(self.thetaL)


	def update_prediction_matrices(self):
		""" Update model matrices for extended kalman filter, with one landmark """


		""" Create Process Model, used for prediction """
		self.process_model = np.array([
			[self.ts*self.V_odom*cos(self.theta_odom)],
			[self.ts*self.V_odom*sin(self.theta_odom)],
			[self.ts*self.omega_odom				 	  ]
		])

		""" Jacobian of the Process Model, Used for covariance propogation equation """
		# self.A = np.array([
		# 	[1, 0, -self.V_odom*self.ts*sin(self.theta_odom)],
		# 	[0, 1,  self.V_odom*self.ts*cos(self.theta_odom)],
		# 	[0, 0, 				1, 								 ]
		# ])
		self.A = np.identity(3+2*self.num_landmarks)
		self.A[0,2] = -self.V_odom*self.ts*sin(self.theta_odom)
		self.A[1,2] = self.V_odom*self.ts*cos(self.theta_odom)

		""" Assemble G (Gaussian Process Noise matrix) """
		self.G = np.array([
			[-self.ts*cos(self.theta_odom), 0],
			[-self.ts*sin(self.theta_odom), 0],
			[0, 						  -self.ts]
		])

		""" Assemble P from already made P  - Propogation Matrix """
		P = np.identity(3+2*self.num_landmarks)
		m,n = self.P.shape
		P[0:m,0:n] = self.P
		self.P = P

		for i in range(self.num_landmarks):
			""" Append landmarks to matrices that need them included """
			self.process_model = np.append(self.process_model,[[0]], axis=0)
			self.process_model = np.append(self.process_model,[[0]], axis=0)
			self.odometry = np.append(self.odometry, [[0]], axis=0)
			self.odometry = np.append(self.odometry, [[0]], axis=0)
			self.G = np.append(self.G, [[0,0]], axis=0)
			self.G = np.append(self.G, [[0,0]], axis=0)

		

	def update_correction_matrices(self, landmarkx, landmarky, landmark_num):
		""" Update the correction matrices with landmark known, H, R, V """

		""" Create H values """
		r = sqrt(pow(self.position[0,0]-landmarkx, 2) + pow(self.position[1,0]-landmarky, 2))
		a = (self.position[0,0]-landmarkx)/r
		b = (self.position[1,0]-landmarky)/r
		c = -(self.position[1,0]-landmarky)/pow(r,2)
		d = (self.position[0,0]-landmarkx)/pow(r,2)

		""" Create H (Jacobian of h) """
		self.H = np.zeros((2,3+2*self.num_landmarks))
		self.H[0:3,0:3] = np.array(
			[
				[a, b, 0],
				[c, d, -1]
			])
		self.H[0:3, 3+landmark_num*2:5+landmark_num*2] = np.array(
			[
				[-a, -b],
				[-c, -d]
			])

		""" Create h (Measurement matrix) """
		self.h = np.array(
			[
				[r],
				[atan((self.position[1,0]-landmarky)/(self.position[0,0]-landmarkx)) - self.position[2,0]]
			])

		""" Create V (measurement noise scalars) """
		self.V = np.array(
			[
				[1, 0],
				[0, 1]
			])

		""" Creates Z matrix  of sensor values"""
		self.Z = np.array(
			[
				[self.range],
				[self.thetaL]
			])


