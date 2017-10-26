import numpy as np
v=[1,1,1]
R=[[np.cos(v[1])*np.cos(v[2]),np.cos(v[2])*np.sin(v[0])*np.sin(v[1])-np.cos(v[0])*np.sin(v[2]), np.cos(v[0])*np.cos(v[2])*np.sin(v[1])+np.sin(v[0])*np.sin(v[2]), 0],
   [np.cos(v[1])*np.sin(v[2]),np.cos(v[0])*np.cos(v[2])+np.sin(v[0])*np.sin(v[1])*np.sin(v[2]), -np.cos(v[2])*np.sin(v[0])+np.cos(v[0])*np.sin(v[1])*np.sin(v[2]), 0],
   [-np.sin(v[1]),np.cos(v[1])*np.sin(v[0]),np.cos(v[0])*np.cos(v[1]),0],
   [0,0,0,1]
   ]

pos=[1.1,2.2,3.3,1]

for i in range(100):
    pos=np.dot(R,pos)
    print pos