import pickle
import os
import numpy as np
from simpleEnv import simpleEnv
import math



def updateTraj(ee_traj, ee_traj_prev, sphereID_prev, rgbaColor=[1, 0, 0, 1]):
    
    sphereID_list = sphereID_prev

    # If the trajectory has changed update the preview plot
    if not np.array_equal(ee_traj, ee_traj_prev):
        
        if sphereID_prev is not None:
            for sphereID in sphereID_prev:
                env.removeBody(sphereID) 

        num_sections = math.ceil(ee_traj.shape[0] / 16)
        split_traj = np.array_split(ee_traj, num_sections)

        sphereID_list = []
        for traj in split_traj:
            _, sphereID = env.showVisualSphere(traj[:, :3], radius=2e-2, rgbaColor=rgbaColor)
            sphereID_list.append(sphereID)

        ee_traj_prev = ee_traj

    return ee_traj_prev, sphereID_list


# Load robot home position
HOME = pickle.load(open('data/pick1.pkl', 'rb'), encoding='latin-1')
HOME = HOME['joint_states'][0]

# Load extreme demonstration for task1 and task2
extreme_0 = []
extreme_1 = []
for traj_file in sorted(os.listdir('1D_demos')):
    data = pickle.load(open('1D_demos/' + traj_file, 'rb'), encoding='latin-1')

    if int(traj_file[4]) < 3:
        if int(traj_file[5]) == 0:
            extreme_0.append(np.asanyarray(data['ee_state']))
        elif int(traj_file[5]) == 1:
            extreme_1.append(np.asanyarray(data['ee_state']))

# extreme_0 = np.concatenate((extreme_0[0], extreme_0[1]))
# extreme_1 = np.concatenate((extreme_1[0], extreme_1[1]))

# Initialize pybullet env
env = simpleEnv(HOME)

ee_traj_prev = np.zeros((21,6))
sphereID_prev = None

while True:
    
    extreme = input("Select extreme demos to show (0 or 1): ")
    
    if int(extreme) == 0:
        ee_traj = extreme_0[1]
        rgbaColor = [1, 0, 0, 1]
    elif int(extreme) == 1:
        ee_traj = extreme_1[1]
        rgbaColor = [0, 1, 0, 1]
        
    ee_traj_prev, sphereID_prev = updateTraj(ee_traj, ee_traj_prev, sphereID_prev, rgbaColor=rgbaColor)


