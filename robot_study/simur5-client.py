import socket
import numpy as np
import scipy as sp
import time
import torch
import pickle
import os
from models import MyModel, SeGMA, calculate_logits
import torch.nn.functional as F
from simpleEnv import simpleEnv
from utils import UR5_fk
import math
import pybullet as p
import datetime

def listen2gui(conn):
    state_length = 1 + 1 + 1 + 1 + 1
    data = conn.recv(1024).decode()
    state_str = list(data.split(","))

    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx + 1: idx + 1 + state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
    except ValueError:
        return None
    
    if len(state_vector) is not state_length:
        return None
    
    state_vector = np.asarray(state_vector)
    state = {}
    state["z_task"] = np.asarray(task_embedding[int(state_vector[0])])
    state["z_style"] = state_vector[1]
    state["traj_flag"] = state_vector[2]
    state["env_flag"] = state_vector[3]
    state["save_flag"] = state_vector[4]

    return state

def guiState(conn):
    while True:
        state = listen2gui(conn)
        if state is not None:
            break
    return state

def send2gui(conn, play_traj_flag, reset_env_flag, distance2human, save_traj_flag):

    send_msg = "s," + str(int(play_traj_flag == True)) + "," + str(int(reset_env_flag == True)) + "," + str(distance2human) + "," + str(int(save_traj_flag == True))
    conn.send(send_msg.encode())


def actions_to_states(data_action, start_state):
    input_dim = 6
    seq_len = 20
    traj_action = data_action
    action_seq = np.reshape(traj_action, (seq_len, input_dim))
    state_seq = [start_state]

    for seq_idx, action in enumerate(action_seq):
        state_seq.append(state_seq[seq_idx] + action)

    return state_seq

# rollout a trajectory given a z_task and z_style
def rollout(z_task, z_style, start_state):

    if algo == '0':
        z_task = torch.FloatTensor(z_task).unsqueeze(0)
        z_style = torch.FloatTensor([[z_style * style_scale]])
        actions = model.decoder(z_task, z_style).detach().squeeze(0).numpy()
    else:
        z_style = torch.FloatTensor([[z_style * style_scale]])
        z_mean = model.means[z_task]
        actions = model.traj_decoder(z_style + z_mean).detach().squeeze(0).numpy()

    joint_traj = actions_to_states(actions, start_state)

    ee_traj = []
    for state in joint_traj:
        ee_traj.append(UR5_fk(state))
    ee_traj = np.asarray(ee_traj)

    return joint_traj, ee_traj

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


def maxdistance2human(ee_traj):

    max_x_distance = 0.53823
    distance2human = (max_x_distance - ee_traj[:, 0])
    dist_idx = np.argpartition(distance2human, -3)[-3:]
    max_dist_avg = np.mean(distance2human)

    return max_dist_avg * 100

# Connect to gui server
HOST = "127.0.0.1"
PORT = 65432
gui_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
gui_conn.connect((HOST, PORT))


# Load pick demos for each task and start state for each rollout
# Create an array of min and max x-distance per task
pick_joint_demos = []
start_states = []

for traj_file in sorted(os.listdir('data')):

    if 'pick' in traj_file:
        data_file = 'data/' + str(traj_file)
        data = pickle.load(open(data_file, 'rb'), encoding='latin-1')
        pick_joint_demos.append(np.asanyarray(data['joint_states']))
        start_states.append(np.asarray(data['joint_states'][-1]))
        

uid = input("Enter user id: ")
algo = input("Select algorithm (0: Ours, 1: Baseline)- ")

# Set robot home position
HOME = pick_joint_demos[0][0]

# Load ur5 pybullet environment
env = simpleEnv(HOME)

# Load dataset
dataset = pickle.load(open('data/labelset.pkl', 'rb'), encoding='latin-1')

if algo == '0':
    algorithm = 'Ours'

    style_scale = 1.0

    # load trained model
    torch.manual_seed(0)
    model = MyModel()
    model.load_state_dict(torch.load('data/model.pt'))
    model.eval()

    # Define task embedding index order
    latent_tasks = model.task_encode(torch.FloatTensor(dataset)).detach().numpy().tolist()
    task_embedding = [latent_tasks[0], latent_tasks[2], latent_tasks[4]]

else:
    algorithm = 'Baseline'

    labels = np.array([[0, 0, 0]]*3 + [[0, 0, 1]]*2 + [[0, 1, 0]]*2 + [[1, 0, 0]]*2)
    style_scale = 1.5

    # load trained model
    torch.manual_seed(0)
    model = SeGMA(120, 3, 1, labels)
    model.load_state_dict(torch.load('data/model_baseline.pt'))
    model.eval()

    # Define task embedding index order
    zs = model.traj_encoder(torch.FloatTensor(dataset))
    task_logits = calculate_logits(zs, model.means, model.variances, model.probs)
    task_probs = F.softmax(task_logits, dim=-1)
    task_predict = np.argmax(task_probs.detach().numpy(), axis=1)
    task_embedding = [task_predict[0], task_predict[2], task_predict[4]]

def main():
    ee_traj_prev = np.zeros((21, 6))
    sphereID_prev = None

    objects = [['Plate Holder', 'Plate'], ['Mug Stand', 'Mug']]
    sim_counter = {'Task 1': 0, 'Task 2': 0, 'Task 3': 0}

    saveloc = 'data/user' + str(uid)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    distance2human = 0

    while True:
        # Update states
        state = guiState(gui_conn)
        traj_flag, env_flag, save_flag = state['traj_flag'], state['env_flag'], state['save_flag']

        # Update pick trajectory
        task_idx = task_embedding.index(state['z_task'].tolist())

        # Rollout a trajectory according to current z parameters
        joint_traj, ee_traj = rollout(state['z_task'], state['z_style'], start_states[task_idx])


        # NOTE: in the actual user study, users were prevented from visualizing the trajectory 
        ee_traj_prev, sphereID_prev = updateTraj(ee_traj, ee_traj_prev, sphereID_prev)
        
        # if task_idx < 2:
        #     # If the trajectories has changed update the preview plot
        #     ee_traj_prev, sphereID_prev = updateTraj(ee_traj, ee_traj_prev, sphereID_prev)

        # elif task_idx >=2 and sphereID_prev != None:

        #     for sphereID in sphereID_prev:
        #         env.removeBody(sphereID)

        #     sphereID_prev = None
        #     ee_traj_prev = ee_traj


        # Play trajectory if the button has been pressed
        if traj_flag:
            
            if task_idx < 2:
                env.reset_obj(objects[task_idx]) # reset object for task
            
            if task_idx == 0:
                pick1_inter = np.array([-1.1709454695331019, -1.9433038870440882, -1.967215363179342, -0.3079474608050745, 2.54880428314209, -0.8505728880511683])
                env.go2pose(pick1_inter, space='joint', action_scale=0.5)
                env.go2pose(start_states[task_idx], space='joint', action_scale=0.5)
            
            elif task_idx > 0:
                env.go2pose(start_states[task_idx], space='joint', action_scale=0.5)

            env.setGripperPose(0)

            joint_traj = np.asanyarray(joint_traj)
            env.playTrajectory(joint_traj, total_time=5, action_scale=1, space='joint')

            traj_flag = False

            env.setGripperPose(0.085)
            env.go2pose(HOME, space='joint', action_scale=0.4)

            distance2human = maxdistance2human(ee_traj)

            sim_counter['Task ' + str(task_idx + 1)] += 1
            pickle.dump(sim_counter, open(saveloc + '/sim_counter_' + algorithm + '.pkl', 'wb'))

        # Save reconstruction for ur5 and save data to user folder
        if save_flag:
            np.save('data/userRecon' + str(task_idx + 1), ee_traj)
            
            distance2human = maxdistance2human(ee_traj)
            recon = {'join_traj': joint_traj, 'ee_traj': ee_traj, 'z_task': state['z_task'], 'z_style' : state['z_style'], 'avg_x_distance': distance2human}
            curr_time = datetime.datetime.now()
            curr_time = curr_time.strftime("%H:%M:%S")
            curr_time = curr_time.replace(" ", "_").replace(":", "_")
            pickle.dump(recon, open(saveloc + '/task' + str(task_idx + 1) + algorithm + '_' + curr_time + '.pkl', 'wb'))
            save_flag = False

        # Reset env if the button has been pressed
        if env_flag:
            env.reset_env()
            env_flag = False

        send2gui(gui_conn, traj_flag, env_flag, distance2human, save_flag) # send flag states to gui


if __name__ == '__main__':
    main()
            