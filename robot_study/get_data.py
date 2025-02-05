from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from utils import UR5_fk

def balance_data(data, max_demo_len):
    balanced_data = []
    for demo in data:
        balanced_demo = demo + [demo[-1]]*(max_demo_len - len(demo))
        balanced_data.append(balanced_demo)

    return np.array(balanced_data)


def resample_data(data, max_length):
    resampled_data, flattened_data = [], []
    for demo in data:
        demo_idx = np.linspace(0, len(demo)-1, num=max_length, dtype=int)
        resampled_demo = np.array(demo)[demo_idx]
        resampled_data.append(resampled_demo)
        flattened_data.append(resampled_demo.flatten())

    return np.array(resampled_data), np.array(flattened_data)


def states_to_actions(data, flat_data):
    n_demos, n_states, n_dim = np.shape(data)
    start_states = data[[0, -1], 0, :]
    data_action = data[:, 1:, :] - data[:, :-1, :]
    flat_data_action = flat_data[:, n_dim:] - flat_data[:, :-n_dim]

    return data_action, flat_data_action, start_states


def get_trajectories(raw_train_data, max_steps, min_steps):
    # preprocess labeled training data
    raw_train_data = balance_data(raw_train_data, max_steps)
    new_len = min([min_steps, min([len(demo) for demo in raw_train_data])])
    raw_data_resample, raw_data_train = resample_data(raw_train_data, new_len)

    ee_data_resample = []
    for traj in raw_data_resample:
        ee_traj = []
        for waypoint in traj:
            ee_traj.append(UR5_fk(waypoint))
        ee_data_resample.append(ee_traj)
    ee_data_resample = np.asanyarray(ee_data_resample)
    
    # task_dist = []
    # for traj in ee_data_resample:
    #     task_dist.append(maxdistance2human(traj))
    # print(task_dist)

    # convert to action sequence
    _, data_train, _ = states_to_actions(raw_data_resample, raw_data_train)
    
    return data_train.tolist()

def maxdistance2human(ee_traj):

    max_x_distance = 0.53823
    distance2human = (max_x_distance - ee_traj[:, 0])
    dist_idx = np.argpartition(distance2human, -3)[-3:]
    max_dist_avg = np.mean(distance2human[dist_idx])

    return max_dist_avg * 100


def main():

    # load trajectories
    label_traj = []
    label_traj_pre = []
    unlabeled_traj = []
    unlabeled_traj_pre = []

    max_steps = 0.
    for traj_file in sorted(os.listdir('1D_demos')):
        data_file = '1D_demos/' + str(traj_file)
        data = pickle.load(open(data_file, 'rb'), encoding='latin-1')

        traj = data['joint_states']

        if int(traj_file[5]) < 2:
            label_traj.append(traj)

        else:
            unlabeled_traj.append(traj)


        if len(traj) > max_steps:
            max_steps = len(traj)

    reduced_steps = 21
    labelset = get_trajectories(label_traj, max_steps, reduced_steps)
    dataset = get_trajectories(unlabeled_traj, max_steps, reduced_steps)
    dataset += labelset

    # save data
    pickle.dump(dataset, open("data/dataset.pkl", "wb"))
    pickle.dump(labelset, open("data/labelset.pkl", "wb"))
    # print(len(dataset), len(labelset))

if __name__ == "__main__":
    main()