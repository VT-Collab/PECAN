import os
import pickle
import scipy as sp
import numpy as np
from copy import deepcopy
from itertools import product, combinations
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from models import *


class MotionData(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


class CustomScaler:

    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, data):
        self.max = np.max(data)
        self.min = np.min(data)

    def transform(self, data):
        data_range = self.max - self.min
        scaled_data = (data - self.min) / data_range

        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)

        return scaled_data

    def inverse_transform(self, scaled_data):
        data_range = self.max - self.min
        data = (scaled_data*data_range) + self.min

        return data


def balance_data(data):
    max_demo_len = max([len(demo) for demo in raw_train_data])
    balanced_data = []
    for demo in data:
        balanced_demo = demo + [demo[-1]]*(max_demo_len - len(demo))
        balanced_data.append(balanced_demo)

    return balanced_data


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


def actions_to_states(data_action, start_state):
    data_states = []
    for traj_action in data_action:
        action_seq = np.reshape(traj_action, (seq_len, input_dim))
        state_seq = [start_state]
        for seq_idx, action in enumerate(action_seq):
            state_seq.append(state_seq[seq_idx] + action)
        data_states.append(np.array(state_seq))

    return data_states


def plot_trajectories(traj_data, n_demos, traj_len, traj_dim):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for demo in traj_data[:n_demos]:
        demo = np.reshape(demo, (traj_len, traj_dim))
        ax.scatter3D(demo[:, 1], demo[:, 2], demo[:, 3], c='g', s=20)
    for demo in traj_data[n_demos:]:
        demo = np.reshape(demo, (traj_len, traj_dim))
        ax.scatter3D(demo[:, 1], demo[:, 2], demo[:, 3], c='b', s=20)


# styles in data
h_vel_list = np.linspace(40, 100, 11)
h_dy_list = np.linspace(10, 30, 11)
h_styles = [hs for hs in product(h_vel_list, h_dy_list)]

# load training data
tasks = ['highway', 'intersection']
n_tasks = len(tasks)

raw_train_data = []
for task_name in tasks:
    task_data = pickle.load(open('data/' + task_name + '_trajectories.pkl', 'rb'))
    raw_train_data += task_data

pickle.dump(raw_train_data, open("data/dataset.pkl", "wb"))

# preprocess training data
raw_train_data = balance_data(raw_train_data)
max_len = min([31, min([len(demo) for demo in raw_train_data])])
raw_data_resample, raw_data_train = resample_data(raw_train_data, max_len)

# convert to action sequence
data_resample, data_train, task_start_states = states_to_actions(raw_data_resample, raw_data_train)
n_train_demos, seq_len, input_dim = np.shape(data_resample)
n_task_demos = int(n_train_demos/n_tasks)

# style extremes
extreme_styles = [es for es in product([min(h_vel_list), max(h_vel_list)],
                                       [min(h_dy_list), max(h_dy_list)])]
extreme_labels = [[extreme_styles.index(hs)] if hs in extreme_styles else [np.nan] for hs in h_styles]
extreme_labels *= n_tasks

# style labels
labels_train = [extreme_styles.index(hs) for hs in extreme_styles]*n_tasks

# Labeled data for training
labeled_data = [traj_data for data_idx, traj_data in enumerate(data_train) if not np.isnan(extreme_labels[data_idx][0])]

# Unlabeled data for training
unlabeled_data = [traj_data for data_idx, traj_data in enumerate(data_train) if np.isnan(extreme_labels[data_idx][0])]
n_task_demos_unlabeled = int(len(unlabeled_data)/n_tasks)
n_task_demos_split = 8
split_task1_idx = np.random.choice(np.arange(0, n_task_demos_unlabeled), n_task_demos_split)
split_task2_idx = np.random.choice(np.arange(n_task_demos_unlabeled, n_task_demos_unlabeled*2), n_task_demos_split)
split_data_task1 = [unlabeled_data[idx] for idx in split_task1_idx]
split_data_task2 = [unlabeled_data[idx] for idx in split_task2_idx]
dataset = split_data_task1 + split_data_task2 + labeled_data

# training data
XY_train = torch.FloatTensor(dataset)
labelset = torch.FloatTensor(labeled_data)
labels = torch.FloatTensor(labels_train).long()

# Initialize model
torch.manual_seed(0)
model = PECAN()

# Training parameters
EPOCH = 6000
BATCH_SIZE = 8
LR = 8e-4
train_data = MotionData(XY_train)
train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=LR)
losses = []

# Run training loop
for epoch in range(EPOCH):
    for batch, x in enumerate(train_set):

        loss_1 = model.loss_func(x, model.decoder(model.task_encode(x), model.style_encode(x)))
        loss_2 = model.cel_func(model.classifier(model.style_encode(labelset)), labels)
        loss = loss_1 + loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f'Epoch [{epoch + 1}/{EPOCH}], Loss: {loss.item():.10f}')

# plot training loss
plt.figure()
plt.plot(losses)
plt.show()

torch.manual_seed(0)
model.eval()

# test task encoder
ZT = model.task_encode(torch.FloatTensor(labelset)).detach().numpy()
print(np.round(ZT, decimals=2))

# test style encoder
ZS = model.style_encode(torch.FloatTensor(labelset)).detach().numpy()
print(np.round(ZS, decimals=3))

# save model
torch.save(model.state_dict(), "data/model_24")
print("Saved model.")

# save tasks embedding
task_embedding = [ZT[0], ZT[4]]
pickle.dump(task_embedding, open('data/task_embedding.pkl', 'wb'))
print(task_embedding)

print("Done.")
