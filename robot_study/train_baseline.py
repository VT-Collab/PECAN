import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
import pickle
import matplotlib.pyplot as plt
import os

class MyData(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

uid = input('Enter user id: ')

# training parameters
EPOCH = 8000
BATCH_SIZE = 9
LR = 2e-3
torch.manual_seed(0)

# dataset
dataset = pickle.load(open("data/dataset.pkl", "rb"))
n_demos, input_dim = np.shape(dataset)

# labeled data
labelset = pickle.load(open("data/labelset.pkl", "rb"))
n_labels, _ = np.shape(labelset)

# labels
labels = np.array([[0, 0, 0]]*(n_demos - n_labels) + [[0, 0, 1]]*2 + [[0, 1, 0]]*2 + [[1, 0, 0]]*2)

# model and optimizer
task_dim, style_dim = 3, 1
model = SeGMA(input_dim, task_dim, style_dim, labels)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# data loader
combinedset = np.hstack((dataset, labels))
train_data = MyData(combinedset)
train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# main training loop
losses = []
for epoch in range(EPOCH+1):
    for batch, xy in enumerate(train_set):

        loss, gmm_means = model.forward(xy, out=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    if epoch % 500 == 0:
        print(epoch, loss.item())
        # print(model.classifier(model.style_encode(labelset)))

# # plot training loss
# plt.figure()
# plt.plot(losses)
# plt.yscale('log')
# plt.show()

torch.save(model.state_dict(), "data/model_baseline.pt")

saveloc = 'data/user' + str(uid)

if not os.path.exists(saveloc):
    os.makedirs(saveloc)

torch.save(model.state_dict(), saveloc + '/model_baseline.pt')

# test model
model.load_state_dict(torch.load("data/model_baseline.pt"))
torch.manual_seed(0)
model.eval()

print("Tasks:")
zs = model.traj_encoder(torch.FloatTensor(labelset))
task_logits = calculate_logits(zs, model.means, model.variances, model.probs)
task_probs = F.softmax(task_logits, dim=-1)
task_predict = np.argmax(task_probs.detach().numpy(), axis=1)
print(task_predict)

print("Styles:")
task_means = model.means.detach().numpy()
task_agnostic_zs = zs.detach().numpy() - task_means[task_predict, :]
print(task_agnostic_zs)

traj_acc = model.criterion(torch.FloatTensor(labelset), model.traj_decoder(zs))
print("Reconstruction accuracy:", traj_acc)

# # plot latent styles
# X = torch.round(task_agnostic_zs), decimals=2).detach().numpy()
# plt.figure()
# plt.plot(X, "o")
# plt.show()

