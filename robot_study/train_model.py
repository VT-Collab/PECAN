import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import MyModel
import pickle
import matplotlib.pyplot as plt
import os

class MyData(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.FloatTensor(self.data[idx])

uid = input('Enter user id: ')

# training parameters
EPOCH = 15000
BATCH_SIZE = 8
LR = 8e-4
torch.manual_seed(0)

# dataset
dataset = pickle.load(open("data/dataset.pkl", "rb"))
train_data = MyData(dataset)
train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# labeled data and labels
labelset = pickle.load(open("data/labelset.pkl", "rb"))
labelset = torch.FloatTensor(labelset)
labels = torch.FloatTensor([0, 1]*3).long()

# model and optimizer
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# main training loop
losses = []
for epoch in range(EPOCH+1):
    for batch, x in enumerate(train_set):
        
        loss_1 = model.loss_func(x, model.decoder(model.task_encode(x), model.style_encode(x)))
        loss_2 = model.cel_func(model.classifier(model.style_encode(labelset)), labels)
        loss = loss_1 + loss_2

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
# plt.show()

# print latent tasks and styles
torch.manual_seed(0)
model.eval()
print("Tasks:")
print(model.task_encode(labelset))
print("Styles:")
print(model.style_encode(labelset))

# # plot latent styles
# X = torch.round(model.style_encode(torch.FloatTensor(dataset)), decimals=2).detach().numpy()
# plt.figure()
# plt.plot(X[:, 0], X[:, 1], "o")
# plt.show()

torch.save(model.state_dict(), "data/model.pt")

saveloc = 'data/user' + str(uid)

if not os.path.exists(saveloc):
    os.makedirs(saveloc)

torch.save(model.state_dict(), saveloc + '/model.pt')
