import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class PECAN(nn.Module):
    def __init__(self):
        super(PECAN, self).__init__()

        # task encoder
        self.tau_enc_1 = nn.Linear(120, 64)
        self.tau_enc_2 = nn.Linear(64, 16)
        self.tau_enc_3 = nn.Linear(16, 2)

        # style encoder
        self.a_enc_1 = nn.Linear(120, 64)
        self.a_enc_2 = nn.Linear(64, 16)
        self.a_enc_3 = nn.Linear(16, 2)

        # decoder
        self.dec_1 = nn.Linear(2 + 2, 16)
        self.dec_2 = nn.Linear(16, 64)
        self.dec_3 = nn.Linear(64, 120)

        # classifier
        self.linear1 = nn.Linear(2, 4)

        # other stuff
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.loss_func = nn.MSELoss()
        self.cel_func = nn.CrossEntropyLoss()

    def task_encode(self, tau):
        x = torch.relu(self.tau_enc_1(tau))
        x = torch.tanh(self.tau_enc_2(x))
        return F.gumbel_softmax(self.tau_enc_3(x), tau=1., hard=True)

    def style_encode(self, tau):
        x = torch.tanh(self.a_enc_1(tau))
        x = torch.tanh(self.a_enc_2(x))
        return torch.tanh(self.a_enc_3(x))

    def decoder(self, z_task, z_style):
        ztask_zstyle = torch.cat((z_task, z_style), 1)
        x = torch.tanh(self.dec_1(ztask_zstyle))
        x = torch.tanh(self.dec_2(x))
        return self.dec_3(x)

    def classifier(self, z_style):
        return self.linear1(z_style)
