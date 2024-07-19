import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

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


class ClearAE(nn.Module):

    def __init__(self, input_dim, latent_dim, timesteps=10):
        super(ClearAE, self).__init__()

        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim*timesteps, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2))

        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim*timesteps, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.Tanh())

        self.traj_decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.Tanh(),
            nn.Linear(32, input_dim*timesteps))

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 2**latent_dim))

        self.apply(weights_init_)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss()

    def forward(self, x, y=torch.FloatTensor([np.NAN])):

        # data
        tau = x

        # trajectory
        zt = nn.functional.gumbel_softmax(self.task_encoder(tau), tau=1, hard=True)

        # style
        zs = self.style_encoder(tau)

        # decoders
        zs_zt = torch.cat((zs, zt), 1)
        tau_decoded2 = self.traj_decoder(zs_zt)

        # reconstruction loss
        loss1 = self.criterion1(tau, tau_decoded2)

        if any(~torch.isnan(y)):
            # label data
            tau_label = x[~torch.isnan(y)]
            target = y[~torch.isnan(y)].to(torch.long)

            # label style
            zs_label = self.style_encoder(tau_label)

            # classification loss
            loss2 = self.criterion2(self.classifier(zs_label), target)
        else:
            loss2 = 0.

        total_loss = loss1 + loss2

        return total_loss
