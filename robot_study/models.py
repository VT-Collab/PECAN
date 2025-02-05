import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# OURS ----------------------------------------------------------------------------------------------------------------


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # task encoder
        self.tau_enc_1 = nn.Linear(120, 64)
        self.tau_enc_2 = nn.Linear(64, 32)
        self.tau_enc_3 = nn.Linear(32, 3)

        # style encoder
        self.a_enc_1 = nn.Linear(120, 64)
        self.a_enc_2 = nn.Linear(64, 32)
        self.a_enc_3 = nn.Linear(32, 1)

        # decoder
        self.dec_1 = nn.Linear(3 + 1, 32)
        self.dec_2 = nn.Linear(32, 64)
        self.dec_3 = nn.Linear(64, 120)

        # classifier
        self.linear1 = nn.Linear(1, 2)

        # other stuff
        self.m = nn.ReLU()
        self.apply(weights_init_)
        self.loss_func = nn.MSELoss()
        self.cel_func = nn.CrossEntropyLoss()

    def task_encode(self, tau):
        x = torch.tanh(self.tau_enc_1(tau))
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


# BASELINE ------------------------------------------------------------------------------------------------------------

def get_labels_mask(tensor_labels):
    """
    Converts one-hot labels to boolean masks.
    True for any class label and False for no class label.
    """
    one = torch.tensor(1, dtype=torch.float32)
    labels_sum = torch.sum(tensor_labels, dim=-1)
    labels_mask = torch.eq(one, labels_sum)
    labels_mask = labels_mask.type(torch.bool)
    return labels_mask


def get_gaussians(z_dim, init, labeled_data, gauss_num):
    """
    Initializes the gaussian mixture model (GMM).
    Creates parameter for means of GMM to pass gradients.
    """
    # initialize the means of the classes
    if gauss_num <= z_dim:
        one_hot = torch.zeros(gauss_num, z_dim)
        one_hot[torch.arange(z_dim) % gauss_num, torch.arange(z_dim)] = 1
        one_hot *= init / z_dim * gauss_num
        one_hot += torch.normal(0, .001, size=one_hot.shape)
    else:
        one_hot = torch.normal(0, init, size=(gauss_num, z_dim))

    if gauss_num == 1:
        means_initializer = torch.zeros([1, z_dim])
    else:
        means_initializer = torch.tensor(one_hot)

    # set means as a parameter for updating
    means = nn.Parameter(means_initializer, requires_grad=True)

    # initialize the variance
    beta = torch.zeros(gauss_num)
    variances = 1.0 + torch.abs(beta)

    # compute class probabilities
    labeled_data = torch.FloatTensor(labeled_data)
    labels_proportions = torch.sum(labeled_data, dim=0) / torch.sum(labeled_data)
    probs = labels_proportions.float().reshape(-1)
    if gauss_num == 1:
        probs = torch.tensor([1.])

    return means, variances, probs


def calculate_logits(tensor_z, means, variance, p):
    """
    Compute likelihood of z for gaussians centered at the GMM means.
    """
    D = means.size(-1)
    diffs = tensor_z.unsqueeze(1) - means.unsqueeze(0)
    exponents = -torch.norm(diffs, dim=-1).pow(2) / (2 * variance)
    class_logits = torch.log(p) - (0.5 * D * torch.log(2 * math.pi * variance)) + exponents
    return class_logits


def calculate_logits_cost(class_logits, tensor_target, labeled_mask):
    """
    Cross-entropy loss for classifying z into GMM classes.
    """
    class_cost = F.cross_entropy(class_logits, tensor_target, reduction='none')
    denominator = tensor_target.sum().item()
    denominator = 1.0 if denominator == 0 else denominator
    casted_mask = labeled_mask.float()
    class_cost_fin = (class_cost * casted_mask).sum() / denominator
    return class_cost_fin


def cramer_wold_distance(X, mu, variance, p, gamma):
    """
    Cramer-Wold distance between latent samples and mixture of gaussians.
    """
    m = X.size(0)
    K = X.size(1)

    variance_matrix = variance.unsqueeze(0) + variance.unsqueeze(1)  # TODO: Check variance matrix dimensions

    X_sub_matrix = X.unsqueeze(0) - X.unsqueeze(1)
    A1 = torch.norm(X_sub_matrix, dim=2)**2
    A1 = torch.sum(phi_d(A1 / (4 * gamma), K))
    A = 1/(m*m * math.sqrt(2 * math.pi * 2 * gamma)) * A1

    mu_X_sub_matrix = mu.unsqueeze(0) - X.unsqueeze(1)
    B1 = torch.norm(mu_X_sub_matrix, dim=2) ** 2
    B2 = phi_d(B1 / (2 * (variance + (2 * gamma))), K)
    B3 = 2 * p / (m * torch.sqrt(2 * math.pi * (variance + (2 * gamma)))) * B2
    B = torch.sum(B3)

    mu_sub_matrix = mu.unsqueeze(0) - mu.unsqueeze(1)
    C1 = torch.norm(mu_sub_matrix, dim=2)**2
    C2 = phi_d(C1 / (2 * variance_matrix + 4 * gamma), K)
    p_mul_matrix = torch.matmul(p.unsqueeze(1), p.unsqueeze(0))
    C3 = p_mul_matrix / torch.sqrt(2 * math.pi * (variance_matrix + 2 * gamma))
    C = torch.sum(C3 * C2)

    return torch.mean(A - B + C)


def phi_d(s, D):
    if D <= 2:
        return phi(s)
    else:
        return 1 / torch.sqrt(1 + (4 * s) / (2 * D - 3))


def phi(x):
    a = torch.FloatTensor([7.5])
    return phi_f(torch.minimum(x, a)) - phi_f(a) + phi_g(torch.maximum(x, a))


def phi_f(s):
    t = s/7.5
    return torch.exp(-s/2) * (
            1 + 3.5156229*t**2 + 3.0899424*t**4 + 1.2067492*t**6
            + 0.2659732*t**8 + 0.0360768*t**10 + 0.0045813*t**12)


def phi_g(s):
    t = s/7.5
    return torch.sqrt(2/s) * (
            0.39894228 + 0.01328592*t**(-1) + 0.00225319*t**(-2)
            - 0.00157565*t**(-3) + 0.0091628*t**(-4) - 0.02057706*t**(-5)
            + 0.02635537*t**(-6) - 0.01647633*t**(-7) + 0.00392377*t**(-8))


class SeGMA(nn.Module):
    """
    Semi-supervised Gaussian Mixture Autoencoder (SeGMA).
    """

    def __init__(self, input_dim, classes_num, latent_dim, labels=None):
        super(SeGMA, self).__init__()

        # inputs
        self.input_dim = input_dim
        self.classes_num = torch.tensor(classes_num)
        self.latent_dim = latent_dim

        # constants
        self.init = 2.0
        self.cw_weight = 1.0
        self.supervised_weight = 2.0

        # encoder
        self.enc_layer1 = nn.Linear(input_dim, 32)
        self.enc_layer2 = nn.Linear(32, 16)
        self.enc_layer3 = nn.Linear(16, latent_dim)

        # decoder
        self.dec_layer1 = nn.Linear(latent_dim, 16)
        self.dec_layer2 = nn.Linear(16, 32)
        self.dec_layer3 = nn.Linear(32, input_dim)

        # GMM
        self.means, self.variances, self.probs = get_gaussians(latent_dim, self.init, labels, classes_num)

        self.apply(weights_init_)
        self.criterion = nn.MSELoss()

    def traj_encoder(self, tau):
        x = F.tanh(self.enc_layer1(tau))
        x = F.tanh(self.enc_layer2(x))
        x_mean = self.enc_layer3(x)

        return x_mean

    def traj_decoder(self, z):
        x = F.tanh(self.dec_layer1(z))
        x = F.tanh(self.dec_layer2(x))
        tau = self.dec_layer3(x)

        return tau

    def forward(self, batch_data, train_labeled=True, out=False):

        # separate data
        tau = batch_data[:, :self.input_dim]
        labels = batch_data[:, self.input_dim:]

        # encode tau --> z
        z = self.traj_encoder(tau)

        # decode z --> tau
        tau_decoded = self.traj_decoder(z)

        # reconstruction loss
        rec_loss = self.criterion(tau, tau_decoded)

        # mask labeled data
        labeled_mask = get_labels_mask(labels)

        # probability P(z, y)
        class_logits = calculate_logits(z, self.means, self.variances, self.probs)
        class_probs = F.softmax(class_logits, dim=-1)

        # classification loss
        class_loss = calculate_logits_cost(class_logits, labels, labeled_mask)

        # number of data points
        unsupervised_z = z if train_labeled else z[~labeled_mask]
        n0 = unsupervised_z.size(0)

        # silverman's rule of thumb for gaussian kernels
        gamma = torch.pow(4 / (3 * torch.tensor(n0)), 0.2).float()

        # cramer-wold distance
        cw_cost = cramer_wold_distance(unsupervised_z, self.means, self.variances, self.probs, gamma)
        log_cw_cost = 8.0 + torch.log(cw_cost)  # TODO: Remove constant?

        # total loss
        total_loss = rec_loss + (self.cw_weight * log_cw_cost) + (self.supervised_weight * class_loss)

        if out:
            return total_loss, self.means
        else:
            return total_loss
