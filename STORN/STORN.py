# Necessary Packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator


class STORN_RNN(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.device = parameters['device']
        self.batch_size = parameters['batch_size']
        self.seq_len = parameters['seq_len']
        self.feature_dim = parameters['feature_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.num_layer = parameters['num_layer']
        # encoder
        self.encoder = nn.RNN(self.feature_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layer)
        self.mean_post = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.logvar_post = nn.Linear(self.hidden_dim, self.hidden_dim)
        # decoder
        self.decoder = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True, num_layers=self.num_layer)
        self.mean_xhat = nn.Linear(self.hidden_dim, self.feature_dim)
        self.logvar_xhat = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, x):
        y, h_en = self.encoder(x)   # y:(batch, seq_len, hidden_dim)
        mu_post = self.mean_post(y)     # mu_post:(batch, seq_len, hidden_dim)
        logsigma2_post = self.logvar_post(y)    # logsigma2_post:(batch, seq_len, hidden_dim)
        eps = torch.randn_like(logsigma2_post)
        std = torch.exp(logsigma2_post / 2)
        z = eps * std + mu_post     # z:(batch, seq_len, hidden_dim)
        xhat0 = torch.zeros(x.shape[0], 1, self.hidden_dim).to(self.device)
        h0_de = torch.zeros(1, x.shape[0], self.hidden_dim).to(self.device)
        outputs = torch.zeros(x.shape[0], self.seq_len, self.hidden_dim).to(self.device)
        for t in range(self.seq_len):
            xhat0 = torch.cat((xhat0, z[:, t, :].unsqueeze(1)), dim=2)
            xhat1, h1_de = self.decoder(xhat0, h0_de)
            xhat0, h0_de = xhat1, h1_de
            outputs[:, t, :] = xhat0.squeeze(1)
        mu_xhat = self.mean_xhat(outputs)
        logsigma2_xhat = self.logvar_xhat(outputs)
        return mu_post, logsigma2_post, mu_xhat, logsigma2_xhat

    def sample(self, z):
        xhat0 = torch.zeros(z.shape[0], 1, self.hidden_dim).to(self.device)
        h0_de = torch.zeros(1, z.shape[0], self.hidden_dim).to(self.device)
        outputs = torch.zeros(z.shape[0], self.seq_len, self.hidden_dim).to(self.device)
        for t in range(self.seq_len):
            xhat0 = torch.cat((xhat0, z[:, t, :].unsqueeze(1)), dim=2)
            xhat1, h1_de = self.decoder(xhat0, h0_de)
            xhat0, h0_de = xhat1, h1_de
            outputs[:, t, :] = xhat0.squeeze(1)
        mu_xhat = self.mean_xhat(outputs)
        logsigma2_xhat = self.logvar_xhat(outputs)
        return mu_xhat, logsigma2_xhat

class STORN_GRU(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.device = parameters['device']
        self.batch_size = parameters['batch_size']
        self.seq_len = parameters['seq_len']
        self.feature_dim = parameters['feature_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.num_layer = parameters['num_layer']
        # encoder
        self.encoder = nn.GRU(self.feature_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layer)
        self.mean_post = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.logvar_post = nn.Linear(self.hidden_dim, self.hidden_dim)
        # decoder
        # TODO: made the change here
        # self.decoder = nn.GRU(self.hidden_dim * 2, self.hidden_dim, batch_first=True, num_layers=self.num_layer)
        self.decoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layer)

        self.mean_xhat = nn.Linear(self.hidden_dim, self.feature_dim)
        self.logvar_xhat = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, x):
        y, h_en = self.encoder(x)                           # y:(batch, seq_len, hidden_dim)
        mu_post = self.mean_post(y)                         # mu_post:(batch, seq_len, hidden_dim)
        logsigma2_post = self.logvar_post(y)                # logsigma2_post:(batch, seq_len, hidden_dim)
        eps = torch.randn_like(logsigma2_post)
        std = torch.exp(logsigma2_post / 2)
        z = eps * std + mu_post     # z:(batch, seq_len, hidden_dim)
        # xhat0 = torch.zeros(x.shape[0], 1, self.hidden_dim).to(self.device)
        # h0_de = torch.zeros(self.num_layer, x.shape[0], self.hidden_dim).to(self.device)
        # outputs = torch.zeros(x.shape[0], self.seq_len, self.hidden_dim).to(self.device)
        # for t in range(self.seq_len):
        #     xhat0 = torch.cat((xhat0, z[:, t, :].unsqueeze(1)), dim=2)
        #     xhat1, h1_de = self.decoder(xhat0, h0_de)
        #     xhat0, h0_de = xhat1, h1_de
        #     outputs[:, t, :] = xhat0.squeeze(1)
        # TODO: made the change here
        outputs, _ = self.decoder(z)                        # outputs:(batch, seq_len, hidden_dim)
        mu_xhat = self.mean_xhat(outputs)                   # mu_xhat:(batch, seq_len, feature_dim)
        logsigma2_xhat = self.logvar_xhat(outputs)          # logsigma2_xhat:(batch, seq_len, feature_dim)
        return mu_post, logsigma2_post, mu_xhat, logsigma2_xhat

    def sample(self, z):
        # xhat0 = torch.zeros(z.shape[0], 1, self.hidden_dim).to(self.device)
        # h0_de = torch.zeros(self.num_layer, z.shape[0], self.hidden_dim).to(self.device)
        # outputs = torch.zeros(z.shape[0], self.seq_len, self.hidden_dim).to(self.device)
        # for t in range(self.seq_len):
        #     xhat0 = torch.cat((xhat0, z[:, t, :].unsqueeze(1)), dim=2)
        #     xhat1, h1_de = self.decoder(xhat0, h0_de)
        #     xhat0, h0_de = xhat1, h1_de
        #     outputs[:, t, :] = xhat0.squeeze(1)
        # TODO: made the change here
        outputs, _ = self.decoder(z)
        mu_xhat = self.mean_xhat(outputs)
        logsigma2_xhat = self.logvar_xhat(outputs)
        eps = torch.randn_like(logsigma2_xhat)
        std = torch.exp(logsigma2_xhat / 2)
        z = eps * std + mu_xhat
        return mu_xhat

def train(ori_data, parameters):
    """TimeGAN function.

    Use original PEMS-BAY as training set to generater synthetic PEMS-BAY (time-series)

    Args:
    - ori_data: original time-series PEMS-BAY
    - parameters: TimeGAN network parameters

    Returns:
    - generated_data: generated time-series PEMS-BAY
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    # Network Parameters
    device          = parameters['device']
    iterations      = parameters['iterations']
    batch_size      = parameters['batch_size']
    seq_len         = parameters['seq_len']
    fearture_dim    = parameters['feature_dim']
    hidden_dim      = parameters['hidden_dim']
    num_layer       = parameters['num_layer']
    learning_rate   = parameters['learning_rate']
    gamma           = parameters['gamma']

    ## Build a RNN network
    model = STORN_GRU(parameters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(iterations):
        model.train()
        X_batch, T_mb = batch_generator(ori_data, ori_time, batch_size)             # X_batch:(batch, seq_len, feature_dim)
        X_batch = torch.as_tensor(X_batch, dtype=torch.float, device=torch.device(device))
        mu_post, logsigma2_post, mu_xhat, logsigma2_xhat = model.forward(X_batch)
        # mu_post, sigma_post = mu_post.squeeze(0), torch.exp(logsigma2_post.squeeze(0) / 2)
        # mu_xhat, sigma_xhat = mu_xhat.squeeze(0), torch.exp(logsigma2_xhat.squeeze(0) / 2)
        # loss_negloglike = Normal(mu_xhat, sigma_xhat).log_prob(X_batch).sum()

        # loss_negloglike = loss_negloglike * (-1.0 / batch_size)
        # latent_features.shape[0] is the batch_size

        # mu_prior, sigma_prior = self.factor_predictor(latent_features)
        # m_encoder = Normal(mu_post, sigma_post)
        # m_predictor = Normal(0, 1)

        #  loss_KL = kl_divergence(m_encoder, m_predictor).sum() / batch_size
        # TODO: made the change here
        loss_KL = torch.mean(0.5 * torch.sum(-1 - logsigma2_post + mu_post.pow(2) + logsigma2_post.exp(), dim=[-1, -2]), dim=0)
        loss_negloglike = torch.mean(torch.sum((mu_xhat - X_batch.clone()) ** 2, dim=[-1, -2]), dim=0)
        # loss_negloglike = torch.mean(0.5 * torch.sum((torch.div((mu_xhat - X_batch.clone()), torch.exp(logsigma2_xhat / 2)) ** 2), dim=[-1, -2]) + 0.5 * torch.sum(logsigma2_xhat, dim=[-1, -2]), dim=0)

        # if i >= (iterations / 2):
        #     gamma_now = gamma * (i - iterations / 2) / (iterations / 2)
        # else:
        #     gamma_now = 0
        loss = loss_negloglike + gamma * loss_KL

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'epoch:{i}, loss:{loss}, loss_negloglike:{loss_negloglike}, loss_kl:{loss_KL}')

    ## Synthetic PEMS-BAY generation
    Z_mb = np.random.normal(0., 1, [no, seq_len, hidden_dim])                       # Z_mb:(batch, seq_len, hidden_dim)
    # Z_mb = random_generator(no, fearture_dim, ori_time, max_seq_len)
    Z_mb = torch.as_tensor(Z_mb, dtype=torch.float, device=torch.device(device))
    # generated_data_curr, _ = model.sample(Z_mb)
    generated_data_curr = model.sample(Z_mb)                                        # generated_data_curr:(batch, seq_len, feature_dim)
    generated_data_curr = generated_data_curr.cpu().detach().numpy()

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    # max_val = np.max(np.max(ori_data, axis=0), axis=0)
    # min_val = np.min(np.min(ori_data, axis=0), axis=0)
    # generated_data = generated_data * max_val
    # generated_data = generated_data + min_val

    return generated_data
