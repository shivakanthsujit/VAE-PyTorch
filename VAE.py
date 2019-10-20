import torch
from torch import nn, optim
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encodinglayer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU()
        )
        self.encodinglayer2_mean = nn.Sequential(nn.Linear(hidden_size, latent_size))
        self.encodinglayer2_logvar = nn.Sequential(nn.Linear(hidden_size, latent_size))
        self.decodinglayer = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.encodinglayer1(x)
        log_var = self.encodinglayer2_logvar(x)
        mean = self.encodinglayer2_mean(x)

        z = self.sample(log_var, mean)
        x = self.decodinglayer(z)

        return x, mean, log_var

