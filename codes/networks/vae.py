import torch
from torch import nn
from torch.nn import functional as F
from typing import List


# convtranspose2d 다음에 BatchNorm2d를 넣어야 하는가?
# relu 대신 LeakyReLU를 써야 하는가?
class Encoder(nn.Module):
    def __init__(self, channels, latent_size, use_batch_norm=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        
        self.layers = []
        for i in range(len(channels) - 1):
            if i == len(channels) - 2:
                self.layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, stride=1, padding=0))
            else:
                self.layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, stride=1, padding=1))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm2d(channels[i + 1]))
            self.layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*self.layers)
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(channels[-1], latent_size)
        self.fc_logsigma = nn.Linear(channels[-1], latent_size)
        
    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        
        return mu, logsigma
    

class Decoder(nn.Module):
    def __init__(self, channels, latent_size, use_batch_norm=False):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        
        channels.reverse()
        
        self.linear = nn.Linear(latent_size, channels[0])
        
        self.layers = []
        num_layers = len(channels) - 1
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 3, stride=2, padding=1))
            else:
                self.layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 3, stride=2))
            
            if i < num_layers - 1:
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm2d(channels[i + 1]))
                self.layers.append(nn.LeakyReLU(inplace=True))
            else:
                self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 1024, 1, 1)
        out = self.layers(x)
        return out


class VAE(nn.Module):
    def __init__(self, in_channels, latent_size, use_batch_norm=False):
        super(VAE, self).__init__()
        
        channels = [in_channels, 32, 64, 128, 256, 1024]
        
        self.encoder = Encoder(channels, latent_size, use_batch_norm)
        self.decoder = Decoder(channels, latent_size, use_batch_norm)

    def encode(self, x):
        mu, log_var = self.encoder(x)
        z = torch.randn_like(log_var) * torch.exp(log_var * 0.5) + mu

        return z, mu, log_var
    
    def decode(self, x):
        recon_x = self.decoder(x)
        return recon_x
        
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, log_var
