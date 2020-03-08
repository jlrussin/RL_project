# Autoencoder for state embedding
# Code adapted from:
# https://github.com/pytorch/examples/blob/master/vae/main.py

import torch
import torch.nn as nn

class VAE_Encoder(nn.Module):
    def __init__(self,in_channels,embedding_size,in_height,in_width):
        super(VAE_Encoder, self).__init__()
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.in_height = in_height
        self.in_width = in_width
        self.conv1 = nn.Conv2d(in_channels,32,kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(32,32,kernel_size=5,stride=2)
        self.conv3 = nn.Conv2d(32,64,kernel_size=5,stride=2)
        self.conv4 = nn.Conv2d(64,64,kernel_size=4,stride=2)
        fc1_in_channels = calculate_FC_in(self,in_height,in_width)
        self.fc = nn.Linear(fc1_in_channels,512)
        self.fc_mu = nn.Linear(512,self.embedding_size // 2)
        self.fc_logvar = nn.Linear(512,self.embedding_size // 2)
        self.relu = nn.ReLU()
    def calculate_FC_in(self,H,W):
        def conv2d_out_shape(H_in,W_in,kernel_size,stride):
            H_out = int((H_in + 2*0 - 1*(kernel_size - 1) - 1)/stride) + 1
            W_out = int((W_in + 2*0 - 1*(kernel_size - 1) - 1)/stride) + 1
            return (H_out,W_out)
        H,W = conv2d_out_shape(H,W,4,2)
        H,W = conv2d_out_shape(H,W,5,2)
        H,W = conv2d_out_shape(H,W,5,2)
        H,W = conv2d_out_shape(H,W,4,2)
        fc1_in_channels = H*W*64
        return fc1_in_channels
    def forward(self,state):
        N = state.size(0) #batch size
        hidden = self.relu(self.conv1(state))
        hidden = self.relu(self.conv2(hidden))
        hidden = self.relu(self.conv3(hidden))
        hidden = self.relu(self.conv4(hidden))
        hidden = self.relu(self.fc(hidden.view(N,-1)))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar


class VAE_Decoder(nn.Module):
    def __init__(self,in_channels,embedding_size):
        super(VAE_Decoder, self).__init__()
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(embedding_size//2,512)
        self.fc2 = nn.Linear(512,576)
        self.convt1 = nn.ConvTranspose2d(64,64,4,2)
        self.convt2 = nn.ConvTranspose2d(64,32,5,2)
        self.convt3 = nn.ConvTranspose2d(32,32,5,2)
        self.convt4 = nn.ConvTranspose2d(32,in_channels,4,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,z):
        N = z.size(0) #batch size
        hidden = self.relu(self.fc1(z))
        hidden = self.relu(self.fc2(hidden))
        hidden = hidden.reshape(N,64,3,3)
        hidden = self.relu(self.convt1(hidden))
        hidden = self.relu(self.convt2(hidden))
        hidden = self.relu(self.convt3(hidden))
        recon_batch = self.sigmoid(self.convt4(hidden))
        return recon_batch


class VAE(nn.Module):
    def __init__(self,in_channels,embedding_size):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.encoder = VAE_Encoder(in_channels,embedding_size)
        self.decoder = VAE_Decoder(in_channels,embedding_size)
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self,state):
        mu, logvar = self.encoder(state)
        z = self.reparameterize(mu,logvar)
        recon_batch = self.decoder(z)
        return recon_batch, mu, logvar

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss,self).__init__()
        self.bce = nn.BCELoss(reduction='sum')

    def forward(self, recon_x, x, mu, logvar):
        BCE = self.bce(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
