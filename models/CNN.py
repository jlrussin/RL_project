"""
CNN for state embedding.
Everything in this file was written by us.
Hyperparameters taken from Mnih et al. (2015)
"""

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,in_channels,embedding_size,in_height,in_width):
        super(CNN, self).__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.conv1 = nn.Conv2d(in_channels,32,kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        fc1_in_channels = self.calculate_FC_in(in_height,in_width)
        self.fc = nn.Linear(fc1_in_channels,512)
        self.out = nn.Linear(512,embedding_size)
        self.relu = nn.ReLU()

    def calculate_FC_in(self,H,W):
        def conv2d_out_shape(H_in,W_in,kernel_size,stride):
            H_out = int((H_in + 2*0 - 1*(kernel_size - 1) - 1)/stride) + 1
            W_out = int((W_in + 2*0 - 1*(kernel_size - 1) - 1)/stride) + 1
            return (H_out,W_out)
        H,W = conv2d_out_shape(H,W,4,2)
        H,W = conv2d_out_shape(H,W,4,2)
        H,W = conv2d_out_shape(H,W,3,1)
        fc1_in_channels = H*W*64
        return fc1_in_channels

    def forward(self,observation):
        N = observation.size(0) # batch size
        embedding = self.relu(self.conv1(observation))
        embedding = self.relu(self.conv2(embedding))
        embedding = self.relu(self.conv3(embedding))
        embedding = self.relu(self.fc(embedding.view(N,-1)))
        embedding = self.relu(self.out(embedding))
        return embedding
