# CNN for state embedding: hyperparameters taken from Mnih et al. (2015)

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,embedding_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.fc = nn.Linear(3136,512)
        self.out = nn.Linear(512,embedding_size)
        self.relu = nn.ReLU()
    def forward(self,observation):
        N = observation.size(0) # batch size
        embedding = self.relu(self.conv1(observation))
        embedding = self.relu(self.conv2(embedding))
        embedding = self.relu(self.conv3(embedding))
        embedding = self.relu(self.fc(embedding.view(N,-1)))
        embedding = self.relu(self.out(embedding))
        return embedding
