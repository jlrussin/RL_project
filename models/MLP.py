"""
Multi-layer perceptron for learning successor features
Everything in this file was written by us.
"""
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embedding_size, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_size,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,embedding_size)
        self.relu = nn.ReLU()
    def forward(self, embedding):
        embedding = self.fc1(embedding)
        embedding = self.relu(embedding)
        embedding = self.fc2(embedding)
        embedding = self.relu(embedding)
        Mhat = self.fc3(embedding)
        return Mhat
