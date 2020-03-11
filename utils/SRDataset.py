import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class SRDataset(Dataset):
    def __init__(self,env,projection,transitions):
        self.env = env
        self.projection = projection
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self,idx):
        transition = self.transitions[idx]
        s_t,s_tp1 = transition
        o_t = self.env.state_dict[s_t]
        o_tp1 = self.env.state_dict[s_tp1]
        e_t = np.dot(self.projection,o_t.flatten())
        e_tp1 = np.dot(self.projection,o_tp1.flatten())
        e_t = torch.tensor(e_t)
        e_tp1 = torch.tensor(e_tp1)
        return e_t, e_tp1
