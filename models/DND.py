# This code is taken from: https://github.com/mjacar/pytorch-nec/blob/master/dnd.py

import torch
from torch.nn import Parameter
from pyflann import FLANN

from utils.utils import get_optimizer

class DND:
    def __init__(self, kernel, num_neighbors, max_memory, optimizer, lr):
        self.kernel = kernel
        self.num_neighbors = num_neighbors
        self.max_memory = max_memory
        self.opt_name = optimizer
        self.lr = lr
        self.keys = None
        self.values = None
        self.kdtree = FLANN()

        # key_cache stores a cache of all keys that exist in the DND
        # This makes DND updates efficient
        self.key_cache = {}
        # stale_index indicates whether or not the index in self.kdtree is stale
        # This allows us to only rebuild the kdtree index when necessary
        self.stale_index = True
        # indexes_to_be_updated will be updated on a call to update_params
        # This allows us to only rebuild the necessary keys of key_cache
        self.indexes_to_be_updated = set()

        # Keys and values to be inserted into self.keys and self.values
        # when commit_insert is called
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None

        # Recently used lookup indexes
        # Moved to the back of self.keys and self.values to get LRU property
        self.move_to_back = set()

    def get_index(self, key):
        """
        If key exists in the DND, return its index
        Otherwise, return None
        """
        key = key.detach().cpu().numpy()
        if self.key_cache.get(tuple(key[0])) is not None:
            if self.stale_index:
                self.commit_insert()
            return int(self.kdtree.nn_index(key, 1)[0][0])
        else:
            return None

    def update(self, value, index):
        """
        Set self.values[index] = value
        """
        values = self.values.detach()
        values[index] = value[0].detach()
        self.values = Parameter(values)
        params = [self.keys, self.values]
        self.optimizer = get_optimizer(self.opt_name,params,self.lr)

    def insert(self, key, value):
        """
        Insert key, value pair into DND
        """
        if self.keys_to_be_inserted is None:
            # Initial insert
            self.keys_to_be_inserted = key.detach()
            self.values_to_be_inserted = value.detach()
        else:
            self.keys_to_be_inserted = torch.cat(
                [self.keys_to_be_inserted, key.detach()], 0)
            self.values_to_be_inserted = torch.cat(
                [self.values_to_be_inserted, value.detach()], 0)
        self.key_cache[tuple(key.detach().cpu().numpy()[0])] = 0
        self.stale_index = True

    def commit_insert(self):
        if self.keys is None:
            self.keys = Parameter(self.keys_to_be_inserted)
            self.values = Parameter(self.values_to_be_inserted)
        elif self.keys_to_be_inserted is not None:
            keys = torch.cat([self.keys.detach(),self.keys_to_be_inserted],0)
            self.keys = Parameter(keys)
            values = [self.values.detach(),self.values_to_be_inserted]
            values = torch.cat(values,0)
            self.values = Parameter(values)

        # Move most recently used key-value pairs to the back
        if len(self.move_to_back) != 0:
            unmoved_ids = list(set(range(len(self.keys))) - self.move_to_back)
            moved_ids = list(self.move_to_back)
            unmoved_keys = self.keys.detach()[unmoved_ids]
            moved_keys = self.keys.detach()[moved_ids]
            self.keys = Parameter(torch.cat([unmoved_keys, moved_keys], 0))
            unmoved_values = self.values.detach()[unmoved_ids]
            moved_values = self.values.detach()[moved_ids]
            self.values = Parameter(torch.cat([unmoved_values,moved_values], 0))
            self.move_to_back = set()

        if len(self.keys) > self.max_memory:
            # Expel oldest key to maintain total memory
            for key in self.keys[:-self.max_memory]:
                del self.key_cache[tuple(key.detach().cpu().numpy())]
            self.keys = Parameter(self.keys[-self.max_memory:].detach())
            self.values = Parameter(self.values[-self.max_memory:].detach())
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None
        params = [self.keys, self.values]
        self.optimizer = get_optimizer(self.opt_name,params,self.lr)
        self.kdtree.build_index(self.keys.detach().cpu().numpy())
        self.stale_index = False

    def lookup(self, lookup_key, update_flag=False):
        """
        Perform DND lookup
        if update_flag:
            add the nearest neighbor indexes to self.indexes_to_be_updated
        """
        lookup_key_np = lookup_key.detach().cpu().numpy()
        num_neighbors = min(self.num_neighbors, len(self.keys))
        lookup_indexes = self.kdtree.nn_index(lookup_key_np,num_neighbors)[0][0]
        output = 0
        kernel_sum = 0
        for i, index in enumerate(lookup_indexes):
            # Skip keys exactly equal to lookup_key (loss non-differentiable)
            if i == 0 and tuple(lookup_key_np[0]) in self.key_cache:
                continue
            if update_flag:
                self.indexes_to_be_updated.add(int(index))
            else:
                self.move_to_back.add(int(index))
            kernel_val = self.kernel(self.keys[int(index)], lookup_key[0])
            output += kernel_val * self.values[int(index)]
            kernel_sum += kernel_val
        output = output / kernel_sum
        return output

    def update_params(self):
        """
        Update self.keys and self.values via backprop
        Use self.indexes_to_be_updated to update self.key_cache accordingly
        Rebuild the index of self.kdtree
        """
        for index in self.indexes_to_be_updated:
            del self.key_cache[tuple(self.keys[index].detach().cpu().numpy())]
        self.optimizer.step()
        self.optimizer.zero_grad()
        for index in self.indexes_to_be_updated:
            self.key_cache[tuple(self.keys[index].detach().cpu().numpy())] = 0
        self.indexes_to_be_updated = set()
        self.kdtree.build_index(self.keys.detach().cpu().numpy())
        self.stale_index = False
