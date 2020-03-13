#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors.kd_tree import KDTree


class QEC:
    def __init__(self, actions, max_memory, num_neighbors, use_Q_max,
                 force_knn, weight_neighbors, delta, q_lr):
        self.actions = actions
        self.max_memory = max_memory
        self.num_neighbors = num_neighbors
        self.use_Q_max = use_Q_max
        self.force_knn = force_knn
        self.weight_neighbors = weight_neighbors
        self.delta = delta
        self.q_lr = q_lr
        self.buffers = tuple([ActionBuffer(max_memory,delta) for _ in actions])
        self.knn_usage = []
        self.replace_usage = []

    def estimate(self, state, action):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index and not self.force_knn:
            self.knn_usage.append(0)
            return buffer.values[state_index]
        else:
            self.knn_usage.append(1)
        if len(buffer) <= self.num_neighbors:
            return float("inf")

        neighbors,weights = buffer.find_neighbors(state, self.num_neighbors)
        if not self.weight_neighbors:
            weights = np.ones_like(weights)/self.num_neighbors
        value = 0.0
        for neighbor,weight in zip(neighbors[0],weights[0]):
            value += weight*buffer.values[neighbor]
        return value

    def update(self, state, action, value, time):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            self.replace_usage.append(1)
            if self.use_Q_max:
                new_value = max(buffer.values[state_index], value)
            else:
                q = buffer.values[state_index]
                new_value = q + self.q_lr*(value - q)
                new_time = time
            new_time = max(buffer.times[state_index], time)
            buffer.replace(state, new_value, new_time, state_index)
        else:
            self.replace_usage.append(0)
            buffer.add(state, value, time)


class ActionBuffer:
    def __init__(self, capacity,delta):
        self._tree = None
        self.capacity = capacity
        self.delta = delta
        self.states = []
        self.values = []
        self.times = []

    def find_state(self, state):
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k):
        if not self._tree:
            return [], []
        else:
            distances,neighbors = self._tree.query([state], k=k,
                                                  return_distance=True)
            weights = self.get_weights(distances)
        return neighbors, weights

    def get_weights(self,distances):
        distances = distances / (np.sum(distances) + 1e-8) # normalize for stability
        similarities = 1 / (distances+self.delta)
        weights = similarities/np.sum(similarities)
        return weights

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(np.array(self.states))

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)
