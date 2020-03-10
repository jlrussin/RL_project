import numpy as np
import random
import matplotlib.pyplot as plt
import gzip
import networkx as nx

class ActionSpace():
    def __init__(self):
        self.n = 4

class FourRooms():
    def __init__(self,room_size,state_type='tabular'):
        assert room_size % 2 == 1, "room_size must be odd"
        self.room_size = room_size
        self.goal = (1,1) # tuple (row,col)
        self.state_type = state_type

        if self.state_type == 'mnist':
            mnist_path = '../data/mnist/train-images-idx3-ubyte.gz'
            with gzip.open(mnist_path) as f:
                # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
                mnist = np.frombuffer(f.read(), 'B', offset=16)
                mnist = mnist.reshape(-1,28,28,1).astype('float32') / 255

        self.n_states = 4*room_size**2 + 4
        self.action_space = ActionSpace()
        self.action_space.n = 4 # number of actions (up,down,left,right)
        self.skip = 1 # no frames are skipped
        # Build array with four rooms, four hallways
        n_side = room_size*2 + 3
        middle = room_size + 1
        self.rooms_array = np.zeros([n_side,n_side])
        self.rooms_array[1:middle,1:middle] = 1 # top left room
        self.rooms_array[1:middle,middle+1:middle+1+room_size] = 1 # top right room
        self.rooms_array[middle+1:-1,1:middle] = 1 # bottom left room
        self.rooms_array[middle+1:-1,middle+1:middle+1+room_size] = 1 # bottom right room
        self.rooms_array[middle,1+room_size//2] = 1 # left hallway
        self.rooms_array[middle,1+middle+room_size//2] = 1 # right hallway
        self.rooms_array[1+room_size//2,middle] = 1 # top hallway
        self.rooms_array[1+middle+room_size//2,middle] = 1 # bottom hallway

        # Set up state dict
        self.state_dict = {}
        counter = 0
        for row in range(len(self.rooms_array)):
            for col in range(len(self.rooms_array)):
                if self.rooms_array[row,col] == 1:
                    if self.state_type == 'tabular':
                        observation = np.zeros([28,28,1])
                        im_index = np.unravel_index(counter,(28,28))
                        observation[im_index] = 1
                        observation = observation.astype(np.float32)
                        self.state_dict[(row,col)] = observation
                    elif self.state_type == 'mnist':
                        observation = mnist[counter,:,:].astype(np.float32)
                        self.state_dict[(row,col)] = observation
                    counter += 1
        # Initial state
        self.random_start()

        # Set up graph representation for getting shortest paths
        self.G = nx.Graph()
        self.G.add_nodes_from(self.state_dict.keys())
        for i,s in enumerate(self.state_dict.keys()):
            for a in range(4):
                self.state = s
                _,_,_,_ = self.step(a)
                s_tp1 = self.state
                self.G.add_edge(s,s_tp1)

    def random_start(self):
        self.state = random.sample(self.state_dict.keys(),1)[0]

    def step(self,action):
        # Take action
        if action == 0:
            # Try to go down
            new_state = (self.state[0]+1,self.state[1])
        elif action == 1:
            # Try to go left
            new_state = (self.state[0],self.state[1]-1)
        elif action == 2:
            # Try to go up
            new_state = (self.state[0]-1,self.state[1])
        elif action == 3:
            # Try to go right
            new_state = (self.state[0],self.state[1]+1)

        # Apply walls
        if self.rooms_array[new_state] != 1:
            new_state = self.state # hit wall, stay in the same place
        self.state = new_state
        observation = self.state_dict[self.state]

        # Reward
        if new_state == self.goal:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False

        return observation,reward,done,'no_info'

    def reset(self):
        self.random_start()
        return self.state_dict[self.state]

    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)

    def render(self):
        im = self.rooms_array.copy()
        im[self.state[0],self.state[1]] = 2
        im[self.goal[0],self.goal[1]] = 3
        plt.imshow(im)
        plt.show()

    def shortest_path_length(self,s):
        shortest_length = nx.shortest_path_length(self.G,s,self.goal)
        return shortest_length
