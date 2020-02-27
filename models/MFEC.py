import random
import torch

from DND import DND
from VAE import VAE
from utils.utils import discount, inverse_distance, get_optimizer

class MFEC:
    def __init__(self, env, args, device='cpu'):
        """
        Instantiate an MFEC Agent
        ----------
        env: gym.Env
            gym environment to train on
        args: args class from argparser
            args are from from train.py: see train.py for help with each arg
        device: string
            'cpu' or 'cuda:0' depending on use_cuda flag from train.py

        Notes (Jake):
            -I'm not sure which of these args you'll need - I just copied these
             from NEC and made a few changes based on what I thought you'd need.
            -And I think of course you'll need to add more args here, and to the
             train.py argparser (e.g. random projection matrix vs. VAE, etc.)
            -I'm not sure if the distance metric to perform lookups (i.e.
             self.kernel below) should be inverse_distance or not
            -I haven't added a flag to the DND class indicating whether the keys
             in the DND should be treated as torch.Parameter() yet. So once I
             add that I'll have to remember to change the initialization of
             dnd_list below.
        """
        self.env = env
        self.device = device
        # Hyperparameters
        self.epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma

        # Autoencoder for state embedding network
        self.embedding_size = args.embedding_size
        self.autoencoder = VAE(self.embedding_size).to(self.device)
        # Optimizer for training autoencoder
        self.batch_size = args.batch_size
        self.optimizer = get_optimizer(args.optimizer,
                                       self.autoencoder.parameters(),
                                       self.lr)

        # Differentiable Neural Dictionary (DND): one for each action
        self.kernel = inverse_distance
        self.num_neighbors = args.num_neighbors
        self.max_memory = args.max_memory
        self.lr = args.lr
        self.dnd_list = []
        for i in range(env.action_space.n):
            self.dnd_list.append(DND(self.kernel, self.num_neighbors,
                                     self.max_memory, args.optimizer, self.lr))
        self.transition_queue = []

    def choose_action(self, state_embedding):
        """
        Choose epsilon-greedy policy according to Q-estimates from DNDs
        """

    def run_episode(self):
        """
        Train an MFEC agent for a single episode:
            Interact with environment
            Perform update
        """

    def warmup(self):
        """
        Warmup DNDs with values from one episode with random policy (as in NEC)
        Collect 1 million frames from random policy to train autoencoder
            -Note (Jake): it might actually be better to do this offline with a
                   separate script. That way you won't have to do it every time
                   you're trying new hyperparameters for the agent.
        Train autoencoder
            -Note (Jake): You could do this offline as well. Although,it might be
                   better to include it here because it will be hard to tell how
                   "good" the representations are, aside from just looking at
                   the loss over the 1-million frame training.
        """
