import random
import torch
from torch.utils.data import TensorDataset,DataLoader

from DND import DND
from VAE import VAE, VAELoss
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
            -Still not sure which args you'll need, but careful not to delete
             anything used for the VAE or for warmup()
            -I have below initialized a list of DND's - this should be changed
             to a list of QECs right?
        """
        self.env = env
        self.frames_to_stack = args.frames_to_stack
        self.device = device
        # Hyperparameters
        self.epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma

        # Autoencoder for state embedding network
        self.batch_size = args.vae_batch_size # batch size for training VAE
        self.vae_epochs = args.vae_epochs # number of epochs to run VAE
        self.embedding_type = args.embedding_type
        self.embedding_size = args.embedding_size
        if self.embedding_type == 'VAE':
            self.batch_size = args.batch_size
            self.vae_loss = VAELoss()
            self.print_every = args.print_every
            self.load_vae_from = args.load_vae_from
            self.vae_weights_file = args.vae_weights_file
            self.vae = VAE(self.frames_to_stack,self.embedding_size)
            self.vae = self.vae.to(self.device)
            self.optimizer = get_optimizer(args.optimizer,
                                           self.autoencoder.parameters(),
                                           self.lr)
        elif self.embedding_type == 'random':
            # TODO:  make random projection matrix
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
        Choose epsilon-greedy policy according to Q-estimates
        """

    def run_episode(self):
        """
        Train an MFEC agent for a single episode:
            Interact with environment
            Perform update

        Pseudocode:
            1. state = self.env.reset()
            2. done = False
            3. while not done:
                a. if self.embedding_type == 'random':
                    i. convert state to np array
                    ii. get embedding from random projection matrix
                b. elif self.embedding_type == 'VAE':
                    i. convert state to torch tensor
                    ii. get embedding from vae with vae.encoder(state)
                        -Should use "with torch.no_grad():" for speedup?
                    iii. convert embeddinng to np array
                c. action = choose_action(state_embedding)
                d. next_state, reward, done, _ = self.env.step(action)

        Note: Above pseudocode doesn't include code for updating qec, etc.
        """


    def warmup(self):
        """
        Collect 1 million frames from random policy and train VAE
        """
        if self.embedding_type == 'VAE':
            if self.load_vae_from is not None:
                self.vae.load_state_dict(torch.load(self.load_vae_from))
                self.vae = self.vae.to(self.device)
            else:
                # Collect 1 million frames from random policy
                print("Generating dataset to train VAE from random policy")
                vae_data = []
                state = self.env.reset()
                total_frames = 0
                while total_frames < 1000000:
                    action = random.randint(0, self.env.action_space.n - 1)
                    state, reward, done, _ = self.env.step(action)
                    vae_data.append(state)
                    total_frames += self.env.skip
                    if done:
                        state = self.env.reset()
                # Dataset, Dataloader for 1 million frames
                vae_data = torch.tensor(vae_data) # (N x H x W x C)
                vae_data = vae_data.permute(0,3,1,2) # (N x C x H x W)
                vae_dataset = TensorDataset(vae_data)
                vae_dataloader = DataLoader(vae_dataset,
                                            batch_size=self.vae_batch_size,
                                            shuffle=True)
                # Training loop
                print("Training VAE")
                self.vae.train()
                for epoch in range(self.vae_epochs):
                    train_loss = 0
                    for batch_idx,batch in enumerate(vae_dataloader):
                        batch = batch[0].to(self.device)
                        self.optimizer.zero_grad()
                        recon_batch, mu, logvar = self.vae(batch)
                        loss = self.vae_loss(recon_batch,batch,mu,logvar)
                        train_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        if batch_idx % self.print_every == 0:
                            msg = 'VAE Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                                epoch,
                                batch_idx * len(batch),
                                len(vae_dataloader.dataset),
                                loss.item() / len(batch))
                            print(msg)
                    print('====> Epoch {} Average loss: {:.4f}'.format(
                        epoch, train_loss / len(vae_dataloader.dataset)))
                    if self.vae_weights_file is not None:
                        torch.save(self.vae.state_dict(),
                                   self.vae_weights_file)
            self.vae.eval()
        else:
            pass # random projection doesn't require warmup
