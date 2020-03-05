import random
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

from models.VAE import VAE, VAELoss
from models.QEC import QEC
from utils.utils import get_optimizer

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
        self.actions=range(self.env.action_space.n)
        self.frames_to_stack = args.frames_to_stack
        self.device = device
        self.rs = np.random.RandomState(args.seed)

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
            self.vae_train_frames = args.vae_train_frames
            self.vae_loss = VAELoss()
            self.print_every = args.print_every
            self.load_vae_from = args.load_vae_from
            self.vae_weights_file = args.vae_weights_file
            self.vae = VAE(self.frames_to_stack,self.embedding_size)
            self.vae = self.vae.to(self.device)
            self.lr = args.lr
            self.optimizer = get_optimizer(args.optimizer,
                                           self.vae.parameters(),
                                           self.lr)
        elif self.embedding_type == 'random':
            self.projection = self.rs.randn(
                self.embedding_size, 84 * 84 * self.frames_to_stack
            ).astype(np.float32)

        # QEC
        self.max_memory = args.max_memory
        self.num_neighbors = args.num_neighbors
        self.qec = QEC(self.actions, self.max_memory, self.num_neighbors)

        #self.state = np.empty(self.embedding_size, self.projection.dtype)
        #self.action = int
        self.time = 0
        self.memory = []

    def choose_action(self, state_embedding):
        """
        Choose epsilon-greedy policy according to Q-estimates
        """
        self.time += 1

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            self.action = self.rs.choice(self.actions)

        # Exploitation
        else:
            values = [
                self.qec.estimate(self.state, action)
                for action in self.actions
            ]
            best_actions = np.argwhere(values == np.max(values)).flatten()
            self.action = self.rs.choice(best_actions)

        return self.action

    def update(self):
            value = 0.0
            for _ in range(len(self.memory)):
                experience = self.memory.pop()
                value = value * self.gamma + experience["reward"]
                self.qec.update(
                    experience["state"],
                    experience["action"],
                    value,
                    experience["time"],
                )

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
        RENDER_SPEED = 0.04
        RENDER = False

        episode_frames = 0
        total_reward = 0

        #self.env.seed(random.randint(0, 1000000))
        self.state = self.env.reset()

        if self.embedding_type == 'random':
            self.state = np.dot(self.projection, np.array(self.state).flatten())
        elif self.embedding_type == 'VAE':
            self.state = torch.tensor(self.state).permute(2,0,1)#(H,W,C)->(C,H,W)
            self.state = self.state.unsqueeze(0).to(self.device)
            with torch.no_grad():
                mu, logvar = self.vae.encoder(self.state)
                self.state = torch.cat([mu, logvar],1)
                self.state = self.state.squeeze() # not sure to do this
                self.state = self.state.cpu().numpy()

        done = False
        while not done:

            if RENDER:
                self.env.render()
                time.sleep(RENDER_SPEED)

            action = self.choose_action(self.state)
            state, reward, done, _ = self.env.step(action)
            self.receive_reward(reward)

            total_reward += reward
            episode_frames += self.env.skip

        self.update()
        return episode_frames, total_reward


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
                while total_frames < self.vae_train_frames:
                    action = random.randint(0, self.env.action_space.n - 1)
                    state, reward, done, _ = self.env.step(action)
                    vae_data.append(state)
                    total_frames += self.env.skip
                    if done:
                        state = self.env.reset()
                # Dataset, Dataloader for 1 million frames
                vae_data = torch.tensor(vae_data) # (N x H x W x C) - (1mill/skip X 84 X 84 X frames_to_stack)
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
