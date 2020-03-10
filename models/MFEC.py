import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

from models.VAE import VAE, VAELoss
from models.QEC import QEC
from utils.utils import get_optimizer
from utils.SRDataset import *

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
        """
        self.environment_type = args.environment_type
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
        self.vae_batch_size = args.vae_batch_size # batch size for training VAE
        self.vae_epochs = args.vae_epochs # number of epochs to run VAE
        self.embedding_type = args.embedding_type
        self.SR_embedding_type = args.SR_embedding_type
        self.embedding_size = args.embedding_size
        self.in_height = args.in_height
        self.in_width = args.in_width

        if self.embedding_type == 'VAE':
            self.vae_train_frames = args.vae_train_frames
            self.vae_loss = VAELoss()
            self.vae_print_every = args.vae_print_every
            self.load_vae_from = args.load_vae_from
            self.vae_weights_file = args.vae_weights_file
            self.vae = VAE(self.frames_to_stack,self.embedding_size,
                           self.in_height,self.in_width)
            self.vae = self.vae.to(self.device)
            self.lr = args.lr
            self.optimizer = get_optimizer(args.optimizer,
                                           self.vae.parameters(),
                                           self.lr)
        elif self.embedding_type == 'random':
            self.projection = self.rs.randn(
                self.embedding_size, self.in_height * self.in_width * self.frames_to_stack
            ).astype(np.float32)
        elif self.embedding_type == 'SR':
            self.SR_train_algo = args.SR_train_algo
            self.SR_gamma = args.SR_gamma
            self.n_hidden = args.n_hidden
            self.SR_train_frames = args.SR_train_frames
            self.SR_filename = args.SR_filename
            if self.SR_embedding_type == 'random':
                self.projection = np.random.randn(
                                        self.embedding_size, self.in_height * self.in_width
                                        ).astype(np.float32)
                if self.SR_train_algo == 'TD':
                    self.mlp=MLP(self.embedding_size,self.n_hidden)
                    self.loss_fn = nn.MSELoss(reduction='sum')
                    params=self.mlp.parameters()
                    self.optimizer = get_optimizer(args.optimizer, params, self.lr)

        # QEC
        self.max_memory = args.max_memory
        self.num_neighbors = args.num_neighbors
        self.qec = QEC(self.actions, self.max_memory, self.num_neighbors)

        #self.state = np.empty(self.embedding_size, self.projection.dtype)
        #self.action = int
        self.memory = []

    def choose_action(self, state_embedding):
        """
        Choose epsilon-greedy policy according to Q-estimates
        """

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            self.action = self.rs.choice(self.actions)

        # Exploitation
        else:
            values = [
                self.qec.estimate(state_embedding, action)
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

    def add_to_memory(self, state_embedding, action, reward, time):
            self.memory.append(
                {
                    "state": state_embedding,
                    "action": action,
                    "reward": reward,
                    "time": time,
                }
            )

    def run_episode(self):
        """
        Train an MFEC agent for a single episode:
            Interact with environment
            Perform update
        """
        RENDER_SPEED = 0.04
        RENDER = False

        episode_frames = 0
        total_reward = 0
        total_steps = 0

        # Update epsilon
        if self.epsilon > self.final_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay

        #self.env.seed(random.randint(0, 1000000))
        state = self.env.reset()
        if self.environment_type == 'fourrooms':
            fewest_steps = self.env.shortest_path_length(self.env.state)
        done = False
        time = 0
        while not done:
            time += 1
            if self.embedding_type == 'random':
                state = np.array(state).flatten()
                state_embedding = np.dot(self.projection,state)
            elif self.embedding_type == 'VAE':
                state = torch.tensor(state).permute(2,0,1) #(H,W,C)->(C,H,W)
                state = state.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    mu, logvar = self.vae.encoder(state)
                    state_embedding = torch.cat([mu, logvar],1)
                    state_embedding = state_embedding.squeeze()
                    state_embedding = state_embedding.cpu().numpy()
            elif self.embedding_type == 'SR':
                if self.SR_train_algo == 'TD':
                    state = np.array(state).flatten()
                    state_embedding = np.dot(self.projection,state)
                    with torch.no_grad():
                        state_embedding = self.mlp(torch.tensor(state_embedding)).numpy()
                elif self.SR_train_algo == 'DP':
                    s = self.env.state
                    state_embedding = self.true_SR_dict[s]
            if RENDER:
                self.env.render()
                time.sleep(RENDER_SPEED)
            action = self.choose_action(state_embedding)
            state, reward, done, _ = self.env.step(action)
            self.add_to_memory(state_embedding,action,reward,time)

            total_reward += reward
            total_steps += 1
            episode_frames += self.env.skip

        self.update()
        if self.environment_type == 'fourrooms':
            n_extra_steps = total_steps - fewest_steps
            return n_extra_steps, episode_frames, total_reward
        else:
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
                        self.optimizer.step()
                        if batch_idx % self.vae_print_every == 0:
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
        elif self.embedding_type == 'SR':
            if self.SR_embedding_type == 'random':
                if self.SR_train_algo == 'TD':
                    total_frames=0
                    transitions=[]
                    while total_frames < self.SR_train_frames:
                        observation = self.env.reset()
                        s_t = self.env.state # will not work on Atari
                        done = False
                        while not done:
                            action = np.random.randint(env.action_space.n)
                            observation, reward, done, _ = env.step(action)
                            s_tp1 = env.state # will not work on Atari
                            transitions.append((s_t, s_tp1))
                            total_frames += env.skip
                            s_t = s_tp1
                    # Dataset, Dataloader
                    dataset = SRDataset(env,projection,transitions)
                    dataloader = DataLoader(dataset,batch_size=SR_batch_size,shuffle=True)
                    train_losses=[]
                    #Training loop
                    for epoch in range(self.SR_epochs):
                        for batch_idx,batch in enumerate(dataloader):
                            self.optimizer.zero_grad()
                            e_t,e_tp1 = batch
                            mhat_t = self.mlp(e_t)
                            mhat_tp1 = self.mlp(e_tp1)
                            target = e_t + self.gamma*mhat_tp1.detach()
                            loss = self.loss_fn(mhat_t,target)
                            loss.backward()
                            self.optimizer.step()
                            train_losses.append(loss.item())
                        print("Epoch:",epoch,"Average loss",np.mean(train_losses))

                    emb_reps = np.zeros([self.env.n_states,self.embedding_size])
                    SR_reps = np.zeros([self.env.n_states,self.embedding_size])
                    labels = []
                    room_size=self.env.room_size
                    for i,(state,obs) in enumerate(self.env.state_dict.items()):
                        emb = np.dot(projection,obs.flatten())
                        emb_reps[i,:] = emb
                        with torch.no_grad():
                            SR = self.mlp(torch.tensor(emb)).numpy()
                        SR_reps[i,:] = SR
                        if state[0] < room_size + 1 and state[1] < room_size + 1:
                            label = 0
                        elif state[0] > room_size + 1 and state[1] < room_size + 1:
                            label = 1
                        elif state[0] < room_size + 1 and state[1] > room_size + 1:
                            label = 2
                        elif state[0] > room_size + 1 and state[1] > room_size + 1:
                            label = 3
                        else:
                            label = 4
                        labels.append(label)
                    np.save('%s_SR_reps.npy' %(self.SR_filename), SR_reps)
                    np.save('%s_emb_reps.npy' %(self.SR_filename), emb_reps)
                    np.save('%s_labels.npy' %(self.SR_filename), labels)
                elif self.SR_train_algo == 'MC':
                    pass
                elif self.SR_train_algo == 'DP':
                    # Use this to ensure same order every time
                    idx_to_state = {i:state for i,state in enumerate(self.env.state_dict.keys())}
                    state_to_idx = {v:k for k,v in idx_to_state.items()}
                    T = np.zeros([self.env.n_states,self.env.n_states])
                    for i,s in idx_to_state.items():
                        for a in range(4):
                            self.env.state = s
                            _,_,_,_ = self.env.step(a)
                            s_tp1 = self.env.state
                            T[state_to_idx[s],state_to_idx[s_tp1]] += 0.25
                    true_SR = np.eye(self.env.n_states)
                    done = False
                    t = 0
                    while not done:
                        t += 1
                        new_SR = true_SR + (self.SR_gamma**t)*(np.matmul(true_SR,T))
                        done = np.max(np.abs(true_SR - new_SR)) < 1e-10
                        true_SR = new_SR
                    self.true_SR_dict = {}
                    for s,obs in self.env.state_dict.items():
                        idx = state_to_idx[s]
                        self.true_SR_dict[s] = true_SR[idx,:]
        else:
            pass # random projection doesn't require warmup
