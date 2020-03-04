import random
import torch

from models.DND import DND
from models.CNN import CNN
from utils.utils import discount, inverse_distance, get_optimizer
from utils.replay_memory import Transition, ReplayMemory

class NEC:
    def __init__(self, env, args, device='cpu'):
        """
        Instantiate an NEC Agent
        ----------
        env: gym.Env
            gym environment to train on
        args: args class from argparser
            args are from from train.py: see train.py for help with each arg
        device: string
            'cpu' or 'cuda:0' depending on use_cuda flag from train.py
        """
        self.env = env
        self.device = device
        # Hyperparameters
        self.epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma
        self.N = args.N
        # Transition queue and replay memory
        self.transition_queue = []
        self.replay_every = args.replay_every
        self.replay_buffer_size = args.replay_buffer_size
        self.replay_memory = ReplayMemory(self.replay_buffer_size)
        # CNN for state embedding network
        self.embedding_size = args.embedding_size
        self.cnn = CNN(self.embedding_size).to(self.device)
        # Differentiable Neural Dictionary (DND): one for each action
        self.kernel = inverse_distance
        self.num_neighbors = args.num_neighbors
        self.max_memory = args.max_memory
        self.lr = args.lr
        self.dnd_list = []
        for i in range(env.action_space.n):
            self.dnd_list.append(DND(self.kernel, self.num_neighbors,
                                     self.max_memory, args.optimizer, self.lr))
        # Optimizer for state embedding CNN
        self.q_lr = args.q_lr
        self.batch_size = args.batch_size
        self.optimizer = get_optimizer(args.optimizer,self.cnn.parameters(),
                                       self.lr)

    def choose_action(self, state_embedding):
        """
        Choose epsilon-greedy policy according to Q-estimates from DNDs
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.action_space.n - 1)
        else:
            qs = [dnd.lookup(state_embedding) for dnd in self.dnd_list]
            action = torch.argmax(torch.cat(qs))
            return action

    def Q_lookahead(self, t, warmup=False):
        """
        Return the N-step Q-value lookahead from time t in the transition queue
        """
        if warmup or len(self.transition_queue) <= t + self.N:
            lookahead = [tr.reward for tr in self.transition_queue[t:]]
            discounted = discount(lookahead, self.gamma)
            Q_N = torch.tensor([discounted],requires_grad=True)
            return Q_N
        else:
            lookahead = [tr.reward for tr in self.transition_queue[t:t+self.N]]
            discounted = discount(lookahead, self.gamma)
            state = self.transition_queue[t+self.N].state
            state = torch.tensor(state).permute(2,0,1).unsqueeze(0) # (N,C,H,W)
            state = state.to(self.device)
            state_embedding = self.cnn(state)
            Q_a = [dnd.lookup(state_embedding) for dnd in self.dnd_list]
            maxQ = torch.cat(Q_a).max()
            Q_N = discounted + (self.gamma ** self.N) * maxQ
            Q_N = torch.tensor([Q_N],requires_grad=True)
            return Q_N

    def Q_update(self, Q, Q_N):
        """
        Return the Q-update for DND updates
        """
        return Q + self.q_lr * (Q_N - Q)

    def update(self):
        """
        Iterate through the transition queue and make NEC updates
        """
        # Insert transitions into DNDs
        for t in range(len(self.transition_queue)):
            tr = self.transition_queue[t]
            action = tr.action
            tr = self.transition_queue[t]
            state = torch.tensor(tr.state).permute(2,0,1) # (C,H,W)
            state = state.unsqueeze(0).to(self.device) # (N,C,H,W)
            state_embedding = self.cnn(state)
            dnd = self.dnd_list[action]

            Q_N = self.Q_lookahead(t).to(self.device)
            embedding_index = dnd.get_index(state_embedding)
            if embedding_index is None:
                dnd.insert(state_embedding.detach(), Q_N.detach().unsqueeze(0))
            else:
                Q = self.Q_update(dnd.values[embedding_index], Q_N)
                dnd.update(Q.detach(), embedding_index)
            Q_N = Q_N.detach().to(self.device)
            self.replay_memory.push(tr.state, action, Q_N)
        # Commit inserts
        for dnd in self.dnd_list:
            dnd.commit_insert()
        # Train CNN on minibatch
        for t in range(len(self.transition_queue)):
            if t % self.replay_every == 0 or t == len(self.transition_queue)-1:
                # Train on random mini-batch from self.replay_memory
                batch = self.replay_memory.sample(self.batch_size)
                actual_Qs = torch.cat([sample.Q_N for sample in batch])
                predicted_Qs = []
                for sample in batch:
                    state = torch.tensor(sample.state).permute(2,0,1) # (C,H,W)
                    state = state.unsqueeze(0).to(self.device) # (N,C,H,W)
                    state_embedding = self.cnn(state)
                    dnd = self.dnd_list[sample.action]
                    predicted_Q = dnd.lookup(state_embedding,update_flag=True)
                    predicted_Qs.append(predicted_Q)
                predicted_Qs = torch.cat(predicted_Qs).to(self.device)
                loss = torch.dist(actual_Qs, predicted_Qs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                for dnd in self.dnd_list:
                    dnd.update_params()

        # Clear out transition queue
        self.transition_queue = []

    def run_episode(self):
        """
        Train an NEC agent for a single episode:
            Interact with environment
            Append (state, action, reward) transitions to transition queue
            Call update at the end of the episode
        """
        if self.epsilon > self.final_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay
        state = self.env.reset()
        total_reward = 0
        total_frames = 0
        done = False
        while not done:
            state_embedding = torch.tensor(state).permute(2,0,1) # (C,H,W)
            state_embedding = state_embedding.unsqueeze(0).to(self.device)
            state_embedding = self.cnn(state_embedding)
            action = self.choose_action(state_embedding)
            next_state, reward, done, _ = self.env.step(action)
            self.transition_queue.append(Transition(state, action, reward))
            total_reward += reward
            total_frames += self.env.skip
            state = next_state
        self.update()
        return total_frames,total_reward

    def warmup(self):
        """
        Warmup the DND with values from an episode with a random policy
        """
        state = self.env.reset()
        total_reward = 0
        total_frames = 0
        done = False
        while not done:
            action = random.randint(0, self.env.action_space.n - 1)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            total_frames += self.env.skip
            self.transition_queue.append(Transition(state, action, reward))
            state = next_state

        for t in range(len(self.transition_queue)):
            tr = self.transition_queue[t]
            state_embedding = torch.tensor(tr.state).permute(2,0,1) # (C,H,W)
            state_embedding = state_embedding.unsqueeze(0).to(self.device)
            state_embedding = self.cnn(state_embedding)
            action = tr.action
            dnd = self.dnd_list[action]

            Q_N = self.Q_lookahead(t, True).to(self.device)
            if dnd.keys_to_be_inserted is None and dnd.keys is None:
                dnd.insert(state_embedding, Q_N.detach().unsqueeze(0))
            else:
                embedding_index = dnd.get_index(state_embedding)
                if embedding_index is None:
                    state_embedding = state_embedding.detach()
                    dnd.insert(state_embedding,Q_N.detach().unsqueeze(0))
                else:
                    Q = self.Q_update(dnd.values[embedding_index], Q_N)
                    dnd.update(Q.detach(), embedding_index)
            self.replay_memory.push(tr.state, action, Q_N.detach())
        for dnd in self.dnd_list:
            dnd.commit_insert()
        # Clear out transition queue
        self.transition_queue = []
        return total_frames, total_reward
