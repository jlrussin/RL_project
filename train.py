import argparse
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.NEC import *
from models.DND import *
from utils.atari_wrappers import make_atari, wrap_deepmind
from utils.utils import inverse_distance

# Things to do:
#   -Should allow for training and loading of model (i.e. saving whole DND?)
#       -Need save/checkpoint and load options
#   -Should do checkpointing? (i.e. testing every once and a while?)
#   -Should do printing? (i.e. printing average score every once in a while?)

parser = argparse.ArgumentParser()
# CUDA
parser.add_argument('--use_cuda', action='store_true',
                    help='Use GPU, if available')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
# Environment
parser.add_argument('--env_id', default='PongNoFrameskip-v0',
                    choices=['PongNoFrameskip-v0'],
                    help='OpenAI gym name for Atari env to use for training')
parser.add_argument('--frames_to_stack', type=int, default=4,
                    help='Number of prev. frames to fold into current state')
# Training
parser.add_argument('--episodes', type=int, default=10000,
                    help='Number of episodes for training')
parser.add_argument('--initial_epsilon', type=float, default=1.0,
                    help='Initial probability of selecting random action')
parser.add_argument('--final_epsilon', type=float, default=0.01,
                    help='Final probability of selecting random action')
parser.add_argument('--epsilon_decay', type=float, default=0.99,
                    help='Decay for probability of selecting random action')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Temporal discounting parameter')
parser.add_argument('--N', type=int, default=100,
                    help='Horizon for N-step Q-estimates')
parser.add_argument('--replay_buffer_size', type=int, default=100000,
                    help='Number of states to store in the replay buffer')
parser.add_argument('--replay_every', type=int, default=16,
                    help='Number of observed frames before replaying')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Minibatch size for replay update')
parser.add_argument('--vae_batch_size', type=int, default=32,
                    help='Minibatch size for vae training')
parser.add_argument('--vae_train_frames', type=int, default=1000000,
                    help='Number of frames to train VAE on')
parser.add_argument('--vae_epochs', type=int, default=10,
                    help='Number of epochs for training VAE on frames')
# Model
parser.add_argument('--agent', choices=['NEC','MFEC'],
                    help='Type of agent to use')
parser.add_argument('--num_neighbors', type=int, default=50,
                    help='Number of nearest neighbors used for lookup')
parser.add_argument('--embedding_size', type=int, default=64,
                    help='Dimension of state embeddings (default from mjacar)')
parser.add_argument('--max_memory', type=int, default=500000,
                    help='Maximum number of memories in DND')
parser.add_argument('--load_vae_from',default=None,
                    help='Path to file to load vae weights from')
# Optimization
parser.add_argument('--optimizer', choices=['Adam','RMSprop'],
                    default='RMSprop',
                    help='Optimizer to use for training')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='Learning rate of optimizer (default from mjacar)')
parser.add_argument('--q_lr', type=float, default=0.01,
                    help='Learning rate for Q-values (default from mjacar)')
# Output options
parser.add_argument('--print_every', type=int, default=1000,
                    help='Number of episodes before printing some score data')
parser.add_argument('--vae_weights_file', default=None,
                    help='Path to file to save vae weights')
parser.add_argument('--out_data_file', default='../results/NEC/results.npy',
                    help='Path to output data file with score history')

def main(args):
    # CUDA
    if args.use_cuda:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        use_cuda = False
        device = "cpu"
    print("Using cuda: ", use_cuda)

    # Environment
    env = make_atari(args.env_id)
    env = wrap_deepmind(env,args.frames_to_stack,scale=True)

    # Random seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # Agent
    if args.agent == 'MFEC':
        agent = MFEC(env,args,device)
    elif args.agent == 'NEC':
        agent = NEC(env,args,device)

    # Pretraining: autoencoder in MFEC or DND warmup in NEC
    agent.warmup()

    # Training loop
    time_history = [] # records time (in sec) of each episode
    num_frames_history = [] # records the number of frames of each episode
    score_history = [] # records total score of each episode
    for episode in range(args.episodes):
        start_time = time.time()
        num_frames,score = agent.run_episode()
        time_history.append(time.time() - start_time)
        num_frames_history.append(num_frames)
        score_history.append(score)
        if episode % args.print_every == 0:
            print("Episode:", episode,
                  "Score:",score_history[-1],
                  "Average score:", np.mean(score_history),
                  "Frames:",num_frames,
                  "Time:",time_history[-1])
    print("Average time per episode:", np.mean(time_history))
    print("Total number of frames:", np.sum(num_frames_history))

    # Testing loop
    # TODO: test with smaller epsilon, no random starting actions, etc.?
    # TODO: can also record rendered frames of a few episodes?

    # Save score history to file
    scores_arr = np.array(score_history)
    frames_arr = np.array(num_frames_history)
    data_arr = np.stack([scores_arr,frames_arr],1)
    np.save(args.out_data_file,data_arr)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
