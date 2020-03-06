#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
for gpu in $gpus
do
echo "Setting fan for" $gpu "to full"
nvidia_fancontrol full $gpu
done

python train.py \
--use_cuda \
--seed 1 \
--env_id PongNoFrameskip-v0 \
--frames_to_stack 4 \
--episodes 1000 \
--initial_epsilon 1.0 \
--final_epsilon 0.01 \
--epsilon_decay 0.99 \
--gamma 0.99 \
--N 100 \
--replay_buffer_size 100000 \
--replay_every  4 \
--batch_size 8 \
--agent NEC \
--num_neighbors 50 \
--embedding_size 64 \
--max_memory 500000 \
--optimizer 'RMSprop' \
--lr 1e-6 \
--q_lr 0.01 \
--print_every 1 \
--out_data_file ../results/NEC/NEC_pong_defaults.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
