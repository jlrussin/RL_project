#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
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
--seed 1 \
--env_id PongNoFrameskip-v0 \
--frames_to_stack 4 \
--episodes 1000 \
--initial_epsilon 1.0 \
--final_epsilon 0.01 \
--epsilon_decay 0.99 \
--gamma 1.0 \
--vae_batch_size 8 \
--vae_train_frames 1000000 \
--vae_epochs 10 \
--agent MFEC \
--num_neighbors 50 \
--embedding_type VAE \
--embedding_size 64 \
--max_memory 500000 \
--optimizer 'RMSprop' \
--lr 1e-5 \
--vae_print_every 100 \
--print_every 1 \
--out_data_file ../results/NEC/MFEC_pong_defaults.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
