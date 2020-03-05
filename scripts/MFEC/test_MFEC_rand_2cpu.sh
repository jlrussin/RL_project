#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH --nodelist=local01
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
--episodes 1 \
--initial_epsilon 0.1 \
--final_epsilon 0.1 \
--epsilon_decay 1.0 \
--gamma 1.0 \
--vae_batch_size 32 \
--vae_train_frames 10 \
--vae_epochs 1 \
--agent MFEC \
--num_neighbors 2 \
--embedding_type 'random' \
--embedding_size 16 \
--max_memory 100 \
--optimizer 'RMSprop' \
--lr 0.000001 \
--print_every 1 \
--out_data_file ../results/MFEC/test_MFEC_rand_cpu.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
