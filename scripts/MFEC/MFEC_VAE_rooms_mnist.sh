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
--environment_type fourrooms \
--room_size 13 \
--fourrooms_state_type mnist \
--frames_to_stack 1 \
--training_frames 1000000 \
--initial_epsilon 0.005 \
--final_epsilon 0.005 \
--epsilon_decay 1.0 \
--gamma 1.0 \
--agent MFEC \
--num_neighbors 11 \
--embedding_type VAE \
--vae_batch_size 4 \
--vae_train_frames 100000 \
--vae_epochs 10 \
--embedding_size 32 \
--max_memory 10000 \
--optimizer 'RMSprop' \
--lr 1e-5 \
--print_every 100 \
--vae_print_every 100 \
--out_data_file ../results/NEC/MFEC_VAE_rooms_mnist.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
