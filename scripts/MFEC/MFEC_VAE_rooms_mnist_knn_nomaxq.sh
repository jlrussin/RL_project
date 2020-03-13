#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G
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
--room_size 9 \
--fourrooms_state_type mnist \
--frames_to_stack 1 \
--n_episodes 20000 \
--initial_epsilon 1.0 \
--final_epsilon 0.1 \
--epsilon_decay 0.9 \
--gamma 0.99 \
--Q_train_algo MC \
--q_lr 0.1 \
--force_knn \
--weight_neighbors \
--delta 0.01 \
--agent MFEC \
--num_neighbors 9 \
--embedding_type VAE \
--vae_batch_size 4 \
--vae_train_frames 100000 \
--vae_epochs 10 \
--embedding_size 32 \
--in_height 28 \
--in_width 28 \
--max_memory 328 \
--optimizer 'RMSprop' \
--lr 1e-5 \
--print_every 20 \
--vae_print_every 100 \
--load_vae_from ../weights/VAE/VAE_rooms_mnist.pt \
--out_data_file ../results/MFEC/MFEC_VAE_rooms_mnist_knn_nomaxq.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
