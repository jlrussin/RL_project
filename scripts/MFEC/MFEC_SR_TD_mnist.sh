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
--room_size 21 \
--fourrooms_state_type mnist \
--frames_to_stack 1 \
--n_episodes 20000 \
--initial_epsilon 1.0 \
--final_epsilon 0.1 \
--epsilon_decay 0.9 \
--gamma 0.99 \
--SR_gamma 0.99 \
--SR_batch_size 64 \
--SR_train_frames 1000000 \
--SR_epochs 20 \
--SR_train_algo TD \
--agent MFEC \
--num_neighbors 15 \
--embedding_type SR \
--SR_embedding_type random \
--embedding_size 32 \
--in_height 28 \
--in_width 28 \
--max_memory 10000 \
--n_hidden 100 \
--lr 0.0001 \
--optimizer 'RMSprop' \
--SR_filename ../results/MFEC_SR/random_TD_mnist \
--print_every 100 \
--out_data_file ../results/MFEC_SR/MFEC_SR_rand_TD_rooms_mnist.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
