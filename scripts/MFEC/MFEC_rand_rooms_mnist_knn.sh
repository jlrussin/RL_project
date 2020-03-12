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
--initial_epsilon 0.1 \
--final_epsilon 0.1 \
--epsilon_decay 0.1 \
--gamma 0.99 \
--Q_train_algo MC \
--use_Q_max \
--force_knn \
--agent MFEC \
--num_neighbors 9 \
--embedding_type random \
--embedding_size 32 \
--in_height 28 \
--in_width 28 \
--max_memory 328 \
--print_every 20 \
--out_data_file ../results/MFEC/MFEC_rand_rooms_mnist_knn.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
