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
--room_size 9 \
--fourrooms_state_type mnist \
--frames_to_stack 1 \
--training_frames 10 \
--initial_epsilon 0.1 \
--final_epsilon 0.1 \
--epsilon_decay 1.0 \
--gamma 0.99 \
--N 50 \
--replay_buffer_size 100000 \
--replay_every  4 \
--batch_size 8 \
--agent NEC \
--num_neighbors 11 \
--embedding_size 64 \
--in_height 28 \
--in_width 28 \
--max_memory 500000 \
--optimizer 'RMSprop' \
--lr 1e-6 \
--q_lr 0.01 \
--print_every 1 \
--out_data_file ../results/NEC/NEC_rooms_mnist.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
