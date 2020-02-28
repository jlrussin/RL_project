#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH --nodelist=local01
#SBATCH -c 8

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
--episodes 10 \
--initial_epsilon 1.0 \
--final_epsilon 1.0 \
--epsilon_decay 1.0 \
--gamma 1.0 \
--N 3 \
--replay_buffer_size 100 \
--replay_every  4 \
--batch_size 32 \
--env_id PongNoFrameskip-v0 \
--agent NEC \
--num_neighbors 50 \
--embedding_size 64 \
--max_memory 100 \
--optimizer 'RMSprop' \
--lr 0.000001 \
--q_lr 0.01 \
--print_every 1 \
--out_data_file ../results/NEC/test_NEC_gpu.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
