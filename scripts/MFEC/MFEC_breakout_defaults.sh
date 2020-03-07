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
--env_id BreakoutNoFrameskip-v0 \
--frames_to_stack 4 \
--training_frames 4e7 \
--initial_epsilon 0.005 \
--final_epsilon 0.005 \
--epsilon_decay 1.0 \
--gamma 1.0 \
--agent MFEC \
--num_neighbors 11 \
--embedding_type random \
--embedding_size 64 \
--max_memory 1000000 \
--optimizer 'RMSprop' \
--print_every 1 \
--out_data_file ../results/NEC/MFEC_breakout_defaults.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
