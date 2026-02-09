#!/bin/bash

#SBATCH --job-name=wandb
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=/net/projects/willettlab/sueparkinson/deeprelu/newmiddlelinear/log/%j.out
#SBATCH --error=/net/projects/willettlab/sueparkinson/deeprelu/newmiddlelinear/log/%j.out
#SBATCH --mem=16G

cd /net/projects/willettlab/sueparkinson/deeprelu/newmiddlelinear

# Activate your conda env
source /home/sueparkinson/miniconda3/etc/profile.d/conda.sh
conda activate cluster_startup

echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

srun "$@"

#usage: 
# sbatch srun.sh __pasted_launch_agent_command__
# sbatch srun.sh command_to_run_on_server
# can do run the same command multiple times from the login node to do the sweep on multiple GPUs