#!/bin/bash

#SBATCH --job-name=super-inr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=/net/projects/willettlab/sueparkinson/deeprelu/newmiddlelinear/log/%j.out
#SBATCH --error=/net/projects/willettlab/sueparkinson/deeprelu/newmiddlelinear/log/%j.out

cd /net/projects/willettlab/sueparkinson/deeprelu/newmiddlelinear

# Activate conda env
source /home/sueparkinson/miniconda3/etc/profile.d/conda.sh
conda activate cluster_startup

echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

srun python /home/sueparkinson/deeprelu/super_inrs/phantom_exp/run_exp.py --exp="$@"

#usage: 
# sbatch submit.sh --exp=pasted_experiment_file_path