#!/bin/bash
# SBATCH --account=guest
# SBATCH --partition=guest-gpu
# SBATCH --qos=low-gpu
# SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
# SBATCH --time=02:00:00
# SBATCH --job-name=gpt
# SBATCH --output=gpt.out
 
# Load modules required for your job
module load cuda/9.0
module load share_modules/HOOMD/2.3.5_sp

srun python ./code/gpt-2-fine-tune.py
