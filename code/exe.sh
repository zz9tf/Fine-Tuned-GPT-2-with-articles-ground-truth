#!/bin/bash
# SBATCH --job-name=gpt
# SBATCH --output=None
# SBATCH --mail-type=BEGIN, FAIL, END
# SBATCH --mail-user=zz9tf@umsystem.edu
# SBATCH --partition=guest-gpu
# SBATCH --qos=low-gpu
# SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
# SBATCH --time=02:00:00
 
# Load modules required for your job
module load cuda/9.0
module load share_modules/HOOMD/2.3.5_sp

python ./code/gpt-2-fine-tune.py > ../log 2>&1
