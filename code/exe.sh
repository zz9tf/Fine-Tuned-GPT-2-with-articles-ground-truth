#!/bin/bash
#SBATCH --job-name=gpt
#SBATCH --output=None
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Zheng_5732021823@outlook.com
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
 
# Activate the virtual environment if needed
$(read_env PY_ENV) --version  > ../log 2>&1

# Load modules required for your job
module load cuda/9.0
module load share_modules/HOOMD/2.3.5_sp

$(read_env PY_ENV) gpt-2-fine-tune.py >> ../log 2>&1
